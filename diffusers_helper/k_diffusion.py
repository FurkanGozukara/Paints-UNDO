import torch
import numpy as np
import time

from tqdm import tqdm


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, progress_tqdm=None):
    """DPM-Solver++(2M)."""
    print(f"[TRACE] Entering sample_dpmpp_2m function")
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None

    bar = tqdm if progress_tqdm is None else progress_tqdm

    print(f"\n[DEBUG] Starting sampling loop with {len(sigmas) - 1} steps")
    print(f"[DEBUG] Latent shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
    print(f"[DEBUG] Effective batch size with CFG: {x.shape[0] * 2} (UNet processes {x.shape[0]} positive + {x.shape[0]} negative)")
    
    # Calculate theoretical workload
    total_pixels = x.shape[0] * 2 * x.shape[2] * x.shape[3]  # batch * height * width (with CFG)
    print(f"[DEBUG] Workload per step: {total_pixels} pixels ({x.shape[2]}x{x.shape[3]} latent resolution)")
    
    if total_pixels < 50000:
        print(f"[WARNING] Small workload detected! GPU may be underutilized.")
        print(f"[WARNING] To increase GPU usage, try:")
        print(f"[WARNING]   - Increase image resolution (current: {x.shape[2]*8}x{x.shape[3]*8} pixels)")
        print(f"[WARNING]   - Add more undo steps (current batch: {x.shape[0]})")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print(f"[TRACE] Starting denoising loop...")
    for i in bar(range(len(sigmas) - 1)):
        print(f"[TRACE] Step {i} - preparing input...", end='', flush=True)
        step_start = time.time()
        
        print(f" calling model...", end='', flush=True)
        model_start = time.time()
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        model_time = time.time() - model_start
        print(f" done ({model_time:.3f}s)", flush=True)
        
        step_time = time.time() - step_start
        
        if i == 0:
            print(f"[DEBUG] Step {i}/{len(sigmas)-1} took {step_time:.3f}s (model: {model_time:.3f}s, first step includes compilation)")
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated(0) / 1024**3
                print(f"[DEBUG] Peak GPU memory usage: {peak_mem:.2f}GB")
                if peak_mem < 8.0:
                    print(f"[INFO] Low memory usage indicates small workload - GPU not fully utilized")
        elif i % 5 == 0:
            print(f"[DEBUG] Step {i}/{len(sigmas)-1} - total: {step_time:.3f}s, model: {model_time:.3f}s")
            
        print(f"[TRACE] Step {i} - processing callback and updates...", end='', flush=True)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        print(f" done", flush=True)
        
    print(f"[TRACE] Denoising loop completed")
    return x


class KModel:
    def __init__(self, unet, timesteps=1000, linear_start=0.00085, linear_end=0.012, linear=False):
        if linear:
            betas = torch.linspace(linear_start, linear_end, timesteps, dtype=torch.float64)
        else:
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, timesteps, dtype=torch.float64) ** 2

        alphas = 1. - betas
        alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)

        self.sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        self.log_sigmas = self.sigmas.log()
        self.sigma_data = 1.0
        self.unet = unet
        return

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape).to(sigma.device)

    def get_sigmas_karras(self, n, rho=7.):
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros([1])])

    def __call__(self, x, sigma, **extra_args):
        print(f"[TRACE] KModel.__call__ - input shape: {x.shape}", end='', flush=True)
        
        prep_start = time.time()
        x_ddim_space = x / (sigma[:, None, None, None] ** 2 + self.sigma_data ** 2) ** 0.5
        x_ddim_space = x_ddim_space.to(dtype=self.unet.dtype)
        t = self.timestep(sigma)
        cfg_scale = extra_args['cfg_scale']
        print(f" prep: {(time.time()-prep_start)*1000:.1f}ms", end='', flush=True)
        
        # Debug: Check if UNet is on GPU
        if not hasattr(self, '_debug_printed'):
            unet_device = next(self.unet.parameters()).device
            print(f"\n[DEBUG] UNet device: {unet_device}, dtype: {next(self.unet.parameters()).dtype}")
            print(f"[DEBUG] Input device: {x_ddim_space.device}, dtype: {x_ddim_space.dtype}")
            print(f"[DEBUG] Batch size: {x.shape[0]}, CFG scale: {cfg_scale}")
            
            # Check GPU memory
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"[DEBUG] GPU Memory - Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB, Total: {mem_total:.2f}GB")
            print(f"[DEBUG] Using separate positive/negative calls (hooked UNet requires this)")
            
            self._debug_printed = True
        
        # Verify everything is on GPU before UNet call
        if not hasattr(self, '_device_check_done'):
            print(f"\n[DEVICE CHECK] x_ddim_space: {x_ddim_space.device}")
            print(f"[DEVICE CHECK] t: {t.device}")
            print(f"[DEVICE CHECK] UNet parameters: {next(self.unet.parameters()).device}")
            print(f"[DEVICE CHECK] encoder_hidden_states: {extra_args['positive']['encoder_hidden_states'].device}")
            if extra_args['positive'].get('cross_attention_kwargs'):
                for k, v in extra_args['positive']['cross_attention_kwargs'].items():
                    if isinstance(v, torch.Tensor):
                        print(f"[DEVICE CHECK] cross_attention_kwargs[{k}]: {v.device}")
            self._device_check_done = True
        
        # REVERT TO ORIGINAL: Two separate calls work better with the hooked UNet
        # The hooks (cat_cond, code_cond) don't play well with manual batching
        print(f" UNet pos...", end='', flush=True)
        unet_start = time.time()
        
        # Ensure CUDA operations are synchronous for accurate timing
        if x_ddim_space.is_cuda:
            torch.cuda.synchronize()
        
        eps_positive = self.unet(x_ddim_space, t, return_dict=False, **extra_args['positive'])[0]
        
        if x_ddim_space.is_cuda:
            torch.cuda.synchronize()
        
        pos_time = time.time() - unet_start
        print(f"{pos_time*1000:.1f}ms", end='', flush=True)
        
        print(f" neg...", end='', flush=True)
        unet_start = time.time()
        
        if x_ddim_space.is_cuda:
            torch.cuda.synchronize()
        
        eps_negative = self.unet(x_ddim_space, t, return_dict=False, **extra_args['negative'])[0]
        
        if x_ddim_space.is_cuda:
            torch.cuda.synchronize()
            
        neg_time = time.time() - unet_start
        print(f"{neg_time*1000:.1f}ms", end='', flush=True)
        
        if not hasattr(self, '_first_unet_call_logged'):
            torch.cuda.synchronize()
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"\n[DEBUG] GPU Memory after UNet calls: {mem_allocated:.2f}GB")
                print(f"[DEBUG] UNet times - Positive: {pos_time*1000:.1f}ms, Negative: {neg_time*1000:.1f}ms")
            self._first_unet_call_logged = True
        
        # Classifier-free guidance
        print(f" CFG...", end='', flush=True)
        cfg_start = time.time()
        noise_pred = eps_negative + cfg_scale * (eps_positive - eps_negative)
        result = x - noise_pred * sigma[:, None, None, None]
        print(f"{(time.time()-cfg_start)*1000:.1f}ms", flush=True)
        
        return result


class KDiffusionSampler:
    def __init__(self, unet, **kwargs):
        self.unet = unet
        self.k_model = KModel(unet=unet, **kwargs)

    @torch.inference_mode()
    def __call__(
            self,
            initial_latent = None,
            strength = 1.0,
            num_inference_steps = 25,
            guidance_scale = 5.0,
            batch_size = 1,
            generator = None,
            prompt_embeds = None,
            negative_prompt_embeds = None,
            cross_attention_kwargs = None,
            same_noise_in_batch = False,
            progress_tqdm = None,
    ):
        print(f"[TRACE] KDiffusionSampler.__call__ started")
        print(f"[TRACE]   batch_size={batch_size}, steps={num_inference_steps}, cfg={guidance_scale}")

        device = self.unet.device
        print(f"[TRACE]   device={device}")
        
        # Handle generator - must be on same device as noise generation
        print(f"[TRACE] Handling generator (current device: {generator.device if generator else 'None'})...")
        if generator is not None and generator.device.type != device.type:
            # Extract seed from CPU generator and create new CUDA generator
            # This avoids ByteTensor device conversion issues
            old_state = generator.get_state()
            # Extract the seed from the state (first 8 bytes interpreted as uint64)
            import struct
            seed = struct.unpack('Q', bytes(old_state[:8]))[0]
            generator = torch.Generator(device=device).manual_seed(seed)
            print(f"[TRACE]   Generator moved to {device}, seed={seed}")

        # Sigmas
        print(f"[TRACE] Computing sigmas...")
        sigmas = self.k_model.get_sigmas_karras(int(num_inference_steps/strength))
        sigmas = sigmas[-(num_inference_steps + 1):].to(device)
        print(f"[TRACE]   Sigmas: min={sigmas.min():.4f}, max={sigmas.max():.4f}")

        # Initial latents
        print(f"[TRACE] Generating initial noise (same_noise={same_noise_in_batch})...")
        noise_start = time.time()
        if same_noise_in_batch:
            noise = torch.randn(initial_latent.shape, generator=generator, device=device, dtype=self.unet.dtype).repeat(batch_size, 1, 1, 1)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1).to(device=device, dtype=self.unet.dtype)
        else:
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1).to(device=device, dtype=self.unet.dtype)
            noise = torch.randn(initial_latent.shape, generator=generator, device=device, dtype=self.unet.dtype)
        print(f"[TRACE]   Noise generation took {(time.time()-noise_start)*1000:.1f}ms")

        print(f"[TRACE] Adding noise to latents...")
        latents = initial_latent + noise * sigmas[0].to(initial_latent)

        # Batch
        print(f"[TRACE] Preparing batched embeddings...")
        batch_start = time.time()
        latents = latents.to(device)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1).to(device)
        negative_prompt_embeds = negative_prompt_embeds.repeat(batch_size, 1, 1).to(device)
        print(f"[TRACE]   Batching took {(time.time()-batch_start)*1000:.1f}ms")
        print(f"[TRACE]   Final latents shape: {latents.shape}")
        print(f"[TRACE]   Prompt embeds shape: {prompt_embeds.shape}")

        # Feeds
        print(f"[TRACE] Preparing sampler kwargs...")
        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs
            ),
            negative=dict(
                encoder_hidden_states=negative_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        )

        # Sample
        print(f"[TRACE] Calling sample_dpmpp_2m...")
        results = sample_dpmpp_2m(self.k_model, latents, sigmas, extra_args=sampler_kwargs, progress_tqdm=progress_tqdm)

        print(f"[TRACE] KDiffusionSampler.__call__ completed, result shape: {results.shape}")
        return results
