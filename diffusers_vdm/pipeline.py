import os
import torch
import einops

from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from huggingface_hub import snapshot_download
from diffusers_vdm.vae import VideoAutoencoderKL
from diffusers_vdm.projection import Resampler
from diffusers_vdm.unet import UNet3DModel
from diffusers_vdm.improved_clip_vision import ImprovedCLIPVisionModelWithProjection
from diffusers_vdm.dynamic_tsnr_sampler import SamplerDynamicTSNR


VIDEO_VAE_TILE_LATENT_SIZE = 32
VIDEO_VAE_TILE_LATENT_OVERLAP = 8
VIDEO_VAE_TILE_SAMPLE_SIZE = VIDEO_VAE_TILE_LATENT_SIZE * 8
VIDEO_VAE_TILE_SAMPLE_OVERLAP = VIDEO_VAE_TILE_LATENT_OVERLAP * 8


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]

    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _scale_range(start, end, source_length, target_length):
    return start * target_length // source_length, end * target_length // source_length


def _tile_blend_weight(height, width, overlap_y, overlap_x, dtype=torch.float32):
    weight = torch.ones((1, 1, height, width), dtype=dtype)

    if overlap_y > 0 and height > 1:
        overlap_y = min(overlap_y, height)
        ramp = torch.linspace(1.0 / (overlap_y + 1), 1.0, overlap_y, dtype=dtype)
        weight[:, :, :overlap_y, :] *= ramp.view(1, 1, overlap_y, 1)
        weight[:, :, -overlap_y:, :] *= ramp.flip(0).view(1, 1, overlap_y, 1)

    if overlap_x > 0 and width > 1:
        overlap_x = min(overlap_x, width)
        ramp = torch.linspace(1.0 / (overlap_x + 1), 1.0, overlap_x, dtype=dtype)
        weight[:, :, :, :overlap_x] *= ramp.view(1, 1, 1, overlap_x)
        weight[:, :, :, -overlap_x:] *= ramp.flip(0).view(1, 1, 1, overlap_x)

    return weight


class LatentVideoDiffusionPipeline(DiffusionPipeline):
    def __init__(self, tokenizer, text_encoder, image_encoder, vae, image_projection, unet, fp16=True, eval=True):
        super().__init__()

        self.loading_components = dict(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            image_encoder=image_encoder,
            image_projection=image_projection
        )

        for k, v in self.loading_components.items():
            setattr(self, k, v)

        if fp16:
            self.vae.half()
            self.text_encoder.half()
            self.unet.half()
            self.image_encoder.half()
            self.image_projection.half()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()

        if eval:
            self.unet.eval()
            self.image_projection.eval()
        else:
            self.unet.train()
            self.image_projection.train()

    def to(self, *args, **kwargs):
        for k, v in self.loading_components.items():
            if hasattr(v, 'to'):
                v.to(*args, **kwargs)
        return self

    def save_pretrained(self, save_directory, **kwargs):
        for k, v in self.loading_components.items():
            folder = os.path.join(save_directory, k)
            os.makedirs(folder, exist_ok=True)
            v.save_pretrained(folder)
        return

    @classmethod
    def from_pretrained(cls, repo_id, fp16=True, eval=True, token=None):
        local_folder = snapshot_download(repo_id=repo_id, token=token)
        return cls(
            tokenizer=CLIPTokenizer.from_pretrained(os.path.join(local_folder, "tokenizer")),
            text_encoder=CLIPTextModel.from_pretrained(os.path.join(local_folder, "text_encoder")),
            image_encoder=ImprovedCLIPVisionModelWithProjection.from_pretrained(os.path.join(local_folder, "image_encoder")),
            vae=VideoAutoencoderKL.from_pretrained(os.path.join(local_folder, "vae")),
            image_projection=Resampler.from_pretrained(os.path.join(local_folder, "image_projection")),
            unet=UNet3DModel.from_pretrained(os.path.join(local_folder, "unet")),
            fp16=fp16,
            eval=eval
        )

    @torch.inference_mode()
    def encode_cropped_prompt_77tokens(self, prompt: str):
        cond_ids = self.tokenizer(prompt,
                                  padding="max_length",
                                  max_length=self.tokenizer.model_max_length,
                                  truncation=True,
                                  return_tensors="pt").input_ids.to(self.text_encoder.device)
        cond = self.text_encoder(cond_ids, attention_mask=None).last_hidden_state
        return cond

    @torch.inference_mode()
    def encode_clip_vision(self, frames):
        b, c, t, h, w = frames.shape
        frames = einops.rearrange(frames, 'b c t h w -> (b t) c h w')
        clipvision_embed = self.image_encoder(frames).last_hidden_state
        clipvision_embed = einops.rearrange(clipvision_embed, '(b t) d c -> b t d c', t=t)
        return clipvision_embed

    @torch.inference_mode()
    def encode_latents(self, videos, return_hidden_states=True, tiled_vae=False):
        b, c, t, h, w = videos.shape
        x = einops.rearrange(videos, 'b c t h w -> (b t) c h w')

        if tiled_vae and (h > VIDEO_VAE_TILE_SAMPLE_SIZE or w > VIDEO_VAE_TILE_SAMPLE_SIZE):
            return self._encode_latents_tiled(x, b=b, t=t, height=h, width=w, return_hidden_states=return_hidden_states)

        encoder_posterior, hidden_states = self.vae.encode(x, return_hidden_states=return_hidden_states)
        z = encoder_posterior.mode() * self.vae.scale_factor
        z = einops.rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)

        if not return_hidden_states:
            return z

        hidden_states = [einops.rearrange(h, '(b t) c h w -> b c t h w', b=b) for h in hidden_states]
        hidden_states = [h[:, :, [0, -1], :, :] for h in hidden_states]  # only need first and last

        return z, hidden_states

    @torch.inference_mode()
    def decode_latents(self, latents, hidden_states, tiled_vae=False):
        B, C, T, H, W = latents.shape

        if tiled_vae and (H > VIDEO_VAE_TILE_LATENT_SIZE or W > VIDEO_VAE_TILE_LATENT_SIZE):
            return self._decode_latents_tiled(latents, hidden_states)

        latents = einops.rearrange(latents, 'b c t h w -> (b t) c h w')
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype) / self.vae.scale_factor
        hidden_states = [h.to(device=self.vae.device, dtype=self.vae.dtype) for h in hidden_states]
        pixels = self.vae.decode(latents, ref_context=hidden_states, timesteps=T)
        pixels = einops.rearrange(pixels, '(b t) c h w -> b c t h w', b=B, t=T)
        return pixels

    @torch.inference_mode()
    def _encode_latents_tiled(self, x, b, t, height, width, return_hidden_states=True):
        y_starts = _tile_starts(height, VIDEO_VAE_TILE_SAMPLE_SIZE, VIDEO_VAE_TILE_SAMPLE_OVERLAP)
        x_starts = _tile_starts(width, VIDEO_VAE_TILE_SAMPLE_SIZE, VIDEO_VAE_TILE_SAMPLE_OVERLAP)

        moments_accum = None
        moments_weight = None
        hidden_accums = None
        hidden_weights = None

        for y0 in y_starts:
            y1 = min(y0 + VIDEO_VAE_TILE_SAMPLE_SIZE, height)
            for x0 in x_starts:
                x1 = min(x0 + VIDEO_VAE_TILE_SAMPLE_SIZE, width)
                tile = x[:, :, y0:y1, x0:x1].to(device=self.vae.device, dtype=self.vae.dtype)
                posterior, hidden = self.vae.encode(tile, return_hidden_states=return_hidden_states)
                moments_tile = posterior.parameters.detach().cpu().float()

                if moments_accum is None:
                    latent_height = height * moments_tile.shape[-2] // (y1 - y0)
                    latent_width = width * moments_tile.shape[-1] // (x1 - x0)
                    moments_accum = torch.zeros(
                        (moments_tile.shape[0], moments_tile.shape[1], latent_height, latent_width),
                        dtype=torch.float32,
                    )
                    moments_weight = torch.zeros((1, 1, latent_height, latent_width), dtype=torch.float32)

                ly0, ly1 = _scale_range(y0, y1, height, moments_accum.shape[-2])
                lx0, lx1 = _scale_range(x0, x1, width, moments_accum.shape[-1])
                overlap_y = VIDEO_VAE_TILE_SAMPLE_OVERLAP * moments_tile.shape[-2] // (y1 - y0)
                overlap_x = VIDEO_VAE_TILE_SAMPLE_OVERLAP * moments_tile.shape[-1] // (x1 - x0)
                weight = _tile_blend_weight(moments_tile.shape[-2], moments_tile.shape[-1], overlap_y, overlap_x)
                moments_accum[:, :, ly0:ly1, lx0:lx1] += moments_tile * weight
                moments_weight[:, :, ly0:ly1, lx0:lx1] += weight

                if return_hidden_states:
                    if hidden_accums is None:
                        hidden_accums = []
                        hidden_weights = []
                        for hidden_tile in hidden:
                            hidden_height = height * hidden_tile.shape[-2] // (y1 - y0)
                            hidden_width = width * hidden_tile.shape[-1] // (x1 - x0)
                            hidden_accums.append(torch.zeros(
                                (hidden_tile.shape[0], hidden_tile.shape[1], hidden_height, hidden_width),
                                dtype=torch.float32,
                            ))
                            hidden_weights.append(torch.zeros((1, 1, hidden_height, hidden_width), dtype=torch.float32))

                    for i, hidden_tile in enumerate(hidden):
                        hidden_tile = hidden_tile.detach().cpu().float()
                        hy0, hy1 = _scale_range(y0, y1, height, hidden_accums[i].shape[-2])
                        hx0, hx1 = _scale_range(x0, x1, width, hidden_accums[i].shape[-1])
                        overlap_y = VIDEO_VAE_TILE_SAMPLE_OVERLAP * hidden_tile.shape[-2] // (y1 - y0)
                        overlap_x = VIDEO_VAE_TILE_SAMPLE_OVERLAP * hidden_tile.shape[-1] // (x1 - x0)
                        weight = _tile_blend_weight(hidden_tile.shape[-2], hidden_tile.shape[-1], overlap_y, overlap_x)
                        hidden_accums[i][:, :, hy0:hy1, hx0:hx1] += hidden_tile * weight
                        hidden_weights[i][:, :, hy0:hy1, hx0:hx1] += weight

                del tile, posterior
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        moments = (moments_accum / moments_weight.clamp_min(1e-6)).to(dtype=self.vae.dtype)
        z = moments.chunk(2, dim=1)[0] * self.vae.scale_factor
        z = einops.rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)

        if not return_hidden_states:
            return z

        hidden_states = [
            (accum / weight.clamp_min(1e-6)).to(dtype=self.vae.dtype)
            for accum, weight in zip(hidden_accums, hidden_weights)
        ]
        hidden_states = [einops.rearrange(h, '(b t) c h w -> b c t h w', b=b, t=t) for h in hidden_states]
        hidden_states = [h[:, :, [0, -1], :, :] for h in hidden_states]
        return z, hidden_states

    @torch.inference_mode()
    def _decode_latents_tiled(self, latents, hidden_states):
        B, C, T, H, W = latents.shape
        output_height = hidden_states[-1].shape[-2]
        output_width = hidden_states[-1].shape[-1]

        y_starts = _tile_starts(H, VIDEO_VAE_TILE_LATENT_SIZE, VIDEO_VAE_TILE_LATENT_OVERLAP)
        x_starts = _tile_starts(W, VIDEO_VAE_TILE_LATENT_SIZE, VIDEO_VAE_TILE_LATENT_OVERLAP)

        pixels_accum = torch.zeros((B, 3, T, output_height, output_width), dtype=torch.float32)
        pixels_weight = torch.zeros((1, 1, 1, output_height, output_width), dtype=torch.float32)

        for y0 in y_starts:
            y1 = min(y0 + VIDEO_VAE_TILE_LATENT_SIZE, H)
            py0, py1 = _scale_range(y0, y1, H, output_height)

            for x0 in x_starts:
                x1 = min(x0 + VIDEO_VAE_TILE_LATENT_SIZE, W)
                px0, px1 = _scale_range(x0, x1, W, output_width)

                tile_latents = latents[:, :, :, y0:y1, x0:x1]
                tile_latents = tile_latents.to(device=self.vae.device, dtype=self.vae.dtype) / self.vae.scale_factor
                tile_latents = einops.rearrange(tile_latents, 'b c t h w -> (b t) c h w')

                tile_hidden_states = []
                for hidden in hidden_states:
                    hy0, hy1 = _scale_range(y0, y1, H, hidden.shape[-2])
                    hx0, hx1 = _scale_range(x0, x1, W, hidden.shape[-1])
                    tile_hidden_states.append(
                        hidden[:, :, :, hy0:hy1, hx0:hx1].to(device=self.vae.device, dtype=self.vae.dtype)
                    )

                tile_pixels = self.vae.decode(tile_latents, ref_context=tile_hidden_states, timesteps=T)
                tile_pixels = einops.rearrange(tile_pixels, '(b t) c h w -> b c t h w', b=B, t=T)
                tile_pixels = tile_pixels.detach().cpu().float()

                overlap_y = VIDEO_VAE_TILE_LATENT_OVERLAP * tile_pixels.shape[-2] // (y1 - y0)
                overlap_x = VIDEO_VAE_TILE_LATENT_OVERLAP * tile_pixels.shape[-1] // (x1 - x0)
                weight = _tile_blend_weight(tile_pixels.shape[-2], tile_pixels.shape[-1], overlap_y, overlap_x)
                weight = weight.unsqueeze(2)
                pixels_accum[:, :, :, py0:py1, px0:px1] += tile_pixels * weight
                pixels_weight[:, :, :, py0:py1, px0:px1] += weight

                del tile_latents, tile_hidden_states, tile_pixels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return pixels_accum / pixels_weight.clamp_min(1e-6)

    @torch.inference_mode()
    def __call__(
            self,
            batch_size: int = 1,
            steps: int = 50,
            guidance_scale: float = 5.0,
            positive_text_cond = None,
            negative_text_cond = None,
            positive_image_cond = None,
            negative_image_cond = None,
            concat_cond = None,
            fs = 3,
            progress_tqdm = None,
    ):
        unet_is_training = self.unet.training

        if unet_is_training:
            self.unet.eval()

        device = self.unet.device
        dtype = self.unet.dtype
        dynamic_tsnr_model = SamplerDynamicTSNR(self.unet)

        # Batch

        concat_cond = concat_cond.repeat(batch_size, 1, 1, 1, 1).to(device=device, dtype=dtype)  # b, c, t, h, w
        positive_text_cond = positive_text_cond.repeat(batch_size, 1, 1).to(concat_cond)  # b, f, c
        negative_text_cond = negative_text_cond.repeat(batch_size, 1, 1).to(concat_cond)  # b, f, c
        positive_image_cond = positive_image_cond.repeat(batch_size, 1, 1, 1).to(concat_cond)  # b, t, l, c
        negative_image_cond = negative_image_cond.repeat(batch_size, 1, 1, 1).to(concat_cond)

        if isinstance(fs, torch.Tensor):
            fs = fs.repeat(batch_size, ).to(dtype=torch.long, device=device)  # b
        else:
            fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=device)  # b

        # Initial latents

        latent_shape = concat_cond.shape

        # Feeds

        sampler_kwargs = dict(
            cfg_scale=guidance_scale,
            positive=dict(
                context_text=positive_text_cond,
                context_img=positive_image_cond,
                fs=fs,
                concat_cond=concat_cond
            ),
            negative=dict(
                context_text=negative_text_cond,
                context_img=negative_image_cond,
                fs=fs,
                concat_cond=concat_cond
            )
        )

        # Sample

        results = dynamic_tsnr_model(latent_shape, steps, extra_args=sampler_kwargs, progress_tqdm=progress_tqdm)

        if unet_is_training:
            self.unet.train()

        return results
