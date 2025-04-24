import os
import shutil
import functools
import random
import gradio as gr
import numpy as np
import torch
import wd14tagger
import memory_management
import uuid
import json
import platform
import argparse

from PIL import Image
from diffusers_helper.code_cond import unet_add_coded_conds
from diffusers_helper.cat_cond import unet_add_concat_conds
from diffusers_helper.k_diffusion import KDiffusionSampler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers_vdm.pipeline import LatentVideoDiffusionPipeline
from diffusers_vdm.utils import resize_and_center_crop, save_bcthw_as_mp4

os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), 'hf_download')
result_dir = os.path.join('./', 'results')
os.makedirs(result_dir, exist_ok=True)

class ModifiedUNet(UNet2DConditionModel):
    @classmethod
    def from_config(cls, *args, **kwargs):
        m = super().from_config(*args, **kwargs)
        unet_add_concat_conds(unet=m, new_channels=4)
        unet_add_coded_conds(unet=m, added_number_count=1)
        return m

model_name = 'lllyasviel/paints_undo_single_frame'
tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder").to(torch.float16)
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(torch.bfloat16)
unet = ModifiedUNet.from_pretrained(model_name, subfolder="unet").to(torch.float16)

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

video_pipe = LatentVideoDiffusionPipeline.from_pretrained(
    'lllyasviel/paints_undo_multi_frame',
    fp16=True
)

memory_management.unload_all_models([
    video_pipe.unet, video_pipe.vae, video_pipe.text_encoder, video_pipe.image_projection, video_pipe.image_encoder,
    unet, vae, text_encoder
])

k_sampler = KDiffusionSampler(
    unet=unet,
    timesteps=1000,
    linear_start=0.00085,
    linear_end=0.020,
    linear=True
)

def create_incremental_folder(base_path, base_name):
    counter = 1
    while True:
        folder_name = f"{base_name}_{counter:04d}"
        full_path = os.path.join(base_path, folder_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        counter += 1

def open_results_folder():
    if platform.system() == "Windows":
        os.startfile("results")
    else:
        os.system(f"xdg-open results")

def get_last_preset():
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    last_preset_file = os.path.join(presets_dir, 'last_preset.txt')
    if os.path.exists(last_preset_file):
        with open(last_preset_file, 'r') as f:
            return f.read().strip()
    return None

def save_last_preset(preset_name):
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    last_preset_file = os.path.join(presets_dir, 'last_preset.txt')
    with open(last_preset_file, 'w') as f:
        f.write(preset_name)

def save_preset(preset_name, settings):
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    os.makedirs(presets_dir, exist_ok=True)
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    with open(preset_path, 'w') as f:
        json.dump(settings, f)
    save_last_preset(preset_name)
    return gr.update(choices=get_preset_list(), value=preset_name)

def save_preset_wrapper(preset_name, input_undo_steps, seed, image_width, image_height, steps, cfg, n_prompt,
                        auto_set_dimensions, lowvram, i2v_input_text, i2v_seed, i2v_cfg_scale, i2v_steps, i2v_fps, use_random_seed):
    settings = {
        'input_undo_steps': input_undo_steps,
        'seed': seed,
        'image_width': image_width,
        'image_height': image_height,
        'steps': steps,
        'cfg': cfg,
        'n_prompt': n_prompt,
        'auto_set_dimensions': auto_set_dimensions,
        'lowvram': lowvram,
        'i2v_input_text': i2v_input_text,
        'i2v_seed': i2v_seed,
        'i2v_cfg_scale': i2v_cfg_scale,
        'i2v_steps': i2v_steps,
        'i2v_fps': i2v_fps,
        'use_random_seed': use_random_seed
    }
    return save_preset(preset_name, settings)

def load_preset(preset_name):
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    preset_path = os.path.join(presets_dir, f"{preset_name}.json")
    try:
        with open(preset_path, 'r') as f:
            settings = json.load(f)
        save_last_preset(preset_name)
        return [settings.get(key, None) for key in [
            'input_undo_steps', 'seed', 'image_width', 'image_height', 'steps', 'cfg', 'n_prompt',
            'auto_set_dimensions', 'lowvram',
            'i2v_input_text', 'i2v_seed', 'i2v_cfg_scale', 'i2v_steps', 'i2v_fps', 'use_random_seed'
        ]]
    except:
        return [None] * 16

def get_preset_list():
    presets_dir = os.path.join(os.path.dirname(__file__), 'presets')
    os.makedirs(presets_dir, exist_ok=True)
    return [f.split('.')[0] for f in os.listdir(presets_dir) if f.endswith('.json')]

def find_best_bucket(h, w, options):
    min_metric = float('inf')
    best_bucket = None
    for (bucket_h, bucket_w) in options:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket = (bucket_h, bucket_w)
    return best_bucket

@torch.inference_mode()
def encode_cropped_prompt_77tokens(txt: str):
    memory_management.load_models_to_gpu(text_encoder)
    cond_ids = tokenizer(txt,
                         padding="max_length",
                         max_length=tokenizer.model_max_length,
                         truncation=True,
                         return_tensors="pt").input_ids.to(device=text_encoder.device)
    text_cond = text_encoder(cond_ids, attention_mask=None).last_hidden_state
    return text_cond

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results

@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h

def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)

@torch.inference_mode()
def interrogator_process(x):
    image = np.array(Image.open(x))
    return wd14tagger.default_interrogator(image)

@torch.inference_mode()
def process(input_fg_path, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg,
            use_random_seed, progress=gr.Progress()):
    if use_random_seed:
        seed = random.randint(0, 1000000)
    rng = torch.Generator(device=memory_management.gpu).manual_seed(int(seed))
    input_fg = np.array(Image.open(input_fg_path))
    memory_management.load_models_to_gpu(vae)
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor

    memory_management.load_models_to_gpu(text_encoder)
    conds = encode_cropped_prompt_77tokens(prompt)
    unconds = encode_cropped_prompt_77tokens(n_prompt)

    memory_management.load_models_to_gpu(unet)
    fs = torch.tensor(input_undo_steps).to(device=unet.device, dtype=torch.long)
    initial_latents = torch.zeros_like(concat_conds)
    concat_conds = concat_conds.to(device=unet.device, dtype=unet.dtype)
    latents = k_sampler(
        initial_latent=initial_latents,
        strength=1.0,
        num_inference_steps=steps,
        guidance_scale=cfg,
        batch_size=len(input_undo_steps),
        generator=rng,
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        cross_attention_kwargs={'concat_conds': concat_conds, 'coded_conds': fs},
        same_noise_in_batch=True,
        progress_tqdm=functools.partial(progress.tqdm, desc='Generating Key Frames')
    ).to(vae.dtype) / vae.config.scaling_factor

    memory_management.load_models_to_gpu(vae)
    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels)
    pixels = [fg] + pixels + [np.zeros_like(fg) + 255]

    input_name = os.path.splitext(os.path.basename(input_fg_path))[0]
    frames_folder = create_incremental_folder(os.path.join(result_dir, 'frames'), f"{input_name}_key_frames")
    
    result = []
    for i, frame in enumerate(pixels):
        file_path = os.path.join(frames_folder, f"frame_{i:04d}.png")
        Image.fromarray(frame).save(file_path)
        result.append(file_path)  # Only append the file path

    return result, frames_folder, seed

@torch.inference_mode()
def process_video_inner(image_1, image_2, prompt, seed=123, steps=25, cfg_scale=7.5, fs=3, progress_tqdm=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    frames = 16

    target_height, target_width = find_best_bucket(
        image_1.shape[0], image_1.shape[1],
        options=[(320, 512), (384, 448), (448, 384), (512, 320)]
    )

    image_1 = resize_and_center_crop(image_1, target_width=target_width, target_height=target_height)
    image_2 = resize_and_center_crop(image_2, target_width=target_width, target_height=target_height)
    input_frames = numpy2pytorch([image_1, image_2])
    input_frames = input_frames.unsqueeze(0).movedim(1, 2)

    memory_management.load_models_to_gpu(video_pipe.text_encoder)
    positive_text_cond = video_pipe.encode_cropped_prompt_77tokens(prompt)
    negative_text_cond = video_pipe.encode_cropped_prompt_77tokens("")

    memory_management.load_models_to_gpu([video_pipe.image_projection, video_pipe.image_encoder])
    input_frames = input_frames.to(device=video_pipe.image_encoder.device, dtype=video_pipe.image_encoder.dtype)
    positive_image_cond = video_pipe.encode_clip_vision(input_frames)
    positive_image_cond = video_pipe.image_projection(positive_image_cond)
    negative_image_cond = video_pipe.encode_clip_vision(torch.zeros_like(input_frames))
    negative_image_cond = video_pipe.image_projection(negative_image_cond)

    memory_management.load_models_to_gpu([video_pipe.vae])
    input_frames = input_frames.to(device=video_pipe.vae.device, dtype=video_pipe.vae.dtype)
    input_frame_latents, vae_hidden_states = video_pipe.encode_latents(input_frames, return_hidden_states=True)
    first_frame = input_frame_latents[:, :, 0]
    last_frame = input_frame_latents[:, :, 1]
    concat_cond = torch.stack([first_frame] + [torch.zeros_like(first_frame)] * (frames - 2) + [last_frame], dim=2)

    memory_management.load_models_to_gpu([video_pipe.unet])
    latents = video_pipe(
        batch_size=1,
        steps=int(steps),
        guidance_scale=cfg_scale,
        positive_text_cond=positive_text_cond,
        negative_text_cond=negative_text_cond,
        positive_image_cond=positive_image_cond,
        negative_image_cond=negative_image_cond,
        concat_cond=concat_cond,
        fs=fs,
        progress_tqdm=progress_tqdm
    )

    memory_management.load_models_to_gpu([video_pipe.vae])
    video = video_pipe.decode_latents(latents, vae_hidden_states)
    return video, image_1, image_2

def auto_set_dimensions(image, lowvram):
    if image is None:
        return gr.update(), gr.update()
    
    width, height = image.shape[1], image.shape[0]
    aspect_ratio = width / height
    
    if lowvram:
        target = 512
    else:
        target = 1024
    
    if width < height:
        new_width = target
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target
        new_width = int(new_height * aspect_ratio)
    
    return gr.update(value=new_width), gr.update(value=new_height)

@torch.inference_mode()
def process_video(keyframes, prompt, steps, cfg, fps, seed, input_fg_path, use_random_seed, progress=gr.Progress()):
    result_frames = []
    cropped_images = []

    for i, (im1, im2) in enumerate(zip(keyframes[:-1], keyframes[1:])):
        im1 = np.array(Image.open(im1[0]))
        im2 = np.array(Image.open(im2[0]))
        if use_random_seed:
            seed = random.randint(0, 1000000)
        frames, im1, im2 = process_video_inner(
            im1, im2, prompt, seed=seed + i, steps=steps, cfg_scale=cfg, fs=3,
            progress_tqdm=functools.partial(progress.tqdm, desc=f'Generating Videos ({i + 1}/{len(keyframes) - 1})')
        )
        result_frames.append(frames[:, :, :-1, :, :])
        cropped_images.append([im1, im2])

    video = torch.cat(result_frames, dim=2)
    video = torch.flip(video, dims=[2])

    input_name = os.path.splitext(os.path.basename(input_fg_path))[0]
    output_filename = generate_unique_filename(os.path.join(result_dir, f"{input_name}_0001.mp4"))
    Image.fromarray(cropped_images[0][0]).save(os.path.join(result_dir, f"{input_name}_0001.png"))
    video = save_bcthw_as_mp4(video, output_filename, fps=fps)
    video = [x.cpu().numpy() for x in video]

    video_frames_folder = create_incremental_folder(os.path.join(result_dir, 'video_frames'), f"{input_name}_final_frames")
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(os.path.join(video_frames_folder, f"frame_{i:04d}.png"))

    return output_filename, video

def generate_unique_filename(base_filename):
    counter = 1
    while os.path.exists(base_filename):
        name, ext = os.path.splitext(base_filename)
        base_name = name.rsplit('_', 1)[0]
        base_filename = f"{base_name}_{counter:04d}{ext}"
        counter += 1
    return base_filename

def create_ui():
    block = gr.Blocks().queue()
    with block:
        gr.Markdown('# Paints-Undo Upgraded - V5 - Source : https://www.patreon.com/posts/121228327')

        with gr.Accordion(label='Step 1: Upload Image and Generate Prompt', open=True):
            with gr.Row():
                with gr.Column():
                    input_fg = gr.Image(label="Image", type="filepath",height=512)
                with gr.Column():
                    prompt_gen_button = gr.Button(value="Generate Prompt", interactive=False)
                    prompt = gr.Textbox(label="Output Prompt", interactive=True)

        with gr.Accordion(label='Preset Management', open=True):
            with gr.Row():
                preset_name = gr.Textbox(label="Preset Name")
                save_preset_btn = gr.Button("Save Preset")
                load_preset_dropdown = gr.Dropdown(label="Load Preset", choices=get_preset_list())
                load_preset_btn = gr.Button("Load Preset")
                refresh_preset_btn = gr.Button("Refresh Presets")

        with gr.Accordion(label='Step 2: Generate Key Frames', open=True):
            with gr.Row():
                auto_set_dimensions_checkbox = gr.Checkbox(label="Auto Set Dimensions", value=True)
                lowvram_checkbox = gr.Checkbox(label="Low VRAM Mode", value=True)
                use_random_seed_checkbox = gr.Checkbox(label="Use Random Seed", value=True)
            with gr.Row():
                with gr.Column():
                    input_undo_steps = gr.Dropdown(label="Operation Steps", value=[400, 600, 800, 900, 950, 999],
                                                   choices=list(range(1000)), multiselect=True)
                    seed = gr.Slider(label='Stage 1 Seed', minimum=0, maximum=50000, step=1, value=12345)
                    image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=3.0, step=0.01)
                    n_prompt = gr.Textbox(label="Negative Prompt",
                                          value='lowres, bad anatomy, bad hands, cropped, worst quality')

                with gr.Column():
                    key_gen_button = gr.Button(value="Generate Key Frames", interactive=False)
                    result_gallery = gr.Gallery(height=512, object_fit='contain', label='Outputs', columns=4)

        with gr.Accordion(label='Step 3: Generate All Videos', open=True):
            with gr.Row():
                with gr.Column():
                    i2v_input_text = gr.Text(label='Prompts', value='1girl, masterpiece, best quality')
                    i2v_seed = gr.Slider(label='Stage 2 Seed', minimum=0, maximum=50000, step=1, value=123)
                    i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5,
                                              elem_id="i2v_cfg_scale")
                    i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps",
                                          label="Sampling steps", value=50)
                    i2v_fps = gr.Slider(minimum=1, maximum=30, step=1, elem_id="i2v_motion", label="FPS", value=4)
                with gr.Column():
                    open_results_btn = gr.Button("Open Results Folder")
                    i2v_end_btn = gr.Button("Generate Video", interactive=False)
                    i2v_output_video = gr.Video(label="Generated Video", elem_id="output_vid", autoplay=True,
                                                show_share_button=True, height=512)
            with gr.Row():
                i2v_output_images = gr.Gallery(height=512, label="Output Frames", object_fit="contain", columns=8)

        def update_on_image_change(image_path, auto_set, lowvram):
            outputs = [
                "",  # prompt
                gr.update(interactive=True),  # prompt_gen_button
                gr.update(interactive=False),  # key_gen_button
                gr.update(interactive=False),  # i2v_end_btn
            ]

            if auto_set and image_path is not None:
                image = np.array(Image.open(image_path))
                new_width, new_height = auto_set_dimensions(image, lowvram)
                outputs.extend([new_width, new_height])
            else:
                outputs.extend([gr.update(), gr.update()])

            return outputs

        input_fg.change(
            fn=update_on_image_change,
            inputs=[input_fg, auto_set_dimensions_checkbox, lowvram_checkbox],
            outputs=[prompt, prompt_gen_button, key_gen_button, i2v_end_btn, image_width, image_height]
        )

        prompt_gen_button.click(
            fn=interrogator_process,
            inputs=[input_fg],
            outputs=[prompt]
        ).then(
            lambda: [gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=False)],
            outputs=[prompt_gen_button, key_gen_button, i2v_end_btn]
        )

        key_gen_button.click(
            fn=process,
            inputs=[input_fg, prompt, input_undo_steps, image_width, image_height, seed, steps, n_prompt, cfg, use_random_seed_checkbox],
            outputs=[result_gallery, gr.State(), seed]
        ).then(
            lambda result, frames_folder, used_seed: [
                gr.update(value=result, label=f"Outputs (saved in {frames_folder})"),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(value=used_seed)
            ],
            inputs=[result_gallery, gr.State(), seed],
            outputs=[result_gallery, prompt_gen_button, key_gen_button, i2v_end_btn, seed]
        )

        i2v_end_btn.click(
            inputs=[result_gallery, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_fps, i2v_seed, input_fg, use_random_seed_checkbox],
            outputs=[i2v_output_video, i2v_output_images],
            fn=process_video
        )

        open_results_btn.click(fn=open_results_folder)

        auto_set_dimensions_checkbox.change(
            fn=auto_set_dimensions,
            inputs=[input_fg, lowvram_checkbox],
            outputs=[image_width, image_height]
        )

        save_preset_btn.click(
            fn=save_preset_wrapper,
            inputs=[
                preset_name, 
                input_undo_steps, seed, image_width, image_height, steps, cfg, n_prompt,
                auto_set_dimensions_checkbox, lowvram_checkbox,
                i2v_input_text, i2v_seed, i2v_cfg_scale, i2v_steps, i2v_fps, use_random_seed_checkbox
            ],
            outputs=[load_preset_dropdown]
        )

        load_preset_btn.click(
            fn=load_preset,
            inputs=[load_preset_dropdown],
            outputs=[
                input_undo_steps, seed, image_width, image_height, steps, cfg, n_prompt,
                auto_set_dimensions_checkbox, lowvram_checkbox,
                i2v_input_text, i2v_seed, i2v_cfg_scale, i2v_steps, i2v_fps, use_random_seed_checkbox
            ]
        )

        refresh_preset_btn.click(
            fn=lambda: gr.update(choices=get_preset_list()),
            outputs=[load_preset_dropdown]
        )

        # Load last preset on startup
        last_preset = get_last_preset()
        if last_preset:
            block.load(
                fn=lambda x: (x, *load_preset(x)), 
                inputs=[gr.Dropdown(value=last_preset)],
                outputs=[
                    load_preset_dropdown,  # Update the dropdown selection
                    input_undo_steps, seed, image_width, image_height, steps, cfg, n_prompt,
                    auto_set_dimensions_checkbox, lowvram_checkbox,
                    i2v_input_text, i2v_seed, i2v_cfg_scale, i2v_steps, i2v_fps, use_random_seed_checkbox
                ]
            )

    return block

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paints-Undo Gradio App")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share feature")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(share=args.share, inbrowser=True)