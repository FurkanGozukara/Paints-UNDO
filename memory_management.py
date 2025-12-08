import torch
from contextlib import contextmanager


# Default to high VRAM mode (can be changed via set_high_vram_mode())
high_vram = True
gpu = torch.device('cuda')
cpu = torch.device('cpu')

torch.zeros((1, 1)).to(gpu, torch.float32)
torch.cuda.empty_cache()

models_in_gpu = []


def set_high_vram_mode(enabled):
    """
    Set high VRAM mode.
    - enabled=True: Keep all models in GPU (recommended for RTX 3090/4090/5090, etc.)
    - enabled=False: Unload models when not in use (for lower VRAM GPUs)
    """
    global high_vram
    high_vram = enabled
    mode = "HIGH VRAM" if enabled else "LOW VRAM"
    print(f"\n[Memory Management] Mode set to: {mode}")
    if enabled:
        print("[Memory Management] All models will stay in GPU memory for maximum performance")
    else:
        print("[Memory Management] Models will be unloaded when not in use to save VRAM")
    print()


@contextmanager
def movable_bnb_model(m):
    if hasattr(m, 'quantization_method'):
        m.quantization_method_backup = m.quantization_method
        del m.quantization_method
    try:
        yield None
    finally:
        if hasattr(m, 'quantization_method_backup'):
            m.quantization_method = m.quantization_method_backup
            del m.quantization_method_backup
    return


def load_models_to_gpu(models):
    global models_in_gpu

    if not isinstance(models, (tuple, list)):
        models = [models]

    # Check which models are actually on GPU vs just tracked
    models_actually_on_gpu = []
    models_tracked_but_on_cpu = []
    
    for m in models_in_gpu:
        try:
            if next(m.parameters()).device.type == 'cuda':
                models_actually_on_gpu.append(m)
            else:
                models_tracked_but_on_cpu.append(m)
        except StopIteration:
            # Model has no parameters, keep it in list
            models_actually_on_gpu.append(m)
    
    models_to_remain = [m for m in set(models) if m in models_actually_on_gpu]
    models_to_load = [m for m in set(models) if m not in models_actually_on_gpu]
    models_to_unload = [m for m in set(models_actually_on_gpu) if m not in models_to_remain]

    # In high VRAM mode, also load any tracked models that are needed
    if high_vram:
        for m in models:
            if m in models_tracked_but_on_cpu and m not in models_to_load:
                models_to_load.append(m)

    if not high_vram:
        # Low VRAM mode: aggressively unload unused models
        for m in models_to_unload:
            with movable_bnb_model(m):
                m.to(cpu)
            print('[VRAM Management] Unload to CPU:', m.__class__.__name__)
        models_in_gpu = models_to_remain
    else:
        # High VRAM mode: keep all models in GPU
        if len(models_to_load) == 0 and len(models_actually_on_gpu) > 0:
            # All requested models are already on GPU
            pass
        elif len(models_actually_on_gpu) > 0:
            print(f'[VRAM Management] High VRAM mode - keeping {len(models_actually_on_gpu)} models in GPU')

    # Load models to GPU
    for m in models_to_load:
        with movable_bnb_model(m):
            m.to(gpu)
        print('[VRAM Management] Load to GPU:', m.__class__.__name__)

    # Update tracking list
    if high_vram:
        # In high VRAM mode, keep all loaded models in the list
        models_in_gpu = list(set(models_in_gpu + models))
    else:
        # In low VRAM mode, only track currently active models
        models_in_gpu = list(set(models_to_remain + models))
    
    # Count actual models on GPU
    if len(models_to_load) > 0:
        actual_gpu_count = sum(1 for m in models_in_gpu 
                              if next(m.parameters(), None) is None 
                              or next(m.parameters()).device.type == "cuda")
        print(f'[VRAM Management] Total models in GPU: {actual_gpu_count}')
    
    torch.cuda.empty_cache()
    return


def unload_all_models(extra_models=None):
    global models_in_gpu

    if extra_models is None:
        extra_models = []

    if not isinstance(extra_models, (tuple, list)):
        extra_models = [extra_models]

    models_in_gpu = list(set(models_in_gpu + extra_models))

    return load_models_to_gpu([])
