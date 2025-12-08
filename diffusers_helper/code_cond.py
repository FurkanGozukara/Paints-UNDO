import torch

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


def unet_add_coded_conds(unet, added_number_count=1):
    unet.add_time_proj = Timesteps(256, True, 0)
    unet.add_embedding = TimestepEmbedding(256 * added_number_count, 1280)

    def get_aug_embed(emb, encoder_hidden_states, added_cond_kwargs):
        coded_conds = added_cond_kwargs.get("coded_conds")
        batch_size = coded_conds.shape[0]
        time_embeds = unet.add_time_proj(coded_conds.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb)
        aug_emb = unet.add_embedding(time_embeds)
        return aug_emb

    unet.get_aug_embed = get_aug_embed

    unet_original_forward = unet.forward

    def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
        import time
        
        prep_start = time.time()
        cross_attention_kwargs = {k: v for k, v in kwargs['cross_attention_kwargs'].items()}
        coded_conds = cross_attention_kwargs.pop('coded_conds')
        kwargs['cross_attention_kwargs'] = cross_attention_kwargs

        repeat_count = sample.shape[0] // coded_conds.shape[0]
        coded_conds = torch.cat([coded_conds] * repeat_count, dim=0).to(sample.device)
        kwargs['added_cond_kwargs'] = dict(coded_conds=coded_conds)
        prep_time = (time.time() - prep_start) * 1000
        
        if not hasattr(hooked_unet_forward, '_first_call'):
            print(f"[TRACE] code_cond hook: repeat={repeat_count}, prep took {prep_time:.1f}ms")
            hooked_unet_forward._first_call = True
        
        forward_start = time.time()
        result = unet_original_forward(sample, timestep, encoder_hidden_states, **kwargs)
        forward_time = (time.time() - forward_start) * 1000
        
        if not hasattr(hooked_unet_forward, '_forward_logged'):
            print(f"[TRACE] code_cond hook: calling cat_cond forward (which calls original UNet)...")
            hooked_unet_forward._forward_logged = True
            
        return result

    unet.forward = hooked_unet_forward

    return
