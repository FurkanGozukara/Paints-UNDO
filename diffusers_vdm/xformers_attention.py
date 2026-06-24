import os

import torch
import torch.nn.functional as F
import xformers.ops
from xformers.ops import fmha


_LOGGED = set()


def _log_once(key, message):
    if key not in _LOGGED:
        print(message)
        _LOGGED.add(key)


def _resolve_op(name):
    name = (name or "auto").strip().lower().replace("-", "_")
    if name in ("auto", "default", ""):
        return name, None
    if name in ("sdpa", "torch"):
        return "sdpa", "sdpa"
    if name == "triton_splitk":
        return name, (fmha.triton_splitk.FwOp, None)
    if name == "cutlass":
        return name, (fmha.cutlass.FwOp, fmha.cutlass.BwOp)
    if name == "cutlass_blackwell":
        return name, (fmha.cutlass_blackwell.FwOp, fmha.cutlass_blackwell.BwOp)
    if name == "flash":
        return name, (fmha.flash.FwOp, fmha.flash.BwOp)
    if name == "flash3":
        return name, (fmha.flash3.FwOp, fmha.flash3.BwOp)
    raise ValueError(
        f"Unknown xformers op '{name}'. Use auto, triton_splitk, cutlass, "
        "cutlass_blackwell, flash, flash3, or sdpa."
    )


def _sdpa(q, k, v, batch_chunk=0):
    if batch_chunk > 0 and not torch.is_grad_enabled():
        chunks = [slice(i, i + batch_chunk) for i in range(0, q.size(0), batch_chunk)]
        return torch.cat(
            [F.scaled_dot_product_attention(q[chunk], k[chunk], v[chunk], attn_mask=None) for chunk in chunks],
            dim=0,
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None)


def memory_efficient_attention(q, k, v, *, env_name, default_op="auto", batch_chunk=0, label="xformers"):
    op_name = os.environ.get(env_name, default_op)
    op_name, op = _resolve_op(op_name)

    if op == "sdpa":
        _log_once((label, env_name, op_name), f"[xformers] {label}: using PyTorch SDPA by request")
        return _sdpa(q, k, v, batch_chunk=batch_chunk)

    try:
        if op is None:
            out = xformers.ops.memory_efficient_attention(q, k, v)
        else:
            out = xformers.ops.memory_efficient_attention(q, k, v, op=op)
        _log_once((label, env_name, op_name), f"[xformers] {label}: using xformers op '{op_name}'")
        return out
    except Exception as exc:
        if q.is_cuda:
            torch.cuda.empty_cache()
        _log_once(
            (label, env_name, op_name, "fallback"),
            f"[xformers] {label}: op '{op_name}' failed ({type(exc).__name__}: "
            f"{str(exc).splitlines()[0]}), using PyTorch SDPA fallback",
        )
        return _sdpa(q, k, v, batch_chunk=batch_chunk)
