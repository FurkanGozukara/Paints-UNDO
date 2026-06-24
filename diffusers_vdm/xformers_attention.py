import os

import torch
import torch.nn.functional as F
import xformers.ops
from xformers.ops import fmha


_LOGGED = set()
_ATTENTION_CACHE = {}
_FAILED_OPS = set()


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


def _candidate_names(value):
    names = []
    for part in (value or "auto").replace(";", ",").split(","):
        name = part.strip().lower().replace("-", "_")
        if not name:
            continue
        # Validate now so a typo fails once with a useful message.
        _resolve_op(name)
        if name not in names:
            names.append(name)

    if "sdpa" not in names:
        names.append("sdpa")
    return names


def _signature(q, k, v, label, env_name, candidates):
    capability = None
    if q.is_cuda:
        capability = torch.cuda.get_device_capability(q.device)
    return (
        label,
        env_name,
        tuple(candidates),
        q.device.type,
        q.device.index,
        str(q.dtype),
        str(k.dtype),
        str(v.dtype),
        capability,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.stride()),
    )


def _sdpa(q, k, v, batch_chunk=0):
    if batch_chunk > 0 and not torch.is_grad_enabled():
        chunks = [slice(i, i + batch_chunk) for i in range(0, q.size(0), batch_chunk)]
        return torch.cat(
            [F.scaled_dot_product_attention(q[chunk], k[chunk], v[chunk], attn_mask=None) for chunk in chunks],
            dim=0,
        )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None)


def _run_attention(name, q, k, v, batch_chunk=0):
    op_name, op = _resolve_op(name)
    if op == "sdpa":
        return _sdpa(q, k, v, batch_chunk=batch_chunk)
    if op is None:
        return xformers.ops.memory_efficient_attention(q, k, v)
    return xformers.ops.memory_efficient_attention(q, k, v, op=op)


def _validate_output(out, label, op_name):
    mode = os.environ.get("PAINTS_UNDO_ATTENTION_VALIDATE", "first").strip().lower()
    if mode in ("0", "false", "off", "none", "no"):
        return
    if not torch.isfinite(out).all().item():
        raise FloatingPointError(f"{label} attention op '{op_name}' produced non-finite output")


def memory_efficient_attention(q, k, v, *, env_name, default_op="auto", batch_chunk=0, label="xformers"):
    candidates = _candidate_names(os.environ.get(env_name, default_op))
    cache_key = _signature(q, k, v, label, env_name, candidates)
    validate_mode = os.environ.get("PAINTS_UNDO_ATTENTION_VALIDATE", "first").strip().lower()

    cached = _ATTENTION_CACHE.get(cache_key)
    if cached is not None:
        try:
            out = _run_attention(cached, q, k, v, batch_chunk=batch_chunk)
            if validate_mode == "always":
                _validate_output(out, label, cached)
            return out
        except Exception as exc:
            _ATTENTION_CACHE.pop(cache_key, None)
            _FAILED_OPS.add((cache_key, cached))
            if q.is_cuda:
                torch.cuda.empty_cache()
            _log_once(
                (label, env_name, cached, "cached-fallback"),
                f"[xformers] {label}: cached op '{cached}' failed ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0]}), retrying fallback chain",
            )

    last_exc = None
    for candidate in candidates:
        if (cache_key, candidate) in _FAILED_OPS:
            continue
        try:
            out = _run_attention(candidate, q, k, v, batch_chunk=batch_chunk)
            _validate_output(out, label, candidate)
            _ATTENTION_CACHE[cache_key] = candidate
            if candidate == "sdpa":
                _log_once((label, env_name, candidate), f"[xformers] {label}: using PyTorch SDPA")
            else:
                _log_once((label, env_name, candidate), f"[xformers] {label}: using xformers op '{candidate}'")
            return out
        except Exception as exc:
            last_exc = exc
            _FAILED_OPS.add((cache_key, candidate))
            if q.is_cuda:
                torch.cuda.empty_cache()
            _log_once(
                (label, env_name, candidate, "fallback"),
                f"[xformers] {label}: op '{candidate}' failed ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0]}), trying next attention backend",
            )

    raise RuntimeError(f"No attention backend worked for {label}") from last_exc
