"""The one intervention mechanism used everywhere: forward hooks on decoder blocks.

An *edit* maps a block's output hidden state h [batch, seq, d] to a new one and applies at every
position (prompt and generated tokens). A *reader* sees (block input, block output) after the edit,
to record activations or what the block wrote. Math runs in float32, results cast back.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch

Edit = Callable[[torch.Tensor], torch.Tensor]
Reader = Callable[[torch.Tensor, torch.Tensor], None]


def add(v: torch.Tensor) -> Edit:
    """h + v (steering)."""
    return lambda h: (h.float() + v).to(h.dtype)


def ablate(u: torch.Tensor) -> Edit:
    """Remove the component along unit vector u: h - <h,u> u."""
    return lambda h: (h.float() - (h.float() @ u)[..., None] * u).to(h.dtype)


def displace(u: torch.Tensor, r: torch.Tensor) -> Edit:
    """Matched-size null for ablate(u): move h by the same amount <h,u> but along random unit r."""
    return lambda h: (h.float() - (h.float() @ u)[..., None] * r).to(h.dtype)


def last_only(edit: Edit) -> Edit:
    """Apply an edit at the final position only (with left padding: the token before the model answers)."""
    return lambda h: torch.cat([h[:, :-1], edit(h[:, -1:])], 1)


def chain(*edits: Edit) -> Edit:
    def f(h):
        for e in edits:
            h = e(h)
        return h
    return f


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


@contextmanager
def hooked(blocks, edits: dict[int, Edit] | None = None, readers: dict[int, Reader] | None = None):
    edits, readers = edits or {}, readers or {}

    def make(i):
        def hook(module, args, kwargs, out):
            h = _hidden(out)
            if i in edits:
                h = edits[i](h)
            if i in readers:
                readers[i](args[0] if args else kwargs["hidden_states"], h)
            if i in edits:
                return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    handles = [blocks[i].register_forward_hook(make(i), with_kwargs=True) for i in sorted(set(edits) | set(readers))]
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()
