from __future__ import annotations

from pathlib import Path

import numpy as np

def uniform_sample_frames(frames: tuple[Path, ...], max_frames: int) -> tuple[Path, ...]:
    if len(frames) <= max_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, num=max_frames)
    indices = np.round(indices).astype(np.int64)
    indices = np.unique(indices)

    if len(indices) < max_frames:
        indices = np.linspace(0, len(frames) - 1, num=max_frames, dtype=np.int64)

    return tuple(frames[int(i)] for i in indices)
