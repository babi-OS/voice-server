"""Deterministic Vocos ISTFT for ONNX spectrum output."""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft


def istft_same(real: np.ndarray, imag: np.ndarray, *, n_fft: int = 1024,
               hop: int = 256) -> np.ndarray:
    """Match the fork's Hann-window, same-padding overlap-add implementation."""
    if real.shape != imag.shape or real.ndim != 3 or real.shape[1] != n_fft // 2 + 1:
        raise ValueError("invalid Vocos spectrum shape")
    if real.shape[2] < 2:
        raise ValueError("Vocos spectrum is too short")
    spec = real.astype(np.float32) + 1j * imag.astype(np.float32)
    frames = irfft(spec, n=n_fft, axis=1).astype(np.float32)
    window = np.hanning(n_fft + 1).astype(np.float32)[:-1]
    frames *= window[None, :, None]
    length = (real.shape[2] - 1) * hop + n_fft
    audio = np.zeros((real.shape[0], length), dtype=np.float32)
    envelope = np.zeros(length, dtype=np.float32)
    squared = window * window
    for frame in range(real.shape[2]):
        start = frame * hop
        audio[:, start:start + n_fft] += frames[:, :, frame]
        envelope[start:start + n_fft] += squared
    pad = (n_fft - hop) // 2
    audio = audio[:, pad:-pad]
    envelope = envelope[pad:-pad]
    if np.min(envelope) <= 1e-11:
        raise ValueError("invalid ISTFT window envelope")
    return audio / envelope[None, :]
