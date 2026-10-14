"""Matcha-TTS ONNX adapters for the published, student, and selected CPU models.

The selected model uses the fork's exact grapheme frontend and an ONNX Vocos
neural network with deterministic CPU ISTFT. Older modes retain their original
symbols.json frontend and two-step flow setting.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

def snapshot(repo, revision, allow_patterns=None):
    return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=allow_patterns))
from dataclasses import dataclass

@dataclass
class ModelSpec:
    repo: str
    revision: str
    mode: str = "selected"
from voice_server.errors import MissingDependency
from voice_server.base import Clock, SynthChunk, SynthOutput, TtsAdapter


class MatchaOnnx(TtsAdapter):
    streaming = False
    sample_rate = 22050

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.root: Path | None = None
        self.encoder = None
        self.decoder = None
        self.vocoder = None
        self.symbols: dict[str, int] = {}
        self.steps = 2
        self.mel_mean = -5.205414772033691
        self.mel_std = 2.5967071056365967
        self.frontend = None
        self.temperature = 0.9

    def load(self) -> str | None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise MissingDependency(
                "onnxruntime is not installed (install the voice-server dependencies)"
            ) from exc
        if self.spec.mode == "selected":
            local_dir = os.environ.get("MATCHA_CPU_DIR", "")
            self.root = Path(local_dir) if local_dir else snapshot(self.spec.repo, self.spec.revision, allow_patterns=["weights/cpu-onnx/*", "scripts/matcha_hybrid_frontend.py"]) / "weights/cpu-onnx"
            if not (self.root / "model.json").is_file():
                raise MissingDependency(f"selected CPU model.json is missing in {self.root}")
        elif self.spec.mode == "student":
            local_dir = os.environ.get("MATCHA_STUDENT_DIR", "")
            if not local_dir:
                raise MissingDependency("MATCHA_STUDENT_DIR is unset")
            self.root = Path(local_dir)
            if not (self.root / "model.json").is_file():
                raise MissingDependency(f"student model.json is missing in {self.root}")
        else:
            self.root = snapshot(
                self.spec.repo, self.spec.revision,
                allow_patterns=["int8/*.onnx", "symbols.json"],
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        if self.spec.mode == "selected":
            metadata = json.loads((self.root / "model.json").read_text())
            if metadata.get("frontend") != "vi_grapheme_original_v1" or metadata.get("vocoder_backend") != "onnx-spectrum-scipy-istft":
                raise RuntimeError("unsupported selected CPU model contract")
            vocab_path = self.root / "phoneme_vocab.json"
            if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != metadata["vocab_sha256"]:
                raise RuntimeError("selected CPU vocabulary hash mismatch")
            for item in metadata["files"].values():
                graph_path = self.root / item["name"]
                if hashlib.sha256(graph_path.read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"selected CPU graph hash mismatch: {graph_path}")
            from voice_server.hybrid_frontend import HybridFrontend

            self.frontend = HybridFrontend(vocab_path)
            self.steps = int(metadata["steps"])
            self.temperature = float(metadata["temperature"])
            self.mel_mean = float(metadata["mel_mean"])
            self.mel_std = float(metadata["mel_std"])
            model_dir = self.root
            encoder_name = "matcha_encoder.onnx"
            decoder_name = "matcha_decoder.onnx"
            vocoder_name = "vocos_spectrum.onnx"
        elif self.spec.mode == "student":
            metadata = json.loads((self.root / "model.json").read_text())
            use_int8 = os.environ.get("MATCHA_STUDENT_PRECISION", "fp32") == "int8"
            suffix = "_int8" if use_int8 else ""
            model_dir = self.root
            encoder_name = f"matcha_encoder{suffix}.onnx"
            decoder_name = f"matcha_decoder{suffix}.onnx"
            self.mel_mean = float(metadata["mel_mean"])
            self.mel_std = float(metadata["mel_std"])
            vocoder_name = "vocos.onnx"
        else:
            quant = "int8" if (self.root / "int8").is_dir() else "fp32"
            model_dir = self.root / quant
            encoder_name = "matcha_encoder.onnx"
            decoder_name = "matcha_decoder.onnx"
            vocoder_name = "vocos.onnx"
        self.encoder = ort.InferenceSession(str(model_dir / encoder_name), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(model_dir / decoder_name), opts, providers=["CPUExecutionProvider"])
        self.vocoder = ort.InferenceSession(str(model_dir / vocoder_name), opts, providers=["CPUExecutionProvider"])
        symbol_path = self.root / "symbols.json"
        if symbol_path.is_file():
            raw = json.loads(symbol_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                mapping = raw.get("_symbol_to_id")
                if isinstance(mapping, dict):
                    self.symbols = {str(k): int(v) for k, v in mapping.items()}
                elif isinstance(raw.get("symbols"), list):
                    self.symbols = {str(sym): i for i, sym in enumerate(raw["symbols"])}
            elif isinstance(raw, list):
                self.symbols = {str(sym): i for i, sym in enumerate(raw)}
        return str(model_dir)

    def _ids(self, text: str) -> list[int]:
        if self.spec.mode == "selected":
            return self.frontend.encode(text)
        blank = self.symbols.get("_", self.symbols.get("<blank>", 0))
        ids: list[int] = []
        if self.spec.mode == "student":
            from voice_server.matcha_frontend import clean_vietnamese_graphemes

            cleaned = clean_vietnamese_graphemes(text, set(self.symbols))
        else:
            cleaned = _clean_text(text)
        for ch in cleaned:
            ch = ch.lower() if ch.isascii() else ch
            if ch not in self.symbols:
                continue
            ids.append(blank)
            ids.append(self.symbols[ch])
        if ids:
            ids.append(blank)
        if not ids:
            raise RuntimeError("symbols.json did not cover any character in the text")
        return ids

    def synthesize(self, text: str) -> SynthOutput:
        if self.encoder is None:
            self.load()
        assert self.encoder and self.decoder and self.vocoder
        try:
            import numpy as np
        except ImportError as exc:
            raise MissingDependency("numpy is not installed") from exc
        clock = Clock()
        ids = np.asarray([self._ids(text)], dtype=np.int64)
        lengths = np.asarray([ids.shape[1]], dtype=np.int64)
        enc_feed = _feed(self.encoder, {"ids": ids, "lengths": lengths, "length_scale": np.asarray(1.0, dtype=np.float32)})
        enc_out = dict(zip(_out_names(self.encoder), self.encoder.run(None, enc_feed)))
        mel = _euler(self.decoder, enc_out, self.steps, np,
                     temperature=self.temperature, sway=self.spec.mode != "selected")
        mel = mel * np.float32(self.mel_std) + np.float32(self.mel_mean)
        if self.spec.mode == "selected":
            mel = mel[:, :, :int(np.asarray(enc_out["y_mask"]).sum())]
        voc_feed = _feed(self.vocoder, {"mel": mel})
        if self.spec.mode == "selected":
            from voice_server.vocos_istft import istft_same

            real, imag = self.vocoder.run(None, voc_feed)
            wave = istft_same(real, imag)
        else:
            wave = self.vocoder.run(None, voc_feed)[0]
        samples = np.asarray(wave).astype("float32").reshape(-1)
        if self.spec.mode == "selected":
            if not np.isfinite(samples).all() or samples.size < 2205:
                raise RuntimeError("selected CPU model returned invalid audio")
            return SynthOutput(
                chunks=[SynthChunk(samples.tolist(), self.sample_rate, clock.now())],
                sample_rate=self.sample_rate, streaming=False,
                notes=[f"steps={self.steps}", "selected-vocos-onnx"],
            )
        if samples.size > 1512:
            samples = samples[:-512]
        fade = min(samples.size, int(self.sample_rate * 0.02))
        if fade:
            samples[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 0:
            samples = samples / peak * 0.95
        return SynthOutput(
            chunks=[SynthChunk(samples.tolist(), self.sample_rate, clock.now())],
            sample_rate=self.sample_rate,
            streaming=False,
            notes=[f"steps={self.steps}"],
        )


def _out_names(session) -> list[str]:
    return [o.name for o in session.get_outputs()]


def _feed(session, available: dict) -> dict:
    """Match session inputs to ids, lengths, or a mel spectrogram by name."""
    feed = {}
    described = []
    for inp in session.get_inputs():
        described.append(f"{inp.name}:{inp.shape}")
        low = inp.name.lower()
        if low == "length_scale":
            feed[inp.name] = available["length_scale"]
        elif any(key in low for key in ("length", "len")):
            feed[inp.name] = available["lengths"]
        elif any(key in low for key in ("token", "text", "phone", "x", "ids", "src")):
            feed[inp.name] = available["ids"]
        elif "mel" in low or "mu" in low or "spec" in low:
            feed[inp.name] = available.get("mel")
        else:
            raise RuntimeError(
                "unrecognized ONNX input "
                + ", ".join(described)
                + ". Refusing to guess."
            )
    if any(v is None for v in feed.values()):
        raise RuntimeError("ONNX feed is incomplete: " + ", ".join(described))
    return feed


def _euler(decoder, enc_out, steps: int, np, *, temperature: float = 0.9,
           sway: bool = True) -> object:
    """A short Euler loop for a flow decoder. Names that are not mu/t/mask/x raise."""
    mu = None
    for key, value in enc_out.items():
        if "mu" in key.lower() or "mel" in key.lower() or "mean" in key.lower():
            mu = value
            break
    if mu is None:
        mu = next(iter(enc_out.values()))
    mask = enc_out.get("y_mask")
    if mask is None:
        raise RuntimeError("Matcha encoder did not return y_mask")
    x = np.random.default_rng(42).normal(0.0, temperature, size=np.asarray(mu).shape).astype("float32")
    for i in range(steps):
        if sway:
            u = i / steps
            u_next = (i + 1) / steps
            t = np.asarray([1.0 - math.cos(math.pi * 0.5 * u)], dtype=np.float32)
            dt = (1.0 - math.cos(math.pi * 0.5 * u_next)) - float(t[0])
        else:
            t = np.asarray([i / steps], dtype=np.float32)
            dt = 1.0 / steps
        feed = {}
        for inp in decoder.get_inputs():
            low = inp.name.lower()
            if low in {"t", "timestep", "time"} or low.endswith("_t"):
                feed[inp.name] = t
            elif "mask" in low:
                feed[inp.name] = mask
            elif low in {"x", "z", "noise"} or "latent" in low:
                feed[inp.name] = x
            elif "mu" in low or "mel" in low or "cond" in low or "mean" in low:
                feed[inp.name] = mu
            else:
                names = [f"{i.name}" for i in decoder.get_inputs()]
                raise RuntimeError("decoder input not mapped: " + ", ".join(names))
        vel = decoder.run(None, feed)[0]
        x = x + dt * np.asarray(vel)
    return x.astype("float32")


def _clean_text(text: str) -> str:
    replacements = {
        "/": " xuyệt ", "\\": " xuyệt ngược ", "_": " gạch dưới ",
        "@": " a còng ", "#": " thăng ", "$": " đô la ",
        "%": " phần trăm ", "^": " mũ ", "&": " và ",
        "*": " sao ", "+": " cộng ", "=": " bằng ",
        "<": " nhỏ hơn ", ">": " lớn hơn ", "|": " hoặc ", "~": " khoảng ",
        "–": "-", "—": "-", "‘": "'", "’": "'",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    text = re.sub(r'[()\[\]{}"`“”«»]', "", text)
    return " ".join(text.split())
