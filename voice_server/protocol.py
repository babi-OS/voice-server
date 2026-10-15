"""Troly voice protocol v2: independent speech operations and framed audio."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

VERSION = 2
PATH = "/v2/voice"
INPUT_RATE = 16000
OUTPUT_RATE = 24000
MAX_JSON_BYTES = 8192
MAX_OPUS_BYTES = 1275
DEFAULT_END_SILENCE_MS = 1000
MAX_END_SILENCE_MS = 3000
RATES = frozenset((8000, 12000, 16000, 24000, 48000))
DURATIONS_MS = frozenset((20, 40, 60))
APPLICATIONS = frozenset(("voip", "audio", "restricted_lowdelay"))
AUDIO_HEADER = struct.Struct("!II")  # stream_id, sequence; WebSocket supplies payload length.


class ProtocolError(ValueError):
    def __init__(self, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioDirection:
    codec: str
    sample_rate: int
    frame_ms: int
    packet_ms: int
    max_packet_bytes: int | None = None
    bitrate: int | None = None
    complexity: int | None = None
    application: str | None = None

    @property
    def packet_bytes(self) -> int:
        return self.sample_rate * self.packet_ms // 1000 * 2

    def wire(self) -> dict[str, Any]:
        value: dict[str, Any] = {"codec": self.codec, "sample_rate": self.sample_rate,
                                 "frame_ms": self.frame_ms, "packet_ms": self.packet_ms}
        if self.codec == "opus":
            value["max_packet_bytes"] = self.max_packet_bytes
            for name in ("bitrate", "complexity", "application"):
                item = getattr(self, name)
                if item is not None:
                    value[name] = item
        return value


@dataclass(frozen=True)
class AudioProfile:
    input: AudioDirection
    output: AudioDirection

    def wire(self) -> dict[str, Any]:
        return {"input": self.input.wire(), "output": self.output.wire()}


def parse_control(message: str) -> dict[str, Any]:
    if len(message.encode("utf-8")) > MAX_JSON_BYTES:
        raise ProtocolError("control message too large", "too_large")
    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise ProtocolError("control requires a type")
    return value


def _integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{label} must be an integer")
    return value


def end_silence_ms(message: dict[str, Any]) -> int:
    """Zero delegates end-of-utterance detection to the client."""
    value = _integer(message.get("end_silence_ms", DEFAULT_END_SILENCE_MS), "end_silence_ms")
    if value != 0 and not 256 <= value <= MAX_END_SILENCE_MS:
        raise ProtocolError(f"end_silence_ms must be 0 or 256-{MAX_END_SILENCE_MS}")
    return value


def _direction(value: Any, label: str, *, output: bool) -> AudioDirection:
    if not isinstance(value, dict):
        raise ProtocolError(f"audio.{label} must be an object")
    allowed = {"codec", "sample_rate", "frame_ms", "packet_ms", "max_packet_bytes", "bitrate", "complexity", "application"}
    if set(value) - allowed:
        raise ProtocolError(f"unsupported audio.{label} setting: {sorted(set(value) - allowed)[0]}")
    codec = value.get("codec")
    if codec not in ("opus", "pcm16"):
        raise ProtocolError(f"audio.{label}.codec must be opus or pcm16")
    rate = _integer(value.get("sample_rate"), f"audio.{label}.sample_rate")
    frame_ms = _integer(value.get("frame_ms"), f"audio.{label}.frame_ms")
    packet_ms = _integer(value.get("packet_ms"), f"audio.{label}.packet_ms")
    if rate not in RATES or frame_ms not in DURATIONS_MS or packet_ms not in DURATIONS_MS:
        raise ProtocolError(f"unsupported audio.{label} rate or duration")
    if packet_ms != frame_ms:
        raise ProtocolError(f"audio.{label}.packet_ms must equal frame_ms; multi-frame packets are unsupported")
    if codec == "pcm16":
        if {"max_packet_bytes", "bitrate", "complexity", "application"} & value.keys():
            raise ProtocolError(f"audio.{label} PCM16 cannot set Opus controls")
        return AudioDirection(codec, rate, frame_ms, packet_ms)
    maximum = _integer(value.get("max_packet_bytes", MAX_OPUS_BYTES), f"audio.{label}.max_packet_bytes")
    if not 1 <= maximum <= MAX_OPUS_BYTES:
        raise ProtocolError(f"audio.{label}.max_packet_bytes must be 1-{MAX_OPUS_BYTES}")
    bitrate = value.get("bitrate", 24000 if output else None)
    if bitrate is not None:
        bitrate = _integer(bitrate, f"audio.{label}.bitrate")
        if not 6000 <= bitrate <= 96000:
            raise ProtocolError(f"audio.{label}.bitrate must be 6000-96000")
    complexity = value.get("complexity")
    if complexity is not None:
        complexity = _integer(complexity, f"audio.{label}.complexity")
        if not 0 <= complexity <= 10:
            raise ProtocolError(f"audio.{label}.complexity must be 0-10")
    application = value.get("application", "audio" if output else None)
    if application is not None and (not isinstance(application, str) or application not in APPLICATIONS):
        raise ProtocolError(f"unsupported audio.{label}.application")
    return AudioDirection(codec, rate, frame_ms, packet_ms, maximum, bitrate, complexity, application)


def validate_hello(message: dict[str, Any]) -> AudioProfile:
    if set(message) != {"type", "version", "audio"} or message.get("type") != "hello" or message.get("version") != VERSION:
        raise ProtocolError("expected v2 hello with audio.input and audio.output")
    audio = message["audio"]
    if not isinstance(audio, dict) or set(audio) != {"input", "output"}:
        raise ProtocolError("audio requires input and output objects")
    return AudioProfile(_direction(audio["input"], "input", output=False),
                        _direction(audio["output"], "output", output=True))


def encode_control(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False, separators=(",", ":"))


def pack_audio(stream_id: int, sequence: int, payload: bytes) -> bytes:
    if not 0 < stream_id <= 0xFFFFFFFF or not 0 <= sequence <= 0xFFFFFFFF:
        raise ProtocolError("invalid audio stream ID or sequence")
    return AUDIO_HEADER.pack(stream_id, sequence) + payload


def unpack_audio(packet: bytes) -> tuple[int, int, bytes]:
    if len(packet) <= AUDIO_HEADER.size:
        raise ProtocolError("audio packet is missing its header or payload")
    stream_id, sequence = AUDIO_HEADER.unpack_from(packet)
    if stream_id == 0:
        raise ProtocolError("audio stream ID cannot be zero")
    return stream_id, sequence, packet[AUDIO_HEADER.size:]


def validate_opus(packet: bytes, max_bytes: int = MAX_OPUS_BYTES) -> None:
    if not packet or len(packet) > max_bytes:
        raise ProtocolError("invalid Opus packet size")


def validate_pcm16(packet: bytes, expected_bytes: int) -> None:
    if len(packet) != expected_bytes:
        raise ProtocolError(f"expected {expected_bytes} PCM16 bytes")
