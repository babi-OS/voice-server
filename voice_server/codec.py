"""Small libopus helpers without application-level frame repacketization."""

from __future__ import annotations

import ctypes

import opuslib_next
from opuslib_next.api import libopus

from voice_server.protocol import AudioDirection, ProtocolError, validate_opus

_nb_frames = libopus.opus_packet_get_nb_frames
_nb_frames.argtypes = (ctypes.c_void_p, ctypes.c_int)
_nb_frames.restype = ctypes.c_int
_samples_per_frame = libopus.opus_packet_get_samples_per_frame
_samples_per_frame.argtypes = (ctypes.c_void_p, ctypes.c_int)
_samples_per_frame.restype = ctypes.c_int


def packet_shape(packet: bytes, sample_rate: int) -> tuple[int, int]:
    """Return frame count and duration in milliseconds, checking the packet header."""
    if not packet:
        raise ProtocolError("empty Opus packet")
    data = ctypes.create_string_buffer(packet)
    count = _nb_frames(data, len(packet))
    samples = _samples_per_frame(data, sample_rate)
    if count <= 0 or samples <= 0 or samples * 1000 % sample_rate:
        raise ProtocolError("invalid Opus packet framing")
    return count, samples * 1000 // sample_rate


def validate_input_packet(packet: bytes, audio: AudioDirection) -> None:
    validate_opus(packet, audio.max_packet_bytes or 0)
    count, frame_ms = packet_shape(packet, audio.sample_rate)
    # libopus may internally represent a 40/60 ms encode call as shorter frames.
    if (frame_ms > audio.frame_ms or audio.frame_ms % frame_ms
            or count * frame_ms != audio.packet_ms):
        raise ProtocolError("Opus packet does not match negotiated frame and packet duration")


def make_encoder(audio: AudioDirection) -> opuslib_next.Encoder:
    application = {
        "voip": opuslib_next.constants.APPLICATION_VOIP,
        "audio": opuslib_next.constants.APPLICATION_AUDIO,
        "restricted_lowdelay": opuslib_next.constants.APPLICATION_RESTRICTED_LOWDELAY,
    }[audio.application or "audio"]
    encoder = opuslib_next.Encoder(audio.sample_rate, 1, application)
    if audio.bitrate is not None:
        encoder.bitrate = audio.bitrate
    if audio.complexity is not None:
        encoder.complexity = audio.complexity
    return encoder


def encode_packet(pcm: bytes, audio: AudioDirection, encoder: opuslib_next.Encoder) -> bytes:
    if len(pcm) != audio.packet_bytes:
        raise ProtocolError("incorrect output PCM packet length")
    packet = encoder.encode(pcm, audio.packet_bytes // 2)
    validate_input_packet(packet, audio)
    return packet
