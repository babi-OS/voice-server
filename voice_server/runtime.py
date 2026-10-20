"""Validated process and native-thread settings for CPU inference."""
import os


def integer_setting(name: str, default: int, *, maximum: int = 32) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer from 1 through {maximum}") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer from 1 through {maximum}")
    return value


def session_options(threads: int):
    import onnxruntime as ort

    spinning = os.environ.get("VOICE_ORT_SPINNING", "1")
    if spinning not in ("0", "1"):
        raise ValueError("VOICE_ORT_SPINNING must be 0 or 1")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", spinning)
    options.add_session_config_entry("session.inter_op.allow_spinning", spinning)
    return options
