from dataclasses import dataclass
from time import monotonic
@dataclass
class Clock:
    def now(self): return monotonic()
@dataclass
class SynthChunk:
    samples: list[float]
    sample_rate: int
    timestamp: float
@dataclass
class SynthOutput:
    chunks: list[SynthChunk]
    sample_rate: int
    streaming: bool
    notes: list[str]
    @property
    def samples(self):
        return self.chunks[0].samples if self.chunks else []
class TtsAdapter: pass
