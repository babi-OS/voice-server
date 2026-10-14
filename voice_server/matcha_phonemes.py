"""Pinned SEA-G2P character inventory shared by Matcha training and inference."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
from pathlib import Path

import regex

LEGACY_TOKENIZER = "ipa_codepoints_v1"
UNIT_TOKENIZER = "ipa_units_v1"
FACTORIZED_TOKENIZER = "ipa_factorized_v1"
TOKENIZERS = {LEGACY_TOKENIZER, UNIT_TOKENIZER, FACTORIZED_TOKENIZER}
_LEADING = {"ˈ", "ˌ"}
_TRAILING = set("0123456789ːˑʰʷʲ˞")
STRESS = ("", "ˈ", "ˌ")
TONE = ("", *tuple("0123456789"))
LENGTH = ("", "ː", "ˑ")


def ipa_units(phonemes: str) -> list[str]:
    """Keep IPA diacritics and suprasegmentals on the sound they modify."""
    units: list[str] = []
    leading = ""
    for cluster in regex.findall(r"\X", phonemes):
        if cluster.isspace():
            if leading:
                raise ValueError("stress marker without a following sound")
            if units and units[-1] != " ":
                units.append(" ")
        elif cluster in _LEADING:
            leading += cluster
        elif cluster in _TRAILING:
            if leading or not units or units[-1] == " ":
                raise ValueError(f"modifier without a preceding sound: {cluster!r}")
            units[-1] += cluster
        elif regex.fullmatch(r"\p{P}+", cluster):
            if leading:
                raise ValueError("stress marker before punctuation")
            units.append(cluster)
        else:
            if units and units[-1].endswith(("͡", "͜")):
                units[-1] += leading + cluster
            else:
                units.append(leading + cluster)
            leading = ""
    if leading:
        raise ValueError("stress marker without a following sound")
    if units and units[-1] == " ":
        units.pop()
    return units


def factorized_units(phonemes: str) -> list[tuple[str, int, int, int]]:
    """One position per phone; suprasegmentals are separate embedding features."""
    result = []
    for unit in ipa_units(phonemes):
        if unit == " " or regex.fullmatch(r"\p{P}+", unit):
            result.append((unit, 0, 0, 0))
            continue
        stress = 0
        if unit[0] in _LEADING:
            stress = STRESS.index(unit[0])
            unit = unit[1:]
        tone = 0
        length = 0
        base = ""
        for char in unit:
            if char in "0123456789":
                if tone:
                    raise ValueError(f"multiple tone marks in {unit!r}")
                tone = TONE.index(char)
            elif char in ("ː", "ˑ"):
                if length:
                    raise ValueError(f"multiple length marks in {unit!r}")
                length = LENGTH.index(char)
            else:
                base += char
        if not base:
            raise ValueError(f"empty base phone in {unit!r}")
        result.append((base, stress, tone, length))
    return result


def pack_factorized(base: int, stress: int, tone: int, length: int, n_vocab: int) -> int:
    if not (0 <= base < n_vocab and 0 <= stress < len(STRESS)
            and 0 <= tone < len(TONE) and 0 <= length < len(LENGTH)):
        raise ValueError("factorized phone outside vocabulary")
    return base + n_vocab * (stress + len(STRESS) * (tone + len(TONE) * length))


def verify_checkpoint_frontend(checkpoint: dict, vocab_path: Path) -> None:
    """Fail closed when a saved acoustic checkpoint uses another tokenizer."""
    metadata = checkpoint.get("frontend")
    digest = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    if metadata is None or metadata.get("vocab_sha256") != digest:
        raise RuntimeError("checkpoint vocabulary does not match inference frontend")
    payload = json.loads(vocab_path.read_text())
    if metadata.get("vocab", {}).get("tokenizer_version") != payload.get("tokenizer_version"):
        raise RuntimeError("checkpoint tokenizer version does not match inference frontend")


class PhonemeFrontend:
    def __init__(self, vocab_path: Path):
        from sea_g2p import SEAPipeline

        payload = json.loads(Path(vocab_path).read_text())
        installed = importlib.metadata.version("sea-g2p")
        if payload["sea_g2p_version"] != installed:
            raise RuntimeError(f"SEA-G2P version mismatch: {installed} != {payload['sea_g2p_version']}")
        self.symbols = payload["symbols"]
        self.tokenizer_version = payload.get("tokenizer_version", LEGACY_TOKENIZER)
        if self.tokenizer_version not in TOKENIZERS:
            raise ValueError(f"unsupported tokenizer: {self.tokenizer_version}")
        if not self.symbols or self.symbols[0] != "<pad>" or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("invalid phoneme vocabulary")
        self.lookup = {symbol: index for index, symbol in enumerate(self.symbols)}
        if self.tokenizer_version == FACTORIZED_TOKENIZER and payload.get("features") != {
                "stress": list(STRESS), "tone": list(TONE), "length": list(LENGTH)}:
            raise ValueError("factorized feature vocabulary mismatch")
        self.pipeline = SEAPipeline(lang="vi")

    def phonemes(self, text: str) -> str:
        result = self.pipeline.run(text)
        if not result or not result.strip():
            raise ValueError("SEA-G2P returned empty phonemes")
        return result

    def encode_phonemes(self, phonemes: str) -> list[int]:
        features = factorized_units(phonemes) if self.tokenizer_version == FACTORIZED_TOKENIZER else None
        units = [unit[0] for unit in features] if features is not None else self.tokenize(phonemes)
        missing = sorted(set(units).difference(self.lookup))
        if missing:
            raise ValueError(f"unknown phoneme symbols: {missing}")
        if features is not None:
            ids = [pack_factorized(self.lookup[base], stress, tone, length, len(self.symbols))
                   for base, stress, tone, length in features]
        else:
            ids = [self.lookup[symbol] for symbol in units]
        interspersed = [0] * (2 * len(ids) + 1)
        interspersed[1::2] = ids
        return interspersed

    def encode(self, text: str) -> list[int]:
        return self.encode_phonemes(self.phonemes(text))

    def tokenize(self, phonemes: str) -> list[str]:
        if self.tokenizer_version == LEGACY_TOKENIZER:
            return list(phonemes)
        if self.tokenizer_version == FACTORIZED_TOKENIZER:
            return [unit[0] for unit in factorized_units(phonemes)]
        return ipa_units(phonemes)

    def prompt_length(self, text: str, word_count: int, full_phonemes: str | None = None) -> int:
        words = text.split()
        if not 0 < word_count < len(words):
            return 0
        full = full_phonemes or self.phonemes(text)
        prefix = self.phonemes(" ".join(words[:word_count]))
        # Punctuation at the end can change the final token's pronunciation.
        if not full.startswith(prefix):
            return 0
        prefix_ids = self.encode_phonemes(prefix)
        if self.encode_phonemes(full)[:len(prefix_ids)] != prefix_ids:
            return 0
        return len(prefix_ids)


def build_vocab(manifest: Path, output: Path, evaluation: list[Path],
                tokenizer_version: str = LEGACY_TOKENIZER) -> dict:
    import hashlib

    from sea_g2p import SEAPipeline

    version = importlib.metadata.version("sea-g2p")
    if tokenizer_version not in TOKENIZERS:
        raise ValueError(f"unsupported tokenizer: {tokenizer_version}")
    pipeline = SEAPipeline(lang="vi")
    symbols = set()
    counts = {"training": 0, "evaluation": 0}
    paths = [(manifest, "training"), *((path, "evaluation") for path in evaluation)]
    for path, kind in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row["text"]
            result = pipeline.run(text)
            if not result.strip():
                raise ValueError(f"empty SEA-G2P output for {row.get('id', text)}")
            if tokenizer_version == LEGACY_TOKENIZER:
                symbols.update(result)
            elif tokenizer_version == FACTORIZED_TOKENIZER:
                symbols.update(base for base, *_ in factorized_units(result))
            else:
                symbols.update(ipa_units(result))
            counts[kind] += 1
    payload = {"symbols": ["<pad>", *sorted(symbols)],
               "tokenizer_version": tokenizer_version,
               "sea_g2p_version": version, "language": "vi",
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "counts": counts}
    if tokenizer_version == FACTORIZED_TOKENIZER:
        payload["features"] = {"stress": list(STRESS), "tone": list(TONE), "length": list(LENGTH)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return {"vocab_size": len(payload["symbols"]), "counts": counts}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default=LEGACY_TOKENIZER)
    args = parser.parse_args()
    print(json.dumps(build_vocab(args.manifest, args.out, args.eval, args.tokenizer), indent=2))
