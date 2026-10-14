"""Preserve Matchtts_VI grapheme IDs, adding namespaced SEA-G2P English phones."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import regex
from voice_server.matcha.text import text_to_sequence
from voice_server.matcha.text.symbols import symbols as GRAPHEME_SYMBOLS

from voice_server.matcha_phonemes import ipa_units

BASE_SIZE = 244  # Published checkpoint embedding rows; 14 are unused by the cleaner.
BASE_SYMBOLS = [*GRAPHEME_SYMBOLS, *(f"<unused_{index}>" for index in range(len(GRAPHEME_SYMBOLS), BASE_SIZE))]
BASE_LOOKUP = {symbol: index for index, symbol in enumerate(GRAPHEME_SYMBOLS)}
WORD = regex.compile(r"\p{L}+(?:'\p{L}+)*")
BASE_VERSION = "vi_grapheme_original_v1"
HYBRID_VERSION = "vi_grapheme_sea_english_v1"


def cleaned_text(text: str) -> str:
    return text_to_sequence(text, ["vietnamese_grapheme_cleaners"])[1]


def english_words(cleaned: str, language: str, terms: list[str] | None) -> list[re.Match]:
    words = list(WORD.finditer(cleaned))
    if language == "en":
        return [word for word in words if re.fullmatch(r"[a-z]+(?:'[a-z]+)*", word.group())]
    if language != "mixed":
        return []
    if not terms:
        raise ValueError("mixed text requires an explicit English term")
    selected = set()
    for term in terms:
        target = cleaned_text(term).strip()
        matches = list(re.finditer(r"(?<![a-z])" + re.escape(target) + r"(?![a-z])", cleaned))
        if len(matches) != 1:
            raise ValueError(f"English term must occur exactly once: {term!r}")
        match = matches[0]
        selected.update(word.start() for word in words
                        if match.start() <= word.start() and word.end() <= match.end()
                        and re.fullmatch(r"[a-z]+(?:'[a-z]+)*", word.group()))
    if not selected:
        raise ValueError("no English words found in tagged terms")
    return [word for word in words if word.start() in selected]


def vocabulary(english_units: set[str], hybrid: bool) -> dict:
    if len(GRAPHEME_SYMBOLS) > BASE_SIZE:
        raise RuntimeError(f"published Matcha has fewer rows than grapheme symbols: {len(GRAPHEME_SYMBOLS)}")
    english = ["<en>", "</en>", *("EN:" + unit for unit in sorted(english_units))] if hybrid else []
    return {"symbols": [*BASE_SYMBOLS, *english], "base_size": BASE_SIZE,
            "tokenizer_version": HYBRID_VERSION if hybrid else BASE_VERSION,
            "sea_g2p_version": importlib.metadata.version("sea-g2p") if hybrid else None,
            "english_processing": "explicit tagged words, isolated-word SEA-G2P lang=vi" if hybrid else "graphemes"}


class HybridFrontend:
    def __init__(self, vocab_path: Path):
        self.path = Path(vocab_path)
        self.payload = json.loads(self.path.read_text())
        self.symbols = self.payload["symbols"]
        self.version = self.payload["tokenizer_version"]
        if self.version not in {BASE_VERSION, HYBRID_VERSION} or self.symbols[:BASE_SIZE] != BASE_SYMBOLS:
            raise ValueError("vocabulary does not preserve published grapheme IDs")
        if self.version == HYBRID_VERSION and self.payload["sea_g2p_version"] != importlib.metadata.version("sea-g2p"):
            raise ValueError("SEA-G2P version changed")
        self.lookup = {symbol: index for index, symbol in enumerate(self.symbols)}
        if len(self.lookup) != len(self.symbols):
            raise ValueError("duplicate vocabulary symbol")
        if self.version == HYBRID_VERSION:
            from sea_g2p import SEAPipeline

            self.g2p = SEAPipeline(lang="vi")
        else:
            self.g2p = None
        self.phone_cache: dict[str, list[int]] = {}

    def _phones(self, word: str) -> list[int]:
        if word not in self.phone_cache:
            phonemes = self.g2p.run(word)
            units = ipa_units(phonemes)
            if not units:
                raise ValueError(f"empty English pronunciation: {word!r}")
            names = ["<en>", *("EN:" + unit for unit in units), "</en>"]
            missing = set(names).difference(self.lookup)
            if missing:
                raise ValueError(f"unencodable English phones for {word!r}: {sorted(missing)}")
            self.phone_cache[word] = [self.lookup[name] for name in names]
        return self.phone_cache[word]

    def encode(self, text: str, language: str = "vi", english_terms: list[str] | None = None) -> list[int]:
        if language not in {"vi", "mixed", "en"}:
            raise ValueError(f"unknown language: {language}")
        if self.version == BASE_VERSION:
            ids, _ = text_to_sequence(text, ["vietnamese_grapheme_cleaners"])
        else:
            cleaned = cleaned_text(text)
            marked = {match.start(): match for match in english_words(cleaned, language, english_terms)}
            ids = []
            index = 0
            while index < len(cleaned):
                word = marked.get(index)
                if word:
                    ids.extend(self._phones(word.group()))
                    index = word.end()
                else:
                    symbol = cleaned[index]
                    if symbol not in BASE_LOOKUP:
                        raise ValueError(f"unencodable Vietnamese grapheme: {symbol!r}")
                    ids.append(BASE_LOOKUP[symbol])
                    index += 1
        if not ids:
            raise ValueError("empty text encoding")
        interspersed = [0] * (2 * len(ids) + 1)
        interspersed[1::2] = ids
        return interspersed

    def prompt_length(self, text: str, word_count: int, language: str = "vi",
                      english_terms: list[str] | None = None) -> int:
        # The matched pilot uses 100% no-prompt training. Keep strict prefix
        # handling for future prompt-conditioned reuse.
        matches = list(re.finditer(r"\S+", text))
        if not 0 < word_count < len(matches):
            return 0
        prefix = text[:matches[word_count - 1].end()]
        if language == "en":
            prefix_terms = None
            prefix_language = "en"
        elif language == "mixed":
            cleaned_prefix = cleaned_text(prefix)
            full_cleaned = cleaned_text(text)
            if not full_cleaned.startswith(cleaned_prefix):
                return 0
            selected = english_words(full_cleaned, "mixed", english_terms)
            prefix_terms = [match.group() for match in selected if match.end() <= len(cleaned_prefix)]
            prefix_language = "mixed" if prefix_terms else "vi"
        else:
            prefix_terms, prefix_language = None, "vi"
        prefix_ids = self.encode(prefix, prefix_language, prefix_terms)
        full_ids = self.encode(text, language, english_terms)
        return len(prefix_ids) if full_ids[:len(prefix_ids)] == prefix_ids else 0


def checkpoint_metadata(vocab: Path) -> dict:
    data = vocab.read_bytes()
    return {"vocab_sha256": hashlib.sha256(data).hexdigest(), "vocab": json.loads(data)}
