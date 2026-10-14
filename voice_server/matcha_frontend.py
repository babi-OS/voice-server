"""Inference text normalization matching Matchtts_VI's grapheme training path.

Adapted from MIT-licensed `matcha/text/cleaners.py` at upstream commit
dd75df84f7216c88c08c649208f1e7fda9a62f61. The exported model keeps its
existing character vocabulary; this is not SEA-G2P or an IPA conversion.
"""

from __future__ import annotations

import re
import unicodedata


def clean_vietnamese_graphemes(text: str, symbols: set[str]) -> str:
    try:
        from num2words import num2words
    except ImportError as exc:
        raise RuntimeError("Matcha student needs num2words for its training text frontend") from exc

    text = unicodedata.normalize("NFC", text)
    text = text.replace("%", " phần trăm ").replace("/", " trên ")
    text = text.replace("–", "-").replace("—", "-").replace("~", " ")
    for old, new in {"ṭ": "t", "ṇ": "n", "ṃ": "m", "ē": "e", "ō": "o",
                     "ā": "a", "ī": "i", "ū": "u"}.items():
        text = text.replace(old, new).replace(old.upper(), new.upper())
    # underthesea.text_normalize separated these punctuation marks in all
    # 2,280 checked training/evaluation/benchmark prompts. Keeping this small
    # normalization local avoids loading its large NLP stack in every worker.
    normalized = re.sub(r"([.,;!?])(?=\s|$)", r" \1 ", text)
    normalized = normalized.replace(":", " : ").replace("-", " ")
    normalized = " ".join(normalized.split())
    words = []
    for word in normalized.split():
        if word == "%":
            words.extend(("phần", "trăm"))
        elif word == "&":
            words.append("và")
        elif re.fullmatch(r"20\d{2}", word):
            words.extend(("hai ngàn không trăm " + num2words(int(word[-2:]), lang="vi")).split())
        elif re.fullmatch(r"\d+", word):
            words.extend(num2words(int(word), lang="vi").replace("nghìn", "ngàn").split())
        elif word in ".,;!?":
            if words:
                words[-1] += word
            else:
                words.append(word)
        else:
            words.append(word)
    cleaned = " ".join(words).strip().lower()
    cleaned = (cleaned.replace("–", "-").replace("‘", "'").replace("’", "'")
               .replace("′", "'").replace("“", '"').replace("”", '"'))
    for character in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        cleaned = cleaned.replace(character, "")
    return "".join(character for character in cleaned if character in symbols)
