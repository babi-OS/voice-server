""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re
import platform
import phonemizer
from unidecode import unidecode
from underthesea import text_normalize
from num2words import num2words

# To avoid excessive logging we set the log level of the phonemizer package to Critical
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Intializing the phonemizer globally significantly reduces the speed
global_phonemizer = None

_ESPEAK_VI = None

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    global global_phonemizer
    if global_phonemizer is None:
        global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True,
            language_switch="remove-flags", logger=critical_logger,
        )
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


_NAM_20xx = re.compile(r"^20\d{2}$")
_CHU_SO = re.compile(r"^\d+$")


def basic_cleaners_ngngngan(text: str) -> str:
    """normalize then phonemize for Vietnamese MC Nguyen Ngoc Ngan"""
    global _ESPEAK_VI
    if _ESPEAK_VI is None:
        if platform.system().lower() == "windows" and not phonemizer.backend.EspeakBackend.is_available():
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
            EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
        _ESPEAK_VI = phonemizer.backend.EspeakBackend(
            "vi", 
            preserve_punctuation=True, 
            language_switch="remove-flags", 
            with_stress=True, 
            tie=True,
            logger=critical_logger
        )

    txt = text_normalize(text).replace("-", " ")
    text_list = []
    for word in txt.split():
        if word == "%":
            text_list.extend(["phần", "trăm"])
        elif word == "&":
            text_list.append("và")
        elif _NAM_20xx.match(word) is not None:
            num = "hai ngàn không trăm " + num2words(int(word[-2:]), lang="vi")
            text_list.extend(num.split(" "))
        elif _CHU_SO.match(word) is not None:
            num = num2words(int(word), lang="vi").replace("nghìn", "ngàn")
            text_list.extend(num.split(" "))
        elif word in ".,;!?":
            if text_list:
                text_list[-1] += word
            else:
                text_list.append(word)
        else:
            text_list.append(word)
            
    ipa_list = _ESPEAK_VI.phonemize(text_list, strip=True)
    return " ".join(ipa_list).strip()


def vietnamese_grapheme_cleaners(text: str) -> str:
    """Normalize Vietnamese text but return graphemes (normal letters) instead of IPA phonemes."""
    import unicodedata
    # NFC normalization to merge decomposed combining accents
    text = unicodedata.normalize("NFC", text)
    
    # Pre-clean some symbols/abbreviations
    text = text.replace("%", " phần trăm ")
    text = text.replace("/", " trên ")
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("~", " ")
    
    # Map special Sanskrit/non-standard letters to standard ones
    sanskrit_map = {
        "ṭ": "t",
        "ṇ": "n",
        "ṃ": "m",
        "ē": "e",
        "ō": "o",
        "ā": "a",
        "ī": "i",
        "ū": "u",
    }
    for k, v in sanskrit_map.items():
        text = text.replace(k, v)
        text = text.replace(k.upper(), v.upper())

    txt = text_normalize(text).replace("-", " ")
    text_list = []
    for word in txt.split():
        if word == "%":
            text_list.extend(["phần", "trăm"])
        elif word == "&":
            text_list.append("và")
        elif _NAM_20xx.match(word) is not None:
            num = "hai ngàn không trăm " + num2words(int(word[-2:]), lang="vi")
            text_list.extend(num.split(" "))
        elif _CHU_SO.match(word) is not None:
            num = num2words(int(word), lang="vi").replace("nghìn", "ngàn")
            text_list.extend(num.split(" "))
        elif word in ".,;!?":
            if text_list:
                text_list[-1] += word
            else:
                text_list.append(word)
        else:
            text_list.append(word)
            
    # Skip IPA conversion, just join the words and lowercase
    cleaned = " ".join(text_list).strip().lower()
    
    # Map common unsupported characters/punctuation to standard equivalents
    replacements = {
        "–": "-",  # en-dash to hyphen
        "‘": "'",  # curly single quote
        "’": "'",  # curly single quote
        "′": "'",  # prime to single quote
        "“": "\"", # curly double quote
        "”": "\"", # curly double quote
        "\u200b": "", # zero-width space
        "\u200c": "", # zero-width non-joiner
        "\u200d": "", # zero-width joiner
        "\ufeff": "", # byte order mark
    }
    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)
        
    # Filter to only keep characters that exist in symbols
    from voice_server.matcha.text.symbols import symbols
    allowed = set(symbols)
    cleaned = "".join([c for c in cleaned if c in allowed])
    
    return cleaned
