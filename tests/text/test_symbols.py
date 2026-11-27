import pytest

from kural_tts.text.symbols import (
    EN_PHONES,
    LANG_TO_ID,
    SYMBOL_TO_ID,
    SYMBOLS,
    SYMBOLS_VERSION,
    TA_PHONES,
    encode,
)


def test_symbols_unique_and_complete():
    assert len(SYMBOLS) == len(set(SYMBOLS))
    assert set(EN_PHONES) <= set(SYMBOLS) and set(TA_PHONES) <= set(SYMBOLS)


def test_ids_are_frozen_for_version_1():
    # Checkpoints depend on these ids. If this fails you reordered symbols: append instead.
    assert SYMBOLS_VERSION == 1
    assert SYMBOLS[:10] == ["_", "^", "$", " ", ",", ".", "?", "!", "ˈ", "ˌ"]
    assert len(SYMBOLS) == 75
    assert SYMBOL_TO_ID["ɑ"] == 10
    assert SYMBOLS[-1] == "ɭ"


def test_encode_wraps_with_bos_eos():
    ids, langs = encode([("n̪", "ta"), ("aː", "ta"), ("n", "ta"), (" ", "ta"), ("h", "en")])
    assert ids[0] == SYMBOL_TO_ID["^"] and ids[-1] == SYMBOL_TO_ID["$"]
    assert langs == [LANG_TO_ID["ta"]] * 5 + [LANG_TO_ID["en"]] * 2


def test_encode_rejects_unknown_symbol():
    with pytest.raises(KeyError):
        encode([("q", "en")])
