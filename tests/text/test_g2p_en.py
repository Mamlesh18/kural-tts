import pytest

from kural_tts.text.g2p_en import LETTERS, arpabet_to_ipa, g2p_en_word, load_cmudict
from kural_tts.text.symbols import SYMBOLS


def phones(word):
    return " ".join(p for p, _ in g2p_en_word(word))


def langs(word):
    return {lang for _, lang in g2p_en_word(word)}


def test_cmudict_loads():
    lex = load_cmudict()
    assert len(lex) > 120_000
    assert lex["meeting"] == "M IY1 T IH0 NG"


def test_arpabet_to_ipa_stress_and_reduction():
    assert arpabet_to_ipa("M IY1 T IH0 NG") == ["m", "ˈ", "i", "t", "ɪ", "ŋ"]
    assert arpabet_to_ipa("AH0 B AW1 T") == ["ə", "b", "ˈ", "aʊ", "t"]
    assert arpabet_to_ipa("B ER1 D ER0") == ["b", "ˈ", "ɝ", "d", "ɚ"]


@pytest.mark.parametrize(
    "word,expected",
    [
        ("meeting", "m ˈ i t ɪ ŋ"),
        ("Meeting", "m ˈ i t ɪ ŋ"),
        ("I'll", "ˈ aɪ l"),
        ("Wi-Fi", "w ˈ aɪ f ˌ aɪ"),
        ("Z", "z ˈ ɛ d"),  # Indian English "zed"
        ("S", "ˈ ɛ s"),
    ],
)
def test_known_words(word, expected):
    assert phones(word) == expected
    assert langs(word) == {"en"}


def test_hyphenated_compound_splits():
    assert phones("passport-size") == phones("passport") + " " + phones("size")


def test_unknown_indian_name_goes_through_tamil_g2p():
    assert langs("Velachery") == {"ta"}
    assert phones("Velachery").startswith("ʋ e l")


def test_all_letter_names_and_outputs_in_inventory():
    inventory = set(SYMBOLS)
    for letter in LETTERS:
        assert {p for p, _ in g2p_en_word(letter)} <= inventory
    for word in ["throughout", "pleasure", "choice", "Tiruchirappalli", "Aadhaar"]:
        assert {p for p, _ in g2p_en_word(word)} <= inventory, word
