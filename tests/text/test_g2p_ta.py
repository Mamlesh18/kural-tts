import pytest

from kural_tts.text.g2p_ta import Unit, g2p_ta_word, parse
from kural_tts.text.symbols import TA_PHONES


def test_parse_units():
    assert parse("நான்") == [Unit("ந", "ஆ"), Unit("ன", None)]
    assert parse("அது") == [Unit(None, "அ"), Unit("த", "உ")]


@pytest.mark.parametrize(
    "word,expected",
    [
        # word-initial stops; initial ச -> s (Chennai)
        ("பால்", "p aː l"),
        ("சரி", "s a ɾ i"),
        ("தமிழ்", "t̪ a m i ɻ"),
        # between vowels
        ("மகன்", "m a ɣ a n"),
        ("அது", "a ð ɯ"),
        ("பசி", "p a s i"),
        ("படம்", "p a ɖ a m"),
        # geminates stay voiceless
        ("பக்கம்", "p a k k a m"),
        ("பச்சை", "p a tʃ tʃ ai"),
        ("பத்து", "p a t̪ t̪ ɯ"),
        # voiced after nasals
        ("எங்க", "e ŋ ɡ a"),
        ("கஞ்சி", "k a ɲ dʒ i"),
        ("பண்டம்", "p a ɳ ɖ a m"),
        ("வந்து", "ʋ a n̪ d̪ ɯ"),
        ("அம்பு", "a m b ɯ"),
        # voiced after sonorants
        ("மார்கழி", "m aː ɾ ɡ a ɻ i"),
        # ற / ன்ற / ஃப / grantha
        ("வெற்றி", "ʋ e t r i"),
        ("ஒன்று", "o n d r ɯ"),
        ("காஃபி", "k aː f i"),
        ("ஸ்ரீ", "s ɾ iː"),
        ("ஜனவரி", "dʒ a n a ʋ a ɾ i"),
        # long vowels, ழ, diphthong
        ("வாழைப்பழம்", "ʋ aː ɻ ai p p a ɻ a m"),
        ("நான்", "n̪ aː n"),
        # standalone suffix glued to an English word
        ("க்கு", "k k ɯ"),
    ],
)
def test_g2p_ta_word(word, expected):
    assert " ".join(g2p_ta_word(word)) == expected


def test_all_outputs_in_inventory():
    words = [
        "இன்று", "சென்னையில்", "லேசான", "மழை", "பெய்ய",
        "வாய்ப்பு", "உள்ளது", "திறந்திருக்கும்", "ஔவை", "அஃது",
    ]  # fmt: skip
    for w in words:
        assert set(g2p_ta_word(w)) <= set(TA_PHONES), w
