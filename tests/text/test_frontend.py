import pytest

from kural_tts.text.frontend import Span, normalize, normalize_text
from kural_tts.text.tamil_script import has_tamil


@pytest.mark.parametrize(
    "text,expected",
    [
        ("நாளைக்கு meeting 10 மணிக்கு இருக்கு.", "நாளைக்கு meeting பத்து மணிக்கு இருக்கு."),
        ("உங்கள் கட்டணம் 2,499 ரூபாய்.", "உங்கள் கட்டணம் இரண்டாயிரத்து நானூற்று தொண்ணூற்றொன்பது ரூபாய்."),
        ("உங்கள் சந்திப்பு ஆகஸ்ட் 15ஆம் தேதி.", "உங்கள் சந்திப்பு ஆகஸ்ட் பதினைந்தாம் தேதி."),
        ("இந்த நூலில் ௨௫ அத்தியாயங்கள் உள்ளன.", "இந்த நூலில் இருபத்தைந்து அத்தியாயங்கள் உள்ளன."),
        ("டாக். ரமேஷ் இன்று வருவார்.", "டாக்டர் ரமேஷ் இன்று வருவார்."),
        ("The approved loan amount is ₹1,50,000.",
         "The approved loan amount is one lakh fifty thousand rupees."),
        ("Dr. Ramesh will see you on the 2nd floor.",
         "Doctor Ramesh will see you on the second floor."),
        ("Your appointment is on 15th August 2026 at 10:30 AM.",
         "Your appointment is on fifteenth August two thousand twenty-six at ten thirty A M."),
        ("naalaikku office ku varen, ok ah?", "நாளைக்கு officeக்கு வரேன், okஆ?"),
    ],
)  # fmt: skip
def test_normalize_text(text, expected):
    assert normalize_text(text) == expected


def test_spans_for_code_switched_sentence():
    assert normalize("Traffic ரொம்ப அதிகமா இருக்கு, so I'll be late.") == [
        Span("Traffic", "en"),
        Span("ரொம்ப அதிகமா இருக்கு,", "ta"),
        Span("so I'll be late.", "en"),
    ]


def test_spans_tanglish_with_english():
    spans = normalize("konjam wait pannunga")
    assert spans == [Span("கொஞ்சம்", "ta"), Span("wait", "en"), Span("பண்ணுங்க", "ta")]


def test_explicit_english_mode():
    assert normalize("romba nalla", lang="en") == [Span("romba nalla", "en")]


def test_spans_never_mix_scripts():
    text = "Mr. Senthil Kumar உங்களை 3 மணிக்கு call பண்ணச் சொன்னார். OTP 5 minutes valid."
    for span in normalize(text):
        if span.lang == "en":
            assert not has_tamil(span.text), span
        else:
            assert not any(c.isascii() and c.isalpha() for c in span.text), span


def test_tanglish_particle_attaches_to_number():
    assert normalize("naalaikku 10:30 ku varen")[0] == Span("நாளைக்கு பத்தரைக்கு வரேன்", "ta")
