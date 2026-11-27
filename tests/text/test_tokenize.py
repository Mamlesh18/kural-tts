from kural_tts.text.tokenize import tokenize


def kinds(text):
    return [(t.text, t.kind) for t in tokenize(text)]


def test_number_shapes():
    assert kinds("₹1,50,000 at 10:30 AM, 8.5% on 15th") == [
        ("₹1,50,000", "currency"),
        ("at", "latin"),
        ("10:30", "time"),
        ("AM", "latin"),
        (",", "punct"),
        ("8.5%", "percent"),
        ("on", "latin"),
        ("15th", "ordinal"),
    ]


def test_phone_and_email():
    assert kinds("call 044-2345-6789 or mail support@example.com.") == [
        ("call", "latin"),
        ("044-2345-6789", "phone"),
        ("or", "latin"),
        ("mail", "latin"),
        ("support@example.com", "email"),
        (".", "punct"),
    ]


def test_latin_words_keep_apostrophes_and_hyphens():
    assert [t.text for t in tokenize("I'll reset the Wi-Fi")] == ["I'll", "reset", "the", "Wi-Fi"]


def test_attached_tamil_suffixes():
    toks = tokenize("ஆகஸ்ட் 15ஆம் தேதி office-ல GB-க்கு")
    assert [(t.text, t.attached) for t in toks] == [
        ("ஆகஸ்ட்", False),
        ("15", False),
        ("ஆம்", True),
        ("தேதி", False),
        ("office", False),
        ("ல", True),
        ("GB", False),
        ("க்கு", True),
    ]


def test_spaced_hyphen_is_not_a_suffix_join():
    toks = tokenize("Chennai - மதுரை")
    assert [t.kind for t in toks] == ["latin", "punct", "ta"]
    assert not toks[-1].attached


def test_space_before_flags():
    toks = tokenize("வணக்கம், Sir!")
    assert [(t.text, t.space_before) for t in toks] == [
        ("வணக்கம்", False),
        (",", False),
        ("Sir", True),
        ("!", False),
    ]
