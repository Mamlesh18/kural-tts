from kural_tts.text.cleanup import clean
from kural_tts.text.langid import assign_languages
from kural_tts.text.tokenize import tokenize


def tag(text, mode="auto"):
    return [(t.text, t.lang) for t in assign_languages(tokenize(clean(text)), mode) if t.lang]


def test_tamil_script_and_english():
    assert tag("நாளைக்கு meeting இருக்கு") == [
        ("நாளைக்கு", "ta"),
        ("meeting", "en"),
        ("இருக்கு", "ta"),
    ]


def test_tanglish_is_transliterated():
    assert tag("konjam wait pannunga") == [
        ("கொஞ்சம்", "ta"),
        ("wait", "en"),
        ("பண்ணுங்க", "ta"),
    ]


def test_ambiguous_word_resolved_by_context():
    assert tag("pal vali romba iruku")[0] == ("பல்", "ta")
    assert tag("my pal is here") == [("my", "en"), ("pal", "en"), ("is", "en"), ("here", "en")]


def test_ambiguous_with_names_falls_back_to_sentence_tanglish():
    tags = tag("naan Chennai la Velachery la irukken")
    assert tags[0] == ("நான்", "ta")
    assert ("Chennai", "en") in tags and ("Velachery", "en") in tags
    assert tags[-1] == ("இருக்கேன்", "ta")


def test_particles_attach():
    toks = assign_languages(tokenize("naalaikku office ku varen"))
    ku = toks[2]
    assert ku.text == "க்கு" and ku.attached and ku.source == "tanglish"


def test_unknown_tamil_looking_word_needs_context():
    assert tag("pannittaanga")[0][1] == "en"  # no supporting context
    assert tag("avanga already pannittaanga")[-1][1] == "ta"


def test_number_language_from_context():
    assert tag("meeting 10 மணிக்கு")[1] == ("10", "ta")
    assert tag("Your appointment is at 10:30 AM")[4] == ("10:30", "en")
    assert tag("ஆகஸ்ட் 5ஆம் தேதி")[1] == ("5", "ta")
    assert tag("Total ₹1,250 ஆகும்")[1] == ("₹1,250", "ta")


def test_english_mode_disables_tanglish():
    assert tag("romba nalla", mode="en") == [("romba", "en"), ("nalla", "en")]


def test_ambiguous_particle_uses_sentence_level_tanglish():
    assert tag("naalaikku 10:30 ku meeting, ok ah?")[-1] == ("ஆ", "ta")
    assert tag("ok ah, see you")[1] == ("ah", "en")  # no Tanglish in the sentence


def test_hyphenated_tanglish_particle_is_split_and_attached():
    toks = [t for t in assign_languages(tokenize("unga account-la KYC")) if t.lang]
    assert [(t.text, t.lang, t.attached) for t in toks] == [
        ("உங்க", "ta", False),
        ("account", "en", False),
        ("ல", "ta", True),
        ("KYC", "en", False),
    ]
    # Ordinary hyphenated English words are untouched.
    assert tag("passport-size photo")[0] == ("passport-size", "en")
