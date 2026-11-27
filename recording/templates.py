"""Templated prompts for numbers, dates, times, currency and phone numbers (TA / EN / CS).

These cover what a voice agent says constantly and what free text rarely contains in volume.
Values are random but seeded, so the same seed always produces the same prompts.
"""

from __future__ import annotations

import random

MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]  # fmt: skip
MONTHS_TA = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்",
             "செப்டம்பர்", "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]  # fmt: skip

TEMPLATES: dict[str, list[str]] = {
    "en": [
        "Your total amount is ₹{amount}.",
        "The refund of ₹{amount} will reach your account in {small} working days.",
        "Your appointment is on {day} {month_en} at {time} {ampm}.",
        "Please call us at {phone}.",
        "Your OTP is {otp}. Do not share it with anyone.",
        "The interest rate is {pct}% per annum.",
        "Your order number is {otp}, and it will arrive by {day} {month_en}.",
        "We have {small} seats left for the {time} {ampm} show.",
    ],
    "ta": [
        "உங்கள் மொத்த தொகை {amount} ரூபாய்.",
        "உங்கள் சந்திப்பு {month_ta} {day}ஆம் தேதி மாலை {time} மணிக்கு.",
        "எங்களை {phone} என்ற எண்ணில் அழைக்கவும்.",
        "வட்டி விகிதம் {pct} சதவீதம்.",
        "இன்னும் {small} நாட்களில் உங்கள் பணம் திரும்ப வரும்.",
        "உங்க ஆர்டர் {month_ta} {day}ஆம் தேதி வந்துடும்.",
    ],
    "cs": [
        "உங்க total amount ₹{amount}, UPI மூலமா pay பண்ணலாம்.",
        "நாளைக்கு meeting {time} மணிக்கு இருக்கு.",
        "உங்க OTP {otp}, யாருக்கும் share பண்ணாதீங்க.",
        "Your appointment {month_ta} {day}ஆம் தேதி confirm ஆயிடுச்சு.",
        "Refund {small} working days-ல உங்க account-க்கு வந்துடும்.",
        "Customer care number {phone}, எப்போ வேணாலும் call பண்ணலாம்.",
    ],
}


def _amount(rng: random.Random) -> str:
    n = rng.choice(
        [rng.randint(10, 999), rng.randint(1000, 99_999), rng.randint(100_000, 9_999_999)]
    )
    s = str(n)
    if n < 1000:
        return s
    head, tail = s[:-3], s[-3:]
    groups = []
    while len(head) > 2:  # Indian grouping: 12,34,567
        groups.insert(0, head[-2:])
        head = head[:-2]
    groups.insert(0, head)
    return ",".join(groups) + "," + tail


def _values(rng: random.Random) -> dict[str, str]:
    hour = rng.randint(1, 12)
    minute = rng.choice([0, 15, 30, 45, rng.randint(1, 59)])
    return {
        "amount": _amount(rng),
        "small": str(rng.randint(2, 15)),
        "day": str(rng.randint(1, 28)),
        "month_en": rng.choice(MONTHS_EN),
        "month_ta": rng.choice(MONTHS_TA),
        "time": f"{hour}:{minute:02d}",
        "ampm": rng.choice(["AM", "PM"]),
        "phone": f"0{rng.randint(40, 99)}-{rng.randint(2000, 9999)}-{rng.randint(1000, 9999)}",
        "otp": str(rng.randint(1000, 999_999)),
        "pct": rng.choice([str(rng.randint(1, 20)), f"{rng.randint(1, 15)}.{rng.randint(1, 9)}"]),
    }


def generate(n_per_category: int = 50, seed: int = 7) -> list[dict]:
    """Return prompt rows ``{"id", "category", "text"}`` with unique texts."""
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for category, templates in TEMPLATES.items():
        count = attempts = 0
        while count < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            text = rng.choice(templates).format(**_values(rng))
            if text in seen:
                continue
            seen.add(text)
            count += 1
            rows.append({"id": f"tpl_{category}_{count:04d}", "category": category, "text": text})
    return rows
