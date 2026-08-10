import re


def clean_text(text):
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)

    # Remove Reuters source markers
    text = re.sub(r"\(\s*reuters\s*\)\s*-?", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\breuters\b", " ", text, flags=re.IGNORECASE)

    # Remove scraping / publishing artifacts
    artifacts = [
        r"\bfeatured image\b",
        r"\bgetty images\b",
        r"\bpic twitter\b",
        r"\btwitter com\b",
        r"\bscreen capture\b",
    ]

    for pattern in artifacts:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Normalize case
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
