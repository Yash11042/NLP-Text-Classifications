# clean_text.py

import re
import string
import contractions

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").strip()
    text = str(contractions.fix(text))            # expand contractions
    text = re.sub(r'<[^>]+>', ' ', text)          # remove HTML
    text = re.sub(r'http\S+|www\.\S+', ' ', text) # remove URLs
    text = re.sub(r'\d+', ' ', text)              # remove numbers
    text = re.sub(r'[^\x00-\x7f]', ' ', text)     # remove non-ascii
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra spaces

    return text.lower()
