import re
import string

from textacy.preprocess import preprocess_text

EMOJI_PATTERN = re.compile(
    u"(\ud83d[\ude00-\ude4f])|"  # emoticons
    u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
    u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
    u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
    u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
    "+", flags=re.UNICODE)
MENTIONS_PATTERN = re.compile(u"@[a-z]+")
HASHTAGS_PATTERN = re.compile(u"#[a-z]+")


def preprocess(text):
    new_text = preprocess_text(text, fix_unicode=True, lowercase=True, no_urls=True,
                               no_emails=True, no_phone_numbers=True, no_numbers=True,
                               no_currency_symbols=True, no_contractions=True,
                               no_accents=True)
    no_mentions_text = re.sub(MENTIONS_PATTERN, u"", new_text)
    no_hashtags_text = re.sub(HASHTAGS_PATTERN, u"", no_mentions_text)
    no_emojis_text = re.sub(EMOJI_PATTERN, u"", no_hashtags_text)
    separated_punctuation_text = no_emojis_text.translate(
        str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
    return separated_punctuation_text


def split_list_pairs(l):
    return [[l[i], l[i + 1]] for i in range(len(l) - 1)]
