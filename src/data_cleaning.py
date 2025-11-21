import pandas as pd
import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_dataframe(df, text_column="review"):
    df_clean = df.copy()
    df_clean[text_column] = df_clean[text_column].apply(clean_text)
    return df_clean