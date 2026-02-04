
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.
    """
    print("Preprocessing data...")
    
    # Ensure NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def clean_text(text):
        text = text.lower()
        text = re.sub('<br />', '', text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'\@w+|\#','', text)
        text = re.sub(r'[^\w\s]','', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    def stem_text(text):
        return " ".join([stemmer.stem(word) for word in text.split()])

    # Apply cleaning
    df['review_cleaned'] = df['review'].apply(clean_text)
    
    # Apply stemming
    df['review_final'] = df['review_cleaned'].apply(stem_text)
    
    # Drop duplicates
    df = df.drop_duplicates('review_final')
    
    # Encode Sentiment using map which is cleaner than replace inplace
    df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    print(f"Data shape after cleaning: {df.shape}")
    
    return df

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
    assert 'review_final' in output.columns, 'Missing processed text column'
    assert 'sentiment_encoded' in output.columns, 'Missing target column'
