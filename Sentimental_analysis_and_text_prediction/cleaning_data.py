import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

data = pd.read_csv("sentimental_analysis_dataset.csv")

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)
    
    
def preprocess_text(sen):
    lemmatizer = WordNetLemmatizer()
    sentence = sen.lower()
    sentence = remove_tags(sentence)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = ' '.join([lemmatizer.lemmatize(word) for word in sentence.split() if word not in stopwords.words('english')])
    return sentence
    
    
X = [preprocess_text(sen) for sen in data['line']]
Y = data['sentiment']

df = pd.DataFrame(list(zip(X, Y)), columns=['Text', 'Target'])

df.to_csv('processed_data.csv', index=False)



