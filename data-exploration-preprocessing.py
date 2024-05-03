
import pandas as pd
import numpy as np
import sys

import re


import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')  # Download NLTK data

from transformers import AutoTokenizer

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

print(train.shape, test.shape)

word_counts = []
for text in train['full_text']:
    tokens = word_tokenize(text)
    word_counts.extend(tokens)

# Generate a histogram of word counts
# plt.figure(figsize=(10, 5))
# plt.hist([len(word_tokenize(text)) for text in train['full_text']], bins=20, color='skyblue')
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.title('Histogram of Word Counts')
# plt.grid(True)
# plt.show()
#
# Generate a bar chart showing the most commonly used words
# fdist = FreqDist(word_counts)
# top_words = fdist.most_common(25)  # Change 25 to the number of top words you want to display
# words, frequencies = zip(*top_words)
# plt.figure(figsize=(10, 5))
# plt.bar(words, frequencies, color='skyblue')
# plt.xlabel('Words')
# plt.ylabel('Frequency')
# plt.title('Top 10 Most Commonly Used Words')
# plt.xticks(rotation=45)
# plt.show()

# Clean the text by removing unnecessary information and lower
def filter_websites(text):
    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    return re.sub(pattern, '', text)

def filter_phone_numbers(text):
    pattern = r'(?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})|(\d{2}[ ]\d{2}[ ]\d{3}[ ]\d{3})'
    return re.sub(pattern, '', text)

def filter_emails(text):
    pattern = r'(?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,})'
    return re.sub(pattern, '', text)

def remove_newlines(text):
    return text.replace("\n", "")

def clean_text(text):
    text = text.lower()
    #text = filter_websites(text)
    #text = filter_phone_numbers(text)
    #text = filter_emails(text)
    text = remove_newlines(text)
    return text

train['cleaned_full_text'] = train.full_text.apply(clean_text)
test['cleaned_full_text'] = test.full_text.apply(clean_text)

train.head()

train.to_csv('./data/train_cleaned.csv', index=False)
test.to_csv('./data/test_cleaned.csv', index=False)

