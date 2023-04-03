
from retrieval import query_retrieve
import pyterrier as pt
import pandas as pd
import os
import nltk
nltk.download('punkt')
import gensim
from gensim.models import Word2Vec
from preprocessing import preprocess_directory

# Training the model
df = pd.read_csv('redditWorldNews.csv')
newsTitles = df["title"].values
newsVec = [nltk.word_tokenize(title) for title in newsTitles]

# Using Word2Vec and Skip Gram
w2v = gensim.models.Word2Vec(newsVec, vector_size=100, window=5, min_count = 5, workers=8, sg=1)

# Print results
# Still can't work because missing keys?
query_retrieve(w2v, runid='w2v-titles', filename='Results-w2v-titles.txt')
query_retrieve(w2v, runid='w2v-titles-descriptions', descriptions=True, filename='Results-w2v-titles-descriptions.txt')

# def testing():
#   # Query the model and write the results
#   query_retrieve(tf_idf, runid='tf_idf-titles', filename='Results-tf_idf-titles.txt')
#   query_retrieve(tf_idf, runid='tf_idf-titles-descriptions', descriptions=True, filename='Results-tf_idf-titles-descriptions.txt')

#   # Using BM25
#   print("Using BM25")
#   bm25 = pt.BatchRetrieve(indexref, wmodel='BM25', num_results=1000)

#   # Using Word2Vec and Skip Gram
#   w2v = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1)

#   # Query the model and write the results
#   query_retrieve(bm25, runid='bm25-titles', filename='Results-bm25-titles.txt')
#   query_retrieve(bm25, runid='bm25-titles-descriptions', descriptions=True, filename='Results-bm25-titles-descriptions.txt')

# Run the testing function
# testing()