from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pyterrier as pt
import string
import re
import os
import nltk
nltk.download('punkt')
from preprocessing import preprocess_directory, preprocess


if not pt.started():
  pt.init()

# Preprocess the collection
preprocessed_documents = preprocess_directory('AP_collection/coll')

# show the first document
print(preprocessed_documents[0])