# Functions and classes for preprocessing the data
from itertools import chain
import string
import re
import os
# import nltk
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
# nltk.download('punkt')

class Document:
  def __init__(self, doc_no, doc_text):
    self.doc_no = doc_no
    self.doc_text = doc_text

  def __str__(self):
    return 'Document Number: ' + self.doc_no + '\nDocument Text: ' + self.doc_text

  def to_dict(self):
    return {'docno': self.doc_no, 'doctext': self.doc_text}

# Get the stop words
def get_stop_words():
  stopwords = set()
  # Open the stop words and add them to the set
  with open('StopWords.txt', 'r') as file:
    for line in file:
      stopwords.add(line.strip())
  return stopwords


# load the stopwords
stop_words = get_stop_words()

# function to perform preprocessing on the text
def preprocess(file):
  with open(file, "r") as f:
    content = f.read()
  documents = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
  preprocessed_documents = []
  for document in documents:
    # Get the document number and text
    raw_no = re.search(r'<DOCNO>(.*?)</DOCNO>', document, re.DOTALL)
    doc_no = raw_no.group(1) if raw_no else ''
    raw_text = re.search(r'<TEXT>(.*?)</TEXT>', document, re.DOTALL)
    doc_text = raw_text.string if raw_text else ''

    # create a document object
    doc = Document(doc_no, doc_text)
    preprocessed_documents.append(doc)
  return preprocessed_documents

# # function to preprocess a single text string
# def preprocess_text(text: str, stem=True, stopwords=True):
#     # lowercase the text
#   text = text.lower()

#   # tokenize the text
#   tokens = word_tokenize(text)
#   # remove stopwords
#   if stopwords:
#     tokens = [token for token in tokens if token not in stop_words]
#   # stem the tokens
#   if stem:
#     # apply the porter stemmer
#     stemmer = PorterStemmer()
#     tokens = [stemmer.stem(token) for token in tokens]
#   # remove punctuation
#   table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
#   stripped = [w.translate(table) for w in tokens]
#   stripped = list(chain(*[w.split() for w in stripped]))

#   # remove empty tokens, stopwords (if applicable) and non-alphabetic tokens
#   stripped = [
#       token for token in stripped if token and (token not in stop_words if stopwords else True) and token.isalpha()]
#   return stripped

# main function to preprocess a directory of text files
def preprocess_directory(directory, num_files=-1):
  preprocessed_documents = []
  ctr = 0
  for filename in os.listdir(directory):
    # print('Preprocessing file: ', filename)
    file = os.path.join(directory, filename)
    preprocessed_documents.extend(preprocess(file))
    ctr += 1
    if ctr == num_files and num_files != -1:
      break
  return preprocessed_documents