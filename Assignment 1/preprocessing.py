# Functions and classes for preprocessing the data
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
import os
import nltk
nltk.download('punkt')
from itertools import chain


class Document:
  def __init__(self, doc_no, doc_text, tokens):
    self.doc_no = doc_no
    self.doc_text = doc_text
    self.tokens = tokens

  def __str__(self):
    return 'Document Number: ' + self.doc_no + '\nDocument Text: ' + self.doc_text + '\nTokens: ' + str(self.tokens) + '\n'

  def to_dict(self):
    return {'docno': self.doc_no, 'doctext': self.doc_text, 'tokens': self.tokens, 'text': ' '.join(self.tokens)}

# Get the stop words
def get_stop_words():
  stopwords = set()
  # Open the stop words and add them to the set
  with open('StopWords.txt', 'r') as file:
    for line in file:
      stopwords.add(line.strip())
  return stopwords

# initialize the stemmer
stemmer = PorterStemmer()

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
    doc_text = raw_text.group(1) if raw_text else ''

    # lowercase the text
    doc_text = doc_text.lower()

    # tokenize the text
    tokens = word_tokenize(doc_text)
    # lowercase all tokens
    tokens = [token.lower() for token in tokens]
    # remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # apply the porter stemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in stemmed_tokens]

    # remove empty tokens, stopwords and non-alphabetic tokens
    stripped = [
        token for token in stripped if token and token not in stop_words and token.isalpha()]

    # create a document object
    doc = Document(doc_no, doc_text, stripped)
    preprocessed_documents.append(doc)
  return preprocessed_documents


def preprocess_text(doc_text):
      # lowercase the text
    doc_text = doc_text.lower()

    # tokenize the text
    tokens = word_tokenize(doc_text)
    # lowercase all tokens
    tokens = [token.lower() for token in tokens]
    # remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    # apply the porter stemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # remove punctuation
    table = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    stripped = [w.translate(table) for w in stemmed_tokens]
    stripped = list(chain(*[w.split() for w in stripped]))
    
    # remove empty tokens, stopwords and non-alphabetic tokens
    stripped = [
        token for token in stripped if token and token not in stop_words and token.isalpha()]
    return ' '.join(stripped)
# main function to preprocess a directory of text files
def preprocess_directory(directory, num_files=-1):
  preprocessed_documents = []
  ctr = 0
  for filename in os.listdir(directory):
    print('Preprocessing file: ', filename)
    file = os.path.join(directory, filename)
    preprocessed_documents.extend(preprocess(file))
    ctr += 1
    if ctr == num_files and num_files != -1:
      break
  return preprocessed_documents



def extract_topics(file):
  with open(file, "r") as f:
    topic_content = f.read()
  all_topics = []
  topics = re.findall(r'<top>(.*?)</top>', topic_content, re.DOTALL)
  for topic in topics:
    raw_title = re.search(r'<title>(.*?)\n\n', topic, re.DOTALL)
    title = raw_title.group(1) if raw_title else ''
    all_topics.append(title)
  return all_topics