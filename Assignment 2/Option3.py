import pandas as pd
import os
# import nltk
import re
from transformers import DistilBertModel
from sentence_transformers import SentenceTransformer, util
import torch
# import torchvision
import numpy as np
import scipy
import datetime
import pytz


# Variables 
# model_name = "all-distilroberta-v1" # will also change the savefile name (csv.gz files)
# model_name = "all-mpnet-base-v2" # will also change the savefile name (csv.gz files)
# model_name = "multi-qa-mpnet-base-dot-v1" # will also change the savefile name (csv.gz files)
# model_name = "msmarco-distilbert-base-tas-b" # will also change the savefile name (csv.gz files)
# model_name = "multi-qa-distilbert-cos-v1" # will also change the savefile name (csv.gz files)
model_name = "msmarco-distilbert-cos-v5" # will also change the savefile name (csv.gz files)

# Modified version of Document where tokens are not included 
class Document:
  def __init__(self, doc_no, doc_text, vector):
    self.doc_no = doc_no
    self.doc_text = doc_text
    # self.tokens = tokens
    self.vector = vector

  def __str__(self):
    # return 'Document Number: ' + self.doc_no + '\nDocument Text: ' + self.doc_text + '\nTokens: ' + str(self.tokens) + '\n'
    return 'Document Number: ' + self.doc_no + '\nDocument Text: ' + self.doc_text  + '\n Vectors: ' + str(self.vector) + '\n'


  def to_dict(self):
    # return {'docno': self.doc_no, 'doctext': self.doc_text, 'tokens': self.tokens, 'text': ' '.join(self.tokens)}
    return {'docno': self.doc_no, 'doctext': self.doc_text, 'vector': self.vector.tolist()}

# Modified version of preprocess where tokenizing is removed 
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
        doc = Document(doc_no, doc_text, None)
        preprocessed_documents.append(doc)
    return preprocessed_documents

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


def extract_topics(file, descriptions=False):
  with open(file, "r") as f:
    topic_content = f.read()
  all_topics = []
  topics = re.findall(r'<top>(.*?)</top>', topic_content, re.DOTALL)
  for topic in topics:
    raw_title = re.search(r'<title>(.*?)\n\n', topic, re.DOTALL)
    title = raw_title.group(1) if raw_title else ''
    if descriptions:
      raw_desc = re.search(r'<desc>(.*?)\n\n', topic, re.DOTALL)
      desc = raw_desc.group(1) if raw_desc else ''
      all_topics.append({'title': title, 'description': desc})
    else:
      all_topics.append({'title': title})
  return all_topics

def print_time_est():
    # Get the current time in EST
    est_time = datetime.datetime.now(tz)

    # Print the current EST time
    print("Current EST time:", est_time)



def search(query, model, preprocessed_documents, doc_embeddings, top_k=20):
  query_embeddings = model.encode([query])
  # compute distances
  distances = scipy.spatial.distance.cdist(query_embeddings, doc_embeddings, "cosine")[0]
  # get the top k results
  results = zip(range(len(distances)), distances)
  results = sorted(results, key=lambda x: x[1])
  # Create a list of tuples with the document number and the distance
  results = [(preprocessed_documents[idx].doc_no, distance) for idx, distance in results[0:top_k]]
  return results

# Go through all the documents and search for the top 1000 results
def query_retrieve(model, preprocessed_documents, doc_embeddings, descriptions=False, runid='runid', filename='Results.txt', top_k=1000):
  # Extract the topics
  topics = extract_topics('topics1-50.txt', descriptions)
  
  file_out = open(filename, 'w')

  for i, topic in enumerate(topics):
    # Search for the documents
    results = search(topic['title'], model, preprocessed_documents, doc_embeddings, top_k)
    for j, (doc_id, distance) in enumerate(results):
      file_out.write(f'{i+1} Q0 {doc_id.strip()} {j+1} {1-distance} {runid}\n')
  file_out.close()
  print('Written results to file ', filename)

def model_encoding():
    for doc in extracted_documents: 
        doc.vector = model.encode([doc.doc_text])
        print(doc.doc_no + " is done.")

extracted_documents = []
extracted_documents = preprocess_directory('AP_collection\coll')
topics = extract_topics("topics1-50.txt")
model = SentenceTransformer(model_name, device="cuda:0")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))



# Set the timezone to Eastern Standard Time (EST)
tz = pytz.timezone('US/Eastern')

print_time_est()
model_encoding()
print_time_est()
