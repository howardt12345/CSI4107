# Functions and classes for preprocessing the data
import re
import os
class Document:
  def __init__(self, doc_no, doc_text):
    self.doc_no = doc_no
    self.doc_text = doc_text

  def __str__(self):
    return 'Document Number: ' + self.doc_no + '\nDocument Text: ' + self.doc_text

  def to_dict(self):
    return {'docno': self.doc_no, 'doctext': self.doc_text}

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