#!/usr/bin/env python
# coding: utf-8

# # Environment Setup 
# 

# In[41]:


# !pip install sentence_transformers
# !pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
# !pip3 install black[jupyter]


# In[42]:


# Variables 
# model_name = "all-distilroberta-v1" # will also change the savefile name (csv.gz files)
# model_name = "all-mpnet-base-v2" # will also change the savefile name (csv.gz files)
# model_name = "multi-qa-mpnet-base-dot-v1" # will also change the savefile name (csv.gz files)
# model_name = "msmarco-distilbert-base-tas-b" # will also change the savefile name (csv.gz files)
# model_name = "multi-qa-distilbert-cos-v1" # will also change the savefile name (csv.gz files)
# model_name = "msmarco-distilbert-cos-v5" # will also change the savefile name (csv.gz files)
# model_name = "gtr-t5-xl" # will also change the savefile name (csv.gz files)
model_name = "multi-qa-MiniLM-L6-cos-v1" # will also change the savefile name (csv.gz files)
extracted_documents = []






# # Function Setup
# 

# In[43]:


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


# In[44]:


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


# In[45]:


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


# In[46]:


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


# In[47]:


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


# In[48]:


# print(extracted_documents[0].doc_no + "\n")
# print(extracted_documents[0].doc_text) 


# # DistillBERT Setup (with Cuda)
# 

# In[49]:


import datetime
import pytz

# Set the timezone to Eastern Standard Time (EST)
tz = pytz.timezone('US/Eastern')

def print_time_est():
    # Get the current time in EST
    est_time = datetime.datetime.now(tz)

    # Print the current EST time
    print("Current EST time:", est_time)


# In[50]:


# len(extracted_documents)


# In[51]:


# extracted_documents[79922].vector


# In[52]:


from sklearn.metrics.pairwise import cosine_similarity


# In[53]:


# extracted_documents[0].vector = model.encode(extracted_documents[0].doc_text)


# # Output to File 
# 

# ### NOT compressed (too big for Github)

# In[54]:


# LEGACY CODE 
# import json 

# with open("embedding_saves/distilroberta.json", "w") as outfile:
#     # for doc in extracted_documents:
#     json.dump(extracted_documents[0].to_dict(),outfile)
#     #     json.dump(doc.to_dict(), outfile)


# #### Possible csv write *(not compressed)
# 

# In[55]:


# import csv

# # assuming you have a list of Document objects called documents
# # and assuming you have already populated the vector attribute of each Document object

# # define the headers for your CSV file
# headers = ['doc_no', 'doc_text', 'vector']

# # open the CSV file in 'w' mode and write the headers
# with open("embedding_saves/{model_name}.csv", mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(headers)

#     # loop through each Document object and write its attributes to the CSV file
#     for document in extracted_documents:
#         writer.writerow([document.doc_no, document.doc_text, document.vector.tolist() if document.vector is not None else None])


# #### Possible csv read *(not compressed)
# 

# In[56]:


# # read the CSV file and create new Document objects
# extracted_documents = []
# # --------------------------------------------------------------
# # CHANGE THIS TO extracted_documents later 
# # --------------------------------------------------------------
# with open("embedding_saves/{model_name}.csv", mode='r') as file:
#     reader = csv.reader(file)
#     headers = next(reader) # skip the header row

#     for row in reader:
#         doc_no = row[0]
#         doc_text = row[1]
#         vector = np.array(row[2], dtype=float)
#         document = Document(doc_no, doc_text, vector)
#         extracted_documents.append(document)


# ### Compressed CSV
# 

# In[57]:


import csv
import gzip
import os

# assuming you have a list of Document objects called documents
# and assuming you have already populated the vector attribute of each Document object

# define the headers for your CSV file
headers = ['doc_no', 'doc_text', 'vector']

# open the CSV file in 'w' mode and write the headers
with open(f"embedding_saves/{model_name}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)

    # loop through each Document object and write its attributes to the CSV file
    for document in extracted_documents:
        writer.writerow([document.doc_no, document.doc_text, document.vector.tolist() if document.vector is not None else None])

# gzip the CSV file
with open(f"embedding_saves/{model_name}.csv", 'rb') as f_in, gzip.open(f"embedding_saves/{model_name}.csv.gz", 'wb') as f_out:
    f_out.writelines(f_in)

os.remove(f"embedding_saves/{model_name}.csv")


# In[58]:


# import csv
# import gzip

# # read the gzip file and create new Document objects
# extracted_documents_csv = []
# with gzip.open("embedding_saves/distilroberta.csv.gz", mode='rb') as file:
#     # read the uncompressed content of the gzip file
#     uncompressed_content = file.read()

#     # parse the uncompressed content as a CSV file
#     csv_content = uncompressed_content.decode('utf-8')
#     reader = csv.reader(csv_content.splitlines())

#     # extract the header row
#     headers = next(reader)

#     # loop through each row and create a new Document object
#     for row in reader:
#         doc_no = row[0]
#         doc_text = row[1]
#         vector = np.array(row[2], dtype=float)
#         document = Document(doc_no, doc_text, vector)
#         extracted_documents_csv.append(document)
import csv
import gzip
import ast

# read the gzip file and create new Document objects
extracted_documents = []

if os.path.isfile(f"embedding_saves/{model_name}.csv.gz"):
    with gzip.open(f"embedding_saves/{model_name}.csv.gz", mode='rb') as file:
        # read the uncompressed content of the gzip file
        uncompressed_content = file.read()

        # parse the uncompressed content as a CSV file
        csv_content = uncompressed_content.decode('utf-8')
        reader = csv.reader(csv_content.splitlines())

        # extract the header row
        headers = next(reader)

        # loop through each row and create a new Document object
        for row in reader:
            doc_no = row[0]
            doc_text = row[1]
            vector = ast.literal_eval(row[2]) if row[2] else None
            if vector is not None:
                vector = np.array(vector, dtype=float)
            document = Document(doc_no, doc_text, vector)
            extracted_documents.append(document)
else:
    print("There is no embedding saves of this model")


# In[59]:


# extracted_documents[0].doc_no
# extracted_documents[0].doc_text
# extracted_documents[0].vector.flatten()


# In[60]:


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


# In[61]:


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


# In[62]:


# # doc_embeddings = model.encode([doc.doc_text for doc in preprocessed_documents])
# test = np.array([])

# for doc in extracted_documents:
#     # test.append(doc.vector)
#     np.append(test, doc.vector)




# In[63]:


# print(type(extracted_documents[0].vector))

# type([doc.vector for doc in extracted_documents])


# # Running Code
# 

# In[64]:


extracted_documents = preprocess_directory('AP_collection\coll')
topics = extract_topics("topics1-50.txt")
model = SentenceTransformer(model_name, device="cuda:0")

print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

print_time_est()
for doc in extracted_documents: 
    doc.vector = model.encode([doc.doc_text])
    print(doc.doc_no + " is done.")
print_time_est()

query_retrieve(model, extracted_documents, np.array([doc.vector.flatten() for doc in extracted_documents]), descriptions=False, runid='runid', filename=f'{model_name}_Results.txt', top_k=1000)


# # Results 

# In[65]:


# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/multi-qa-MiniLM-L6-cos-v1_Results.txt


# ### multi-qa-mpnet-base-dot-v1_Results
# - map = 0.3041
# - P_10 = 0.4460

# In[ ]:


# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/multi-qa-mpnet-base-dot-v1_Results.txt

# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1426
# map                     all     0.3041
# gm_map                  all     0.1712
# Rprec                   all     0.3133
# bpref                   all     0.3937
# recip_rank              all     0.6907
# iprec_at_recall_0.00    all     0.7293
# iprec_at_recall_0.10    all     0.6050
# iprec_at_recall_0.20    all     0.5128
# iprec_at_recall_0.30    all     0.4411
# iprec_at_recall_0.40    all     0.3793
# iprec_at_recall_0.50    all     0.2888
# iprec_at_recall_0.60    all     0.2212
# iprec_at_recall_0.70    all     0.1742
# iprec_at_recall_0.80    all     0.1089
# iprec_at_recall_0.90    all     0.0505
# iprec_at_recall_1.00    all     0.0315
# P_5                     all     0.4800
# P_10                    all     0.4460
# P_15                    all     0.4000
# P_20                    all     0.3730
# P_30                    all     0.3173
# P_100                   all     0.1686
# P_200                   all     0.1038
# P_500                   all     0.0509
# P_1000                  all     0.0285
# ```


# In[ ]:


# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/multi-qa-mpnet-base-dot-v1_Results.txt

# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1426
# map                     all     0.3041
# gm_map                  all     0.1712
# Rprec                   all     0.3133
# bpref                   all     0.3937
# recip_rank              all     0.6907
# iprec_at_recall_0.00    all     0.7293
# iprec_at_recall_0.10    all     0.6050
# iprec_at_recall_0.20    all     0.5128
# iprec_at_recall_0.30    all     0.4411
# iprec_at_recall_0.40    all     0.3793
# iprec_at_recall_0.50    all     0.2888
# iprec_at_recall_0.60    all     0.2212
# iprec_at_recall_0.70    all     0.1742
# iprec_at_recall_0.80    all     0.1089
# iprec_at_recall_0.90    all     0.0505
# iprec_at_recall_1.00    all     0.0315
# P_5                     all     0.4800
# P_10                    all     0.4460
# P_15                    all     0.4000
# P_20                    all     0.3730
# P_30                    all     0.3173
# P_100                   all     0.1686
# P_200                   all     0.1038
# P_500                   all     0.0509
# P_1000                  all     0.0285
# ```


# ### multi-qa-distilbert-cos-v1 
# - map = 0.2733
# - P_10 = 0.4000
# 

# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/multi-qa-distilbert-cos-v1_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1442
# map                     all     0.2733
# gm_map                  all     0.1501
# Rprec                   all     0.3047
# bpref                   all     0.3707
# recip_rank              all     0.6945
# iprec_at_recall_0.00    all     0.7310
# iprec_at_recall_0.10    all     0.5705
# iprec_at_recall_0.20    all     0.4815
# iprec_at_recall_0.30    all     0.4024
# iprec_at_recall_0.40    all     0.3367
# iprec_at_recall_0.50    all     0.2626
# iprec_at_recall_0.60    all     0.1827
# iprec_at_recall_0.70    all     0.1248
# iprec_at_recall_0.80    all     0.0858
# iprec_at_recall_0.90    all     0.0518
# iprec_at_recall_1.00    all     0.0269
# P_5                     all     0.4480
# P_10                    all     0.4000
# P_15                    all     0.3720
# P_20                    all     0.3430
# P_30                    all     0.3000
# P_100                   all     0.1670
# P_200                   all     0.1054
# P_500                   all     0.0508
# P_1000                  all     0.0288
# ```

# ### msmarco-distilbert-base-tas-b
# - map = 0.1748
# - P_10 = 0.3260
# 
# 

# In[ ]:


# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/msmarco-distilbert-base-tas-b_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1157
# map                     all     0.1748
# gm_map                  all     0.0553
# Rprec                   all     0.2057
# bpref                   all     0.3011
# recip_rank              all     0.5585
# iprec_at_recall_0.00    all     0.5814
# iprec_at_recall_0.10    all     0.4165
# iprec_at_recall_0.20    all     0.3066
# iprec_at_recall_0.30    all     0.2441
# iprec_at_recall_0.40    all     0.1892
# iprec_at_recall_0.50    all     0.1495
# iprec_at_recall_0.60    all     0.1076
# iprec_at_recall_0.70    all     0.0696
# iprec_at_recall_0.80    all     0.0436
# iprec_at_recall_0.90    all     0.0271
# iprec_at_recall_1.00    all     0.0162
# P_5                     all     0.3960
# P_10                    all     0.3260
# P_15                    all     0.2693
# P_20                    all     0.2480
# P_30                    all     0.2067
# P_100                   all     0.1074
# P_200                   all     0.0694
# P_500                   all     0.0379
# P_1000                  all     0.0231
# ```


# ### all-distilroberta-v1
# - map = 0.1800
# - P_10 = 0.2940

# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/all-distilroberta-v1_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1303
# map                     all     0.1800
# gm_map                  all     0.0727
# Rprec                   all     0.2166
# bpref                   all     0.3167
# recip_rank              all     0.5108
# iprec_at_recall_0.00    all     0.5577
# iprec_at_recall_0.10    all     0.3966
# iprec_at_recall_0.20    all     0.3339
# iprec_at_recall_0.30    all     0.2623
# iprec_at_recall_0.40    all     0.2090
# iprec_at_recall_0.50    all     0.1560
# iprec_at_recall_0.60    all     0.1166
# iprec_at_recall_0.70    all     0.0853
# iprec_at_recall_0.80    all     0.0468
# iprec_at_recall_0.90    all     0.0171
# iprec_at_recall_1.00    all     0.0134
# P_5                     all     0.3320
# P_10                    all     0.2940
# P_15                    all     0.2760
# P_20                    all     0.2510
# P_30                    all     0.2193
# P_100                   all     0.1276
# P_200                   all     0.0816
# P_500                   all     0.0440
# P_1000                  all     0.0261
# ```

# ### all-mpnet-base-v2
# - map = 0.3201
# - P_10 = 0.4700

# ```
# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/all-mpnet-base-v2_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1518
# map                     all     0.3201
# gm_map                  all     0.1956
# Rprec                   all     0.3327
# bpref                   all     0.4019
# recip_rank              all     0.7221
# iprec_at_recall_0.00    all     0.7723
# iprec_at_recall_0.10    all     0.6283
# iprec_at_recall_0.20    all     0.5385
# iprec_at_recall_0.30    all     0.4573
# iprec_at_recall_0.40    all     0.3758
# iprec_at_recall_0.50    all     0.3270
# iprec_at_recall_0.60    all     0.2525
# iprec_at_recall_0.70    all     0.1777
# iprec_at_recall_0.80    all     0.1232
# iprec_at_recall_0.90    all     0.0657
# iprec_at_recall_1.00    all     0.0338
# P_5                     all     0.5160
# P_10                    all     0.4700
# P_15                    all     0.4213
# P_20                    all     0.3940
# P_30                    all     0.3307
# P_100                   all     0.1786
# P_200                   all     0.1114
# P_500                   all     0.0540
# P_1000                  all     0.0304
# ```

# ### msmarco-distilbert-cos-v5
# - map = 0.2058
# - P_10 = 0.3580

# In[66]:


# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/msmarco-distilbert-cos-v5_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1224
# map                     all     0.2058
# gm_map                  all     0.0666
# Rprec                   all     0.2360
# bpref                   all     0.3251
# recip_rank              all     0.6229
# iprec_at_recall_0.00    all     0.6478
# iprec_at_recall_0.10    all     0.4898
# iprec_at_recall_0.20    all     0.3684
# iprec_at_recall_0.30    all     0.2985
# iprec_at_recall_0.40    all     0.2190
# iprec_at_recall_0.50    all     0.1736
# iprec_at_recall_0.60    all     0.1296
# iprec_at_recall_0.70    all     0.0787
# iprec_at_recall_0.80    all     0.0512
# iprec_at_recall_0.90    all     0.0249
# iprec_at_recall_1.00    all     0.0194
# P_5                     all     0.4160
# P_10                    all     0.3580
# P_15                    all     0.3187
# P_20                    all     0.2850
# P_30                    all     0.2413
# P_100                   all     0.1262
# P_200                   all     0.0801
# P_500                   all     0.0412
# P_1000                  all     0.0245


# ### gtr-t5-xl_Results
# - map = 0.3120
# - P_10 = 0.4620

# In[67]:


# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/gtr-t5-xl_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1474
# map                     all     0.3120
# gm_map                  all     0.1880
# Rprec                   all     0.3399
# bpref                   all     0.4019
# recip_rank              all     0.7349
# iprec_at_recall_0.00    all     0.7740
# iprec_at_recall_0.10    all     0.6408
# iprec_at_recall_0.20    all     0.5610
# iprec_at_recall_0.30    all     0.4350
# iprec_at_recall_0.40    all     0.3674
# iprec_at_recall_0.50    all     0.3141
# iprec_at_recall_0.60    all     0.2124
# iprec_at_recall_0.70    all     0.1426
# iprec_at_recall_0.80    all     0.0849
# iprec_at_recall_0.90    all     0.0529
# iprec_at_recall_1.00    all     0.0412
# P_5                     all     0.5240
# P_10                    all     0.4620
# P_15                    all     0.4027
# P_20                    all     0.3610
# P_30                    all     0.3100
# P_100                   all     0.1656
# P_200                   all     0.1023
# P_500                   all     0.0516
# P_1000                  all     0.0295


# ### multi-qa-MiniLM-L6-cos-v1
# - map = 0.2689
# - P_10 = 0.3900

# In[ ]:


# ./trec_eval /home/shang/Info_Retrieval/qrels1-50ap.txt /home/shang/Info_Retrieval/multi-qa-MiniLM-L6-cos-v1_Results.txt
# runid                   all     runid
# num_q                   all     50
# num_ret                 all     50000
# num_rel                 all     2099
# num_rel_ret             all     1411
# map                     all     0.2689
# gm_map                  all     0.1387
# Rprec                   all     0.2891
# bpref                   all     0.3678
# recip_rank              all     0.6554
# iprec_at_recall_0.00    all     0.6913
# iprec_at_recall_0.10    all     0.5359
# iprec_at_recall_0.20    all     0.4541
# iprec_at_recall_0.30    all     0.4034
# iprec_at_recall_0.40    all     0.3483
# iprec_at_recall_0.50    all     0.2720
# iprec_at_recall_0.60    all     0.1875
# iprec_at_recall_0.70    all     0.1425
# iprec_at_recall_0.80    all     0.0844
# iprec_at_recall_0.90    all     0.0443
# iprec_at_recall_1.00    all     0.0296
# P_5                     all     0.4520
# P_10                    all     0.3900
# P_15                    all     0.3533
# P_20                    all     0.3320
# P_30                    all     0.2900
# P_100                   all     0.1620
# P_200                   all     0.1010
# P_500                   all     0.0503
# P_1000                  all     0.0282

