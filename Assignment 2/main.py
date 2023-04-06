# Functions and classes for preprocessing the data
import os
import torch
import pickle
import gzip
from sentence_transformers import SentenceTransformer
from preprocessing import preprocess_directory
from retrieval import query_retrieve
import subprocess, sys

print(f'Cuda available: {torch.cuda.is_available()}')
print(f'Current device: {torch.cuda.current_device()}')
print(torch.cuda.get_device_name(0))

# Preprocess the collection
preprocessed_documents = preprocess_directory('AP_collection/coll')

model_names = [
  'all-mpnet-base-v2',
  'msmarco-distilbert-cos-v5',
]

for model_name in model_names:
  # Print the model name
  print(f'---\nModel: {model_name}')
  # Load the model
  model = SentenceTransformer(f'sentence-transformers/{model_name}', device='cuda:0')

  # If the embeddings have already been computed, load them
  if os.path.exists(f"embedding_saves/{model_name}.pickle.gz"):
    # print the message
    print(f'Loading embeddings from file: embedding_saves/{model_name}.pickle.gz')
    # unzip the pickle file 
    with gzip.open(f"embedding_saves/{model_name}.pickle.gz", 'rb') as f_in:
      doc_embeddings = pickle.load(f_in)
  else:
    # print the message
    print(f'Embeddings not found, computing {model_name}')
    # Compute the embeddings
    doc_embeddings = model.encode([doc.doc_text for doc in preprocessed_documents], show_progress_bar=True)
    # print the message
    print(f'Embeddings computed, saving to file: embedding_saves/{model_name}.pickle.gz')
    # store the embeddings in a pickle file
    with open(f"embedding_saves/{model_name}.pickle", 'wb') as f:
      pickle.dump(doc_embeddings, f)
    # gzip the pickle file
    with open(f"embedding_saves/{model_name}.pickle", 'rb') as f_in, gzip.open(f"embedding_saves/{model_name}.pickle.gz", 'wb') as f_out:
      f_out.writelines(f_in)

  # print the message
  print(f'Computing results for {model_name}')
  # Run the query_retrieve function
  query_retrieve(model, preprocessed_documents, doc_embeddings, descriptions=False, runid='runid', filename=f'Results-{model_name}.txt', top_k=1000)

  # Run the trec_eval command
  p = subprocess.Popen(["powershell",f".\\trec_eval -m map qrels1-50ap.txt Results-{model_name}.txt"], stdout=sys.stdout)
  p.communicate()
  p = subprocess.Popen(["powershell",f".\\trec_eval -m P qrels1-50ap.txt Results-{model_name}.txt"], stdout=sys.stdout)
  p.communicate()