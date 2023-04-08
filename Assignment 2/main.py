# Functions and classes for preprocessing the data
import gzip
import os
import pickle
import subprocess
import sys
import numpy as np

import torch
from preprocessing import preprocess_directory
from retrieval import query_retrieve
from sentence_transformers import SentenceTransformer
import logging

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

print(f'Cuda available: {torch.cuda.is_available()}')
print(f'Current device: {torch.cuda.current_device()}')
print(torch.cuda.get_device_name(0))

# Preprocess the collection
preprocessed_documents = preprocess_directory('AP_collection/coll')

model_names = [
  # 'all-MiniLM-L12-v2',
  # 'all-MiniLM-L6-v2',
  # 'all-distilroberta-v1',
  # 'all-mpnet-base-v2',
  # 'all-roberta-large-v1',
  # 'msmarco-MiniLM-L12-cos-v5',
  # 'msmarco-MiniLM-L6-cos-v5',
  # 'msmarco-distilbert-base-tas-b',
  # 'msmarco-distilbert-cos-v5',
  # 'multi-qa-MiniLM-L6-cos-v1',
  # 'multi-qa-distilbert-cos-v1',
  # 'multi-qa-mpnet-base-dot-v1',
  # 'nli-mpnet-base-v2',
  # 'paraphrase-MiniLM-L3-v2',
  # 'paraphrase-MiniLM-L6-v2',
  # 'paraphrase-albert-small-v2',
  # 'paraphrase-mpnet-base-v2',
  # 'sentence-t5-base',
  # 'stsb-mpnet-base-v2',
  # 'bert-base-nli-mean-tokens',
  # 'distilbert-base-nli-mean-tokens',
  # 'distilbert-base-nli-stsb-mean-tokens',
  # 'distiluse-base-multilingual-cased-v2',
  # 'LaBSE',
  # 'multi-qa-mpnet-base-cos-v1',
  # 'paraphrase-distilroberta-base-v2',
  # 'xlm-r-distilroberta-base-paraphrase-v1',
  # 'gtr-t5-base',
  # 'gtr-t5-large',
  'gtr-t5-xl',
  # 'gtr-t5-xxl',
]

model_names.sort()

# list the models to be used, separated by new lines
print('Models to be used:\n-', '\n- '.join(model_names))


for model_name in model_names:
  # Print the model name
  print(f'---\nModel: {model_name}')
  # Load the model
  model = SentenceTransformer(f'sentence-transformers/{model_name}', device='cuda:0', cache_folder='./.cache')

  # If the embeddings have already been computed, load them
  if os.path.exists(f"embedding_saves/{model_name}.pickle.gz"):
    # print the message
    logging.info(f'Loading embeddings from file: embedding_saves/{model_name}.pickle.gz')
    # unzip the pickle file 
    with gzip.open(f"embedding_saves/{model_name}.pickle.gz", 'rb') as f_in:
      doc_embeddings = pickle.load(f_in)
  else:
    # print the message
    logging.info(f'Embeddings not found, computing {model_name}')
    # Compute the embeddings
    doc_embeddings = []
    for x, doc in enumerate(preprocessed_documents):
      # Clear the cache
      torch.cuda.empty_cache()
      # get the embedding for the document if it exists, otherwise compute it
      if os.path.exists(f'embedding_saves/{model_name}/{doc.doc_no.strip()}.pickle'):
        with open(f'embedding_saves/{model_name}/{doc.doc_no.strip()}.pickle', 'rb') as f:
          doc_embed = pickle.load(f)
      else:
        # Calculate embedding for each document
        logging.info(f'Embedding {doc.doc_no.strip()} {x}/{len(preprocessed_documents)}...')
        doc_embed = model.encode(doc.doc_text, show_progress_bar=False)
        # write the document embedding to a file
        os.makedirs(f'embedding_saves/{model_name}', exist_ok=True)
        with open(f'embedding_saves/{model_name}/{doc.doc_no.strip()}.pickle', 'wb') as f:
          pickle.dump(doc_embed, f)
      doc_embeddings.append(doc_embed)

    # Save the embeddings
    doc_embeddings = np.array(doc_embeddings)

    # print the message
    logging.info(f'Embeddings computed, saving to file: embedding_saves/{model_name}.pickle.gz')
    # store the embeddings in a pickle file
    with open(f"embedding_saves/{model_name}.pickle", 'wb') as f:
      pickle.dump(np.array(doc_embeddings), f)
    # gzip the pickle file
    with open(f"embedding_saves/{model_name}.pickle", 'rb') as f_in, gzip.open(f"embedding_saves/{model_name}.pickle.gz", 'wb') as f_out:
      f_out.writelines(f_in)

  # print the message
  logging.info(f'Computing results for {model_name}')
  # Run the query_retrieve function
  query_retrieve(model, preprocessed_documents, doc_embeddings, descriptions=False, runid='runid', filename=f'Results-{model_name}.txt', top_k=1000)

  # Run the trec_eval command
  p = subprocess.Popen(["powershell",f".\\trec_eval -m map qrels1-50ap.txt Results-{model_name}.txt"], stdout=sys.stdout)
  p.communicate()
  p = subprocess.Popen(["powershell",f".\\trec_eval -m P qrels1-50ap.txt Results-{model_name}.txt"], stdout=sys.stdout)
  p.communicate()

print('---\nDone')