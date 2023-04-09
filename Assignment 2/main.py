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
preprocessed_documents.sort(key=lambda x: x.doc_no)

sentence_transformers = [
    'all-MiniLM-L12-v2',
    'all-MiniLM-L6-v2',
    'all-distilroberta-v1',
    'all-mpnet-base-v2',
    'all-roberta-large-v1',
    'msmarco-MiniLM-L12-cos-v5',
    'msmarco-MiniLM-L6-cos-v5',
    'msmarco-distilbert-base-tas-b',
    'msmarco-distilbert-cos-v5',
    'multi-qa-MiniLM-L6-cos-v1',
    'multi-qa-distilbert-cos-v1',
    'multi-qa-mpnet-base-dot-v1',
    'nli-mpnet-base-v2',
    'paraphrase-MiniLM-L3-v2',
    'paraphrase-MiniLM-L6-v2',
    'paraphrase-albert-small-v2',
    'paraphrase-mpnet-base-v2',
    'sentence-t5-base',
    'stsb-mpnet-base-v2',
    'bert-base-nli-mean-tokens',
    'distilbert-base-nli-mean-tokens',
    'distilbert-base-nli-stsb-mean-tokens',
    'distiluse-base-multilingual-cased-v2',
    'LaBSE',
    'multi-qa-mpnet-base-cos-v1',
    'paraphrase-distilroberta-base-v2',
    'xlm-r-distilroberta-base-paraphrase-v1',
    'gtr-t5-base',
    'gtr-t5-large',
    'gtr-t5-xl',
    'gtr-t5-xxl',
]
# redefine the above list, but with a device specified
sentence_transformers = [
    ('all-MiniLM-L12-v2', 'cuda:0'),
    ('all-MiniLM-L6-v2', 'cuda:0'),
    ('all-distilroberta-v1', 'cuda:0'),
    ('all-mpnet-base-v2', 'cuda:0'),
    ('all-roberta-large-v1', 'cuda:0'),
    ('msmarco-MiniLM-L12-cos-v5', 'cuda:0'),
    ('msmarco-MiniLM-L6-cos-v5', 'cuda:0'),
    ('msmarco-distilbert-base-tas-b', 'cuda:0'),
    ('msmarco-distilbert-cos-v5', 'cuda:0'),
    ('multi-qa-MiniLM-L6-cos-v1', 'cuda:0'),
    ('multi-qa-distilbert-cos-v1', 'cuda:0'),
    ('multi-qa-mpnet-base-dot-v1', 'cuda:0'),
    ('nli-mpnet-base-v2', 'cuda:0'),
    ('paraphrase-MiniLM-L3-v2', 'cuda:0'),
    ('paraphrase-MiniLM-L6-v2', 'cuda:0'),
    ('paraphrase-albert-small-v2', 'cuda:0'),
    ('paraphrase-mpnet-base-v2', 'cuda:0'),
    ('sentence-t5-base', 'cuda:0'),
    ('stsb-mpnet-base-v2', 'cuda:0'),
    ('bert-base-nli-mean-tokens', 'cuda:0'),
    ('distilbert-base-nli-mean-tokens', 'cuda:0'),
    ('distilbert-base-nli-stsb-mean-tokens', 'cuda:0'),
    ('distiluse-base-multilingual-cased-v2', 'cuda:0'),
    ('LaBSE', 'cuda:0'),
    ('multi-qa-mpnet-base-cos-v1', 'cuda:0'),
    ('paraphrase-distilroberta-base-v2', 'cuda:0'),
    ('xlm-r-distilroberta-base-paraphrase-v1', 'cuda:0'),
    ('gtr-t5-base', 'cuda:0'),
    ('gtr-t5-large', 'cuda:0'),
    ('gtr-t5-xl', 'cuda:0'),
    ('gtr-t5-xxl', 'cpu'),
]
muennighoff = [
    'SGPT-125M-weightedmean-nli-bitfit',
    # 'SGPT-5.8B-weightedmean-msmarco-specb-bitfit',
    # 'SGPT-1.3B-weightedmean-msmarco-specb-bitfit',
    # 'SGPT-125M-weightedmean-msmarco-specb-bitfit',
    't5-small-finetuned-xsum',
]


sentence_transformers.sort(key=lambda x: x[0])
muennighoff.sort()

models = {
    'sentence-transformers': sentence_transformers,
    # 'Muennighoff': muennighoff,
}

# list the models to be used, separated by new lines
print('Models to be used:\n-', '\n- '.join([x[0] for x in sentence_transformers]))

embed_each_document = False
descriptions = False

for lib, value in models.items():
  for v in value:
    model_name, device = v
    print(f'---\nModel: {model_name}{" (descriptions)" if descriptions else ""}')
    # Load the model
    torch.cuda.empty_cache()
    logging.info(f'Using device: {device}')
    model = SentenceTransformer(f'{lib}/{model_name}', device=device, cache_folder='./.cache')

    # If the embeddings have already been computed, load them
    if os.path.exists(f"embedding_saves/{lib}/{model_name}.pickle.gz"):
      logging.info(
          f'Loading embeddings from file: embedding_saves/{lib}/{model_name}.pickle.gz')
      # unzip the pickle file
      with gzip.open(f"embedding_saves/{lib}/{model_name}.pickle.gz", 'rb') as f_in:
        doc_embeddings = pickle.load(f_in)
    else:
      if embed_each_document:
        logging.info(f'Embeddings not found, computing {model_name}')
        # Compute the embeddings
        os.makedirs(f'embedding_saves/{lib}/{model_name}', exist_ok=True)
        for x, doc in enumerate(preprocessed_documents):
          # Clear the cache
          torch.cuda.empty_cache()
          # get the embedding for the document if it exists, otherwise compute it
          if os.path.exists(f'embedding_saves/{lib}/{model_name}/{doc.doc_no.strip()}.pickle'):
            continue
          else:
            # Calculate embedding for each document
            logging.info(
                f'Embedding {doc.doc_no.strip()} {x}/{len(preprocessed_documents)}...')
            doc_embed = model.encode(doc.doc_text, show_progress_bar=False)
            # write the document embedding to a file
            with open(f'embedding_saves/{lib}/{model_name}/{doc.doc_no.strip()}.pickle', 'wb') as f:
              pickle.dump(doc_embed, f)

        # Read all the embeddings from the files in the directory
        doc_embeddings = []
        for x, doc in enumerate(preprocessed_documents):
          if x % 1000 == 0:
            logging.info(f'{x}/{len(preprocessed_documents)}')
          filename = doc.doc_no.strip()
          if os.path.exists(f'embedding_saves/{lib}/{model_name}/{filename}.pickle'):
            # print(f'Loading embedding for {model_name}/{filename} {x}/{len(preprocessed_documents)}')
            with open(f'embedding_saves/{lib}/{model_name}/{filename}.pickle', 'rb') as f:
              doc_embeddings.append(pickle.load(f))
          else:
            logging.info(
                f'Embedding for {model_name}/{filename} doesn\'t exist {x}/{len(preprocessed_documents)}')

        # Save the embeddings
        doc_embeddings = np.array(doc_embeddings)
      else:
        logging.info(f'Embeddings not found, computing {model_name}')
        # Compute the embeddings
        doc_embeddings = model.encode(
            [doc.doc_text for doc in preprocessed_documents], show_progress_bar=True)

      logging.info(
          f'Embeddings computed, saving to file: embedding_saves/{lib}/{model_name}.pickle.gz')
      # store the embeddings in a pickle file
      with open(f"embedding_saves/{lib}/{model_name}.pickle", 'wb') as f:
        pickle.dump(np.array(doc_embeddings), f)
      # gzip the pickle file
      with open(f"embedding_saves/{lib}/{model_name}.pickle", 'rb') as f_in, gzip.open(f"embedding_saves/{lib}/{model_name}.pickle.gz", 'wb') as f_out:
        f_out.writelines(f_in)

    logging.info(f'Computing results for {model_name}')
    # Run the query_retrieve function
    query_retrieve(f'{lib}/{model_name}', model, preprocessed_documents, doc_embeddings, descriptions,
                   runid='runid', filename=f'results/Results-{model_name}{"-descriptions" if descriptions else ""}.txt', top_k=1000)

    # Run the trec_eval command
    p = subprocess.Popen(
        ["powershell", f".\\trec_eval -m map qrels1-50ap.txt results/Results-{model_name}.txt"], stdout=sys.stdout)
    p.communicate()
    p = subprocess.Popen(
        ["powershell", f".\\trec_eval -m P qrels1-50ap.txt results/Results-{model_name}.txt"], stdout=sys.stdout)
    p.communicate()

print('---\nDone')
