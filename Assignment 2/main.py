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

class SentenceTransformerSpecb(SentenceTransformer):
    # Requires:
    # pip install git+https://github.com/Muennighoff/sentence-transformers.git@sgpt_poolings_specb
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokens = ["[SOS]", "{SOS}"]
        self._first_module().tokenizer.add_tokens(tokens, special_tokens=True)
        self._first_module().auto_model.resize_token_embeddings(len(self._first_module().tokenizer))
        # Will be replaced with the rep tokens in the model ones
        # The problem is we don't know if a text is query or document when tokenizing in the Transformer.py module, 
        # so we use the SOS tokens as an identifier if we have a query or document at hand & then replace them
        # If we would directly use the brackets here, they may become part of another token
        self._first_module().bos_spec_token_q = self._first_module().tokenizer.encode("[SOS]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d = self._first_module().tokenizer.encode("{SOS}", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_q_rep = self._first_module().tokenizer.encode("[", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_q = self._first_module().tokenizer.encode("]", add_special_tokens=False)[0]
        self._first_module().bos_spec_token_d_rep = self._first_module().tokenizer.encode("{", add_special_tokens=False)[0]
        self._first_module().eos_spec_token_d = self._first_module().tokenizer.encode("}", add_special_tokens=False)[0]
        self._first_module().replace_bos = True

    def encode(self, sentences, **kwargs):
        is_query = kwargs.pop("is_query", True)
        if is_query:
            sentences = "[SOS]" + sentences if isinstance(sentences, str) else ["[SOS]" + sent for sent in sentences]
        else:
            sentences = "{SOS}" + sentences if isinstance(sentences, str) else ["{SOS}" + sent for sent in sentences]    
        return super().encode(sentences, **kwargs)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

print(f'Cuda available: {torch.cuda.is_available()}')
print(f'Current device: {torch.cuda.current_device()}')
print(torch.cuda.get_device_name(0))

# Preprocess the collection
logging.info('Preprocessing the collection...')
preprocessed_documents = preprocess_directory('AP_collection/coll')
preprocessed_documents.sort(key=lambda x: x.doc_no)

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
    # ('SGPT-5.8B-weightedmean-msmarco-specb-bitfit', 'cuda:0'),
    # ('SGPT-1.3B-weightedmean-msmarco-specb-bitfit',  'cuda:0'),
    ('SGPT-125M-lasttoken-msmarco-specb', 'cuda:0'),
    # ('SGPT-125M-learntmean-nli', 'cuda:0'),
    # ('SGPT-125M-mean-nli', 'cuda:0'),
    # ('SGPT-125M-mean-nli-bitfit', 'cuda:0'),
    # ('SGPT-125M-mean-nli-linear5', 'cuda:0'),
    # ('SGPT-125M-mean-nli-linearthenpool5', 'cuda:0'),
    # ('SGPT-125M-scratchmean-nli', 'cuda:0'),
    # ('SGPT-125M-weightedmean-msmarco', 'cuda:0'),
    # ('SGPT-125M-weightedmean-msmarco-asym', 'cuda:0'),
    ('SGPT-125M-weightedmean-msmarco-specb', 'cuda:0'),
    ('SGPT-125M-weightedmean-msmarco-specb-bitfit', 'cuda:0'),
    ('SGPT-125M-weightedmean-msmarco-specb-bitfitwte', 'cuda:0'),
    # ('SGPT-125M-weightedmean-nli', 'cuda:0'),
    # ('SGPT-125M-weightedmean-nli-bitfit', 'cuda:0'),
    # ('SGPT-125M-weightedmean-nli-bitfit-linearthenpool1-noact', 'cuda:0'),
]


sentence_transformers.sort(key=lambda x: x[0])
muennighoff.sort()

models = {
    'sentence-transformers': sentence_transformers,
    'Muennighoff': muennighoff,
}

# list the models to be used, separated by new lines
print('Models to be used:')
for lib, value in models.items():
  for v in value:
    model_name, device = v
    print(f'- {lib}/{model_name} ({device})')

embed_each_document = False
descriptions = True


for lib, value in models.items():
  for v in value:
    model_name, device = v
    print(f'---\nModel: {model_name}{" (descriptions)" if descriptions else ""}')
    # Load the model
    torch.cuda.empty_cache()
    logging.info(f'Using device: {device}')
    if lib == 'Muennighoff' and 'specb' in model_name:
      model = SentenceTransformerSpecb(
        f'{lib}/{model_name}', device=device, cache_folder='./.cache')
    else:
      model = SentenceTransformer(
        f'{lib}/{model_name}', device=device, cache_folder='./.cache')

    # If the embeddings have already been computed, load them
    model_dir = f'embedding_saves/{lib}/{model_name}'
    model_pickle = f'{model_dir}.pickle'
    model_pickle_gz = f'{model_dir}.pickle.gz'

    if os.path.exists(model_pickle_gz):
      logging.info(
          f'Loading embeddings from file: {model_pickle_gz} (unzipped: {model_pickle})')
      # unzip the pickle file
      with gzip.open(model_pickle_gz, 'rb') as f_in:
        doc_embeddings = pickle.load(f_in)
    else:
      if embed_each_document:
        logging.info(f'Embeddings not found, computing {model_name}')
        # Compute the embeddings
        os.makedirs(model_dir, exist_ok=True)
        for x, doc in enumerate(preprocessed_documents):
          # Clear the cache
          torch.cuda.empty_cache()
          # get the embedding for the document if it exists, otherwise compute it
          doc_embed_filename = f'{model_dir}/{doc.doc_no.strip()}.pickle'
          if os.path.exists(doc_embed_filename):
            continue
          else:
            # Calculate embedding for each document
            logging.info(
                f'Embedding {doc.doc_no.strip()} {x}/{len(preprocessed_documents)}...')
            doc_embed = model.encode(doc.doc_text, show_progress_bar=False)
            # write the document embedding to a file
            with open(doc_embed_filename, 'wb') as f:
              pickle.dump(doc_embed, f)

        # Read all the embeddings from the files in the directory
        doc_embeddings = []
        for x, doc in enumerate(preprocessed_documents):
          if x % 1000 == 0:
            logging.info(f'{x}/{len(preprocessed_documents)}')
          filename = doc.doc_no.strip()
          doc_embed_filename = f'{model_dir}/{doc.doc_no.strip()}.pickle'
          if os.path.exists(doc_embed_filename):
            # print(f'Loading embedding for {model_name}/{filename} {x}/{len(preprocessed_documents)}')
            with open(doc_embed_filename, 'rb') as f:
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
          f'Embeddings computed, saving to file: {model_pickle_gz}')
      # store the embeddings in a pickle file
      with open(model_pickle, 'wb') as f:
        pickle.dump(np.array(doc_embeddings), f)
      # gzip the pickle file
      with open(model_pickle, 'rb') as f_in, gzip.open(model_pickle_gz, 'wb') as f_out:
        f_out.writelines(f_in)

    logging.info(f'Computing results for {model_name}')
    filename = f'results/Results-{model_name}{"-descriptions" if descriptions else ""}.txt'
    # Run the query_retrieve function
    query_retrieve(f'{lib}/{model_name}', model, preprocessed_documents, doc_embeddings, descriptions=descriptions,
                   runid=f'{model_name}{"-descriptions" if descriptions else ""}', filename=filename, top_k=1000)

    # Run the trec_eval command
    p = subprocess.Popen(
        ["powershell", f".\\trec_eval -m map qrels1-50ap.txt {filename}"], stdout=sys.stdout)
    p.communicate()
    p = subprocess.Popen(
        ["powershell", f".\\trec_eval -m P qrels1-50ap.txt {filename}"], stdout=sys.stdout)
    p.communicate()

print('---\nDone')
