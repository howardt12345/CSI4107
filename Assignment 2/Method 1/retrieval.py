# Query and write to file
import re
import os
import torch

import scipy
# from preprocessing import preprocess_text

# function to extract the topics from the topics file
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

def search(n, topic, model_name, model, preprocessed_documents, doc_embeddings, descriptions=False, top_k=20):
  # Get the query
  query = topic['title']
  if descriptions:
    query += ' ' + topic['description']
  filename = f'embedding_saves/{model_name}/query-{n}{"-descriptions" if descriptions else ""}.pickle'
  # Fetch the embeddings for the query if it exists, otherwise compute it
  if os.path.exists(filename):
    query_embeddings = torch.load(filename)
  else:
    query_embeddings = model.encode([query])
    os.makedirs(f'embedding_saves/{model_name}', exist_ok=True)
    torch.save(query_embeddings, filename)
  # compute distances
  distances = scipy.spatial.distance.cdist(query_embeddings, doc_embeddings, "cosine")[0]
  # get the top k results
  results = zip(range(len(distances)), distances)
  results = sorted(results, key=lambda x: x[1])
  # Create a list of tuples with the document number and the distance
  results = [(preprocessed_documents[idx].doc_no, distance) for idx, distance in results[0:top_k]]
  return results


# Go through all the documents and search for the top 1000 results
def query_retrieve(model_name, model, preprocessed_documents, doc_embeddings, descriptions=False, runid='runid', filename='Results.txt', top_k=1000):
  # Extract the topics
  topics = extract_topics('topics1-50.txt', descriptions)

  file_out = open(filename, 'w')

  for i, topic in enumerate(topics):
    # Search for the documents
    results = search(i, topic, model_name, model, preprocessed_documents, doc_embeddings, descriptions, top_k)
    for j, (doc_id, distance) in enumerate(results):
      file_out.write(f'{i+1} Q0 {doc_id.strip()} {j+1} {1-distance} {runid}\n')
  file_out.close()
  # print('Written results to file', filename)

