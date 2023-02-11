
from retrieval import query_retrieve
import pyterrier as pt
import pandas as pd
import os
import nltk
nltk.download('punkt')
from preprocessing import preprocess_directory


if not pt.started():
  pt.init()

# Function to generate the index
def generate_index():
  # Preprocess the collection
  preprocessed_documents = preprocess_directory('AP_collection/coll')

  # Create a dataframe from the preprocessed documents
  df = pd.DataFrame.from_records([doc.to_dict() for doc in preprocessed_documents])
  df.head()

  # Create a Terrier index from the dataframe
  pd_indexer = pt.IterDictIndexer(os.path.abspath('./pd_index'), overwrite=True)
  indexref = pd_indexer.index(df.to_dict(orient='records'))

  return indexref

# Check if the index exists, if not create it
if not os.path.exists('./pd_index'):
  print("Index does not exist, creating it...")
  indexref = generate_index()
else:
  print("Index exists, loading it...")
  indexref = pt.IndexFactory.of(os.path.abspath('./pd_index/data.properties'))
print("Index loaded successfully!")

# Create a retrieval model
weighting = "TF_IDF"
print("Using weighting model:", weighting)
model = pt.BatchRetrieve(indexref, wmodel=weighting, num_results=1000)

# Query the model and write the results
query_retrieve(model, descriptions=True)