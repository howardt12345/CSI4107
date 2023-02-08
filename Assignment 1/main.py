
import pyterrier as pt
import pandas as pd
import nltk
nltk.download('punkt')
from preprocessing import preprocess_directory, preprocess


if not pt.started():
  pt.init()

# Preprocess the collection
preprocessed_documents = preprocess_directory('AP_collection/coll', 1)

# Create a dataframe from the preprocessed documents
df = pd.DataFrame.from_records([doc.to_dict() for doc in preprocessed_documents])
df.head()

print(df)

# Create a Terrier index from the dataframe
pd_indexer = pt.DFIndexer("pd_index")
indexref = pd_indexer.index(df["tokens"], df["docno"])

# Create a BM25 retrieval model
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")

# use the BM25 model to retrieve the top 10 documents for the query "information retrieval"
bm25.search("information retrieval", wmodel="BM25", k=10)

