
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
  print(preprocessed_documents)

  # Create a dataframe from the preprocessed documents
  df = pd.DataFrame.from_records([doc.to_dict() for doc in preprocessed_documents])
  df.head()

  # Create a Terrier index from the dataframe
  pd_indexer = pt.IterDictIndexer(os.path.abspath('./pd_index'), overwrite=True)
  indexref = pd_indexer.index(df.to_dict(orient='records'))

  return indexref

# Check if the index exists, if not create it
if not os.path.exists('./pd_index'):
  indexref = generate_index()
else:
  indexref = pt.IndexFactory.of(os.path.abspath('./pd_index/data.properties'))

# Create a BM25 retrieval model
bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
query_retrieve(bm25)

# # Query and write to
# topics = extract_topics("topics1-50.txt")

# bm_file_out = open('Results.txt', 'w')

# for i in range(len(topics)):
#     print("topic " + str(i) +": " + topics[i])
#     curr_result = bm25.search(preprocess_text((str(topics[i]))))
#     for j in range(len(curr_result)):
#         result_row = curr_result.iloc[j]
#         bm_file_out.write(str(i+1) + " " + "Q0 " + result_row['docno'] + " " + str(result_row['rank']+1) + " " + str(result_row['score']) + " " + "runid\n")
        
# bm_file_out.close()



# # use the BM25 model to index
# result = bm25.search("Coping with overcrowded prisons")
# print('BM25')
# print(result)

# #print file out to Results 
# bm_file_out = open('Results.txt', 'w')
# bm_file_out.write(result.to_string())
# bm_file_out.close()

# # Use the tf-idf retrieval model to index
# tfidf = pt.BatchRetrieve(indexref, wmodel="TF_IDF")
# result = tfidf.search("Coping with overcrowded prisons")
# print('\nTF-IDF')
# print(result)

