# Query and write to file

import re
from preprocessing import preprocess_text

# function to extract the topics from the topics file
def extract_topics(file):
  with open(file, "r") as f:
    topic_content = f.read()
  all_topics = []
  topics = re.findall(r'<top>(.*?)</top>', topic_content, re.DOTALL)
  for topic in topics:
    raw_title = re.search(r'<title>(.*?)\n\n', topic, re.DOTALL)
    title = raw_title.group(1) if raw_title else ''
    all_topics.append(title)
  return all_topics

# function to query the model and write the results to a file
def query_retrieve(model):
  topics = extract_topics("topics1-50.txt")

  bm_file_out = open('Results.txt', 'w')

  for i, topic in enumerate(topics):
    t = " ".join(preprocess_text(topic, stem=False, stopwords=False))
    curr_result = model.search(t)
    for j in range(len(curr_result)):
      result_row = curr_result.iloc[j]
      bm_file_out.write(str(i+1) + " " + "Q0 " + result_row['docno'] + " " + str(
          result_row['rank']+1) + " " + str(result_row['score']) + " " + "runid\n")

  bm_file_out.close()
  print("Written results to Results.txt")