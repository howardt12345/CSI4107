# Query and write to file
import re
from preprocessing import preprocess_text

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

# function to query the model and write the results to a file
def query_retrieve(model, descriptions=False, runid='runid', filename='Results.txt'):
  topics = extract_topics("topics1-50.txt", descriptions)

  bm_file_out = open(filename, 'w')

  for i, topic in enumerate(topics):
    curr_result = query(model, topic, descriptions)
    for j in range(len(curr_result)):
      result_row = curr_result.iloc[j]
      bm_file_out.write(str(i+1) + " " + "Q0 " + result_row['docno'] + " " + str(
          result_row['rank']+1) + " " + str(result_row['score']) + " " + runid + "\n")

  bm_file_out.close()
  print("Written results to file: " + filename)

def query(model, topic, descriptions=False):
  t = " ".join(preprocess_text(topic['title'], stem=False, stopwords=False))
  if descriptions:
    t += " " + \
        " ".join(preprocess_text(
            topic['description'], stem=False, stopwords=False))
  return model.wv.most_similar(t)