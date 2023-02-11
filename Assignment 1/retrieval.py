# Query and write to file 

import string
from preprocessing import extract_topics


def query_retrieve(model):
    topics = extract_topics("topics1-50.txt")

    bm_file_out = open('Results.txt', 'w')

    for i, topic in enumerate(topics):
        t = topic.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        curr_result = model.search(t)
        for j in range(len(curr_result)):
            result_row = curr_result.iloc[j]
            bm_file_out.write(str(i+1) + " " + "Q0 " + result_row['docno'] + " " + str(result_row['rank']+1) + " " + str(result_row['score']) + " " + "runid\n")
            
    bm_file_out.close()