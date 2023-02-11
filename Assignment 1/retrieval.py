# Query and write to file 

from preprocessing import preprocess_text, extract_topics


def query_retrieve(model):
    topics = extract_topics("topics1-50.txt")

    bm_file_out = open('Results.txt', 'w')

    for i in range(len(topics)):
        curr_result = model.search(preprocess_text((str(topics[i]))))
        for j in range(len(curr_result)):
            result_row = curr_result.iloc[j]
            bm_file_out.write(str(i+1) + " " + "Q0 " + result_row['docno'] + " " + str(result_row['rank']+1) + " " + str(result_row['score']) + " " + "runid\n")
            
    bm_file_out.close()