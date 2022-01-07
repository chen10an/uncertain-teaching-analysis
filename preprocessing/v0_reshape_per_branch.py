# Reshape ending chunks into a dataframe, where each row represents a branch of a student-participant's rule

##
import pandas as pd
import numpy as np
import sklearn.metrics
import jmespath
import json
import os

OUTPUT_DIR_PATH ='../ignore/output/v0/'
SAVE_PATH = '../ignore/output/v0/branches.csv'

##
# valid ending chunks
with open(os.path.join(OUTPUT_DIR_PATH, 'valid_end_chunks_00x.json')) as f:
    ending_chunks = json.load(f)

# ids
with open('../ignore/data/d_prolific_worker_ids_00x.tsv') as f:
    id_df = pd.read_csv(f, sep='\t')
##
# query chunks json for quiz-related data
chunks = jmespath.search("[?seq_key=='End'].{sessionId: sessionId, end_time: timestamp, route: route, condition_name: condition_name, is_trouble: is_trouble, quiz_data: quiz_data, bonus_per_q: bonus_per_q}", ending_chunks)

##
df_list = []
for chunk in chunks:
    quiz = chunk['quiz_data']
    for form in quiz:
        exs = quiz[form]['teaching_ex']

        if 'rule' not in quiz[form]:
            branches = []
        else:
            branches = [x['branch'] for x in quiz[form]['rule']]
        for i in range(len(branches)):
            df = pd.DataFrame({
                'session_id': chunk['sessionId'],
                'form': form,
                'branch_dex': i,
                'reliability': branches[i]['reliability'],
                'blicket_comparator': branches[i]['blicket_comparator'],
                'blicket_num': branches[i]['blicket_num'],
                'nonblicket_comparator': branches[i]['nonblicket_comparator'],
                'nonblicket_num': branches[i]['nonblicket_num'],
                'exs': [exs],  # make sure the exs array is put into the df as-is, without expanding into multiple rows
            })

            df_list.append(df)

branch_df = pd.concat(df_list, ignore_index = True)

##
branch_df.to_csv(SAVE_PATH, index = False)
print(f"Saved branch_df to {SAVE_PATH}!")
