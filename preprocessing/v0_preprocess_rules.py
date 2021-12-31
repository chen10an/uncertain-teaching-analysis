# Reshape ending chunks into a dataframe, where each row represents a teacher's example and the student-participant's rule for that example

##
import pandas as pd
import numpy as np
import sklearn.metrics
import jmespath
import json
import os

OUTPUT_DIR_PATH ='../ignore/output/v0/'
SAVE_PATH = '../ignore/output/v0/rules.csv'

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
        for ex in exs:
            df = pd.DataFrame({
                'session_id': chunk['sessionId'],
                'form': form,
                'branches': [branches],  # make sure the branches array is put into the df as-is, without expanding into multiple rows
                'teaching_combo': ex['blicket_nonblicket_combo'],
                'teaching_activation': ex['detector_state']
            })

            df_list.append(df)

rule_df = pd.concat(df_list, ignore_index = True)
assert(rule_df.shape[0] == len(chunks) * 7 * 5)

##
print("Just give full bonus for missing rules and then look into it:")
print(rule_df[rule_df.branches.apply(lambda x: x == [])])

##
def eval_branch(branch, teaching_combo):
    """Evaluate a branch on a teaching combo

    :param branch: dict containing comparators and numbers for
        blickets and non-blickets; used to construct a function
        representation of the branch
    :param teaching_combo: string of '*' (blickets) and "."
        (non-blickets)

    :returns: float between 0 and 1 representing the probability of
              the branch outputting a positive activation for
              teaching_combo
    """
    # sanity check that comparators are filled in
    assert(branch['blicket_comparator'] is not None)
    assert(branch['nonblicket_comparator'] is not None)

    combo_blicket_num = teaching_combo.count('*')
    combo_nonblicket_num = teaching_combo.count('.')

    nonblicket_comparator = '==' if branch['nonblicket_comparator'] == '=' else branch['nonblicket_comparator']
    nonblicket_num = branch['nonblicket_num']

    if nonblicket_comparator == 'any':
        nonblicket_bool = True
    else:
        # sanity check that corresponding num is filled in
        assert(nonblicket_num is not None)
        nonblicket_bool = eval(f"{combo_nonblicket_num} {nonblicket_comparator} {nonblicket_num}")

    blicket_comparator = '==' if branch['blicket_comparator'] == '=' else branch['blicket_comparator']
    blicket_num = nonblicket_num if branch['blicket_num'] == 'nonblicket_num' else branch['blicket_num']
    
    if blicket_comparator == 'any':
        blicket_bool = True
    elif branch['blicket_num'] == 'nonblicket_num' and (nonblicket_num is None):
        # case when blicket num is supposed to be the same as nonblicket num and there can be any number of nonblickets
        assert(nonblicket_comparator == 'any')
        blicket_bool = True
    else:
        # sanity check that corresponding num is filled in
        assert(blicket_num is not None)
        
        blicket_bool = eval(f"{combo_blicket_num} {blicket_comparator} {blicket_num}")
        
    return branch['reliability'] * (blicket_bool and nonblicket_bool)

# test cases:
test_branch = {
    'reliability': 1,
    'blicket_comparator': '>=',
    'nonblicket_comparator': '=',
    'blicket_num': 3,
    'nonblicket_num': 1
}
assert(eval_branch(test_branch, "***.") == 1)
assert(eval_branch(test_branch, "*.**") == 1)
assert(eval_branch(test_branch, "*****.") == 1)
assert(eval_branch(test_branch, "***...") == 0)
assert(eval_branch(test_branch, "***") == 0)
assert(eval_branch(test_branch, "**.") == 0)
assert(eval_branch(test_branch, "") == 0)
assert(eval_branch(test_branch, ".") == 0)

test_branch_2 = {
    'reliability': 0.75,
    'blicket_comparator': '<=',
    'nonblicket_comparator': 'any',
    'blicket_num': 2,
    'nonblicket_num': None
}
assert(eval_branch(test_branch_2, "**..") == 0.75)
assert(eval_branch(test_branch_2, "**....") == 0.75)
assert(eval_branch(test_branch_2, "**") == 0.75)
assert(eval_branch(test_branch_2, "*.....") == 0.75)
assert(eval_branch(test_branch_2, "......") == 0.75)
assert(eval_branch(test_branch_2, "") == 0.75)
assert(eval_branch(test_branch_2, "***") == 0)
assert(eval_branch(test_branch_2, "***..") == 0)
assert(eval_branch(test_branch_2, "******") == 0)

def eval_dnf_branches(branches, teaching_combo):
    """Evaluate the disjunction of a list of branches

    :param branches: list of branches (dicts)
    :param teaching_combo: string of '*' (blickets) and "."
        (non-blickets)

    :returns: float between 0 and 1 representing the taking the
              maximum, or "disjunction", of all branches
    """
    if branches == []:
        return 0
    
    return max([eval_branch(b, teaching_combo) for b in branches])

##
rule_df['branches_p'] = rule_df.apply(lambda row: eval_dnf_branches(row.branches, row.teaching_combo) , axis=1)

##
def p_to_activation(row):
    if row.branches_p == 0.75:
        # make this noise match the teaching_activation for the sake of a generous f1-based bonus calculation
        return row.teaching_activation
    else:
        assert(row.branches_p in [0, 1])
        return bool(row.branches_p)

rule_df['branches_activation'] = rule_df.apply(p_to_activation, axis=1)
##
def f1(df):
    return sklearn.metrics.f1_score(y_true = df.teaching_activation, y_pred = df.branches_activation, pos_label = True, average = 'binary')

##
f1_scores = rule_df.groupby(['session_id', 'form']).apply(f1)
assert(f1_scores.shape[0] == len(ending_chunks) * 7)

##
bonuses = f1_scores * ending_chunks[0]['bonus_per_q']
bonuses = bonuses.groupby('session_id').sum().round(decimals=2)
bonuses = pd.merge(bonuses.to_frame("bonus"), id_df[['participant_id', 'session_id']], on='session_id')

print(bonuses.describe())

##
bonuses['participant_id'].to_csv('../ignore/bonus/v0/bonus_ids_00x.csv', index = False)
bonuses[['participant_id', 'bonus']].to_csv('../ignore/bonus/v0/bonus_ids_amounts_00x.csv', index = False)
print("Saved csvs to be used for bulk bonusing 00x on Prolifix!")
