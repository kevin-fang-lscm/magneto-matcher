import os
import pandas as pd
from utils import are_similar_ignore_case_and_punctuation, are_equal_ignore_case_and_punctuation



if __name__ == '__main__':

    files = ['Dou.csv', 'Krug.csv', 'Clark.csv',  'Vasaikar.csv',
             'Wang.csv', 'Satpathy.csv', 'Cao.csv', 'Huang.csv', 'Gilette.csv']
    # file = files[-1]

    positions_global = []

    gts = []
    for file in files:
       
        gt_df = pd.read_csv(
        f'./data/gdc/ground-truth/{file}', encoding='utf-8', engine='python')
        gt = list(gt_df.itertuples(index=False, name=None))

        gts.extend(gt)

    gts = [tuple(map(str.lower, row)) for row in gts]

    print(len(gts))

count = 0
for pair in gts:
    print(pair)
    if are_equal_ignore_case_and_punctuation(pair[0], pair[1]) or are_similar_ignore_case_and_punctuation(pair[0], pair[1],80):
        count += 1
        # if not are_equal_ignore_case_and_punctuation(pair[0], pair[1]) and are_similar_ignore_case_and_punctuation(pair[0], pair[1],80):
        #     print(f"Similar: {pair}")
    # else:
    #     print(pair)

print(f"Number of pairs that are equal or similar: {count}")

       

       

  