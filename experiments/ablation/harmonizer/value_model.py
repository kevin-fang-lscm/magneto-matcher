from utils import compute_mean_ranking_reciprocal

import pandas as pd
import os
import sys

from datasets_builder import get_triplets, get_gt
from valentine import valentine_match


project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.topk.harmonizer.harmonizer as hm


def run():

    triplets = get_triplets('GDC')

    results = []

    for source, target, gt in triplets:
        
        print('Source:', source)

        df_source = pd.read_csv(source)
        df_target = pd.read_csv(target)
        ground_truth = get_gt(gt)

        # matcher = hm.Harmonizer(use_instances=True)
        matcher = hm.Harmonizer(value_model_name='./models/gdc_values_fine_tuned_model', use_instances=True)

        matches = valentine_match(df_source, df_target, matcher)

        # print(matches)

        mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

        all_metrics = matches.get_metrics(ground_truth)

        recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

        results.append((mrr_score, recallAtGT))

        print('Harmonizer', " with MRR Score: ",
              mrr_score, " and RecallAtGT: ", recallAtGT)

        # print(df_source.columns)
        # print(df_target.columns)
        # print(ground_truth)

        # break
    print('Results:', )
    print(results)

    mean_mrr = sum([x[0] for x in results])/len(results)
    mean_recall = sum([x[1] for x in results])/len(results)
    print('Mean MRR:', mean_mrr)
    print('Mean Recall:', mean_recall)


if __name__ == "__main__":
    run()
