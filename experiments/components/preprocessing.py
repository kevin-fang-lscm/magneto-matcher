import os
import sys
import pandas as pd
import time

project_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.topk.indexed_similarity as isim
import algorithms.schema_matching.topk.cl.cl as cl
import sm_components.src.preprocessing.data_preprocessor as dp
# from algorithms.schema_matching.topk.indexed_similarity import IndexedSimilarityMatcher

from valentine.metrics import F1Score, PrecisionTopNPercent
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher, Coma
import pprint

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)

def run():

    gdc_input_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-discovery.csv')
    gdc_target_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-confirmatory.csv')




    gdc_input_df = dp.DataPreprocessor(gdc_input_df).clean_data().get_df()
    gdc_target_df = dp.DataPreprocessor(gdc_target_df).clean_data().get_df()

    
    #TODO adjust GT to match the transformed column names

    gt_df = pd.read_csv(project_path+'/data/gdc_alt/gt.csv')
    gt_df.dropna(inplace=True)
    ground_truth = list(gt_df.itertuples(index=False, name=None))



    matchers = [Coma()]

    # Coma()

    for matcher in matchers:
        print("Matcher: ", matcher)

        start_time = time.time()

        matches = valentine_match(gdc_input_df, gdc_target_df, matcher)
        metrics = matches.get_metrics(ground_truth)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime for valentine_match: {runtime:.4f} seconds")


        print("Normal Metrics:")
        pp.pprint(metrics)

        matches = matches.one_to_one()
        metrics = matches.get_metrics(ground_truth)
        print("One2One Metrics:")
        pp.pprint(metrics)
        print("\n\n")

if __name__ == '__main__':
    run()
