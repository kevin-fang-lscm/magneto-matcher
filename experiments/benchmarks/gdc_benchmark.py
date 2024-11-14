import os
import sys
import pandas as pd
import time
import datetime

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_path))


import algorithms.schema_matching.topk.match_maker.match_maker as mm

import algorithms.schema_matching.topk.retrieve_match.retrieve_match as rema
from experiments.benchmarks.utils import compute_mean_ranking_reciprocal, compute_mean_ranking_reciprocal_detail, create_result_file, record_result,extract_matchings

from valentine import valentine_match
from valentine.algorithms import Coma
import pprint
import sys

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)


def get_matcher(method):
    if method == 'Coma':
        return Coma()
    elif method == 'ComaInst':
        return Coma(use_instances=True, java_xmx="10096m")   
    elif method == 'MatchMaker':
        return mm.MatchMaker()
    elif method == 'MatchMakerFT':
        model_path = os.path.join(os.path.expanduser("~"), 'models', 'mpnet-gdc-exact_semantic.pth')
        return mm.MatchMaker(fine_tune_path=model_path)
    elif method == 'MatchMakerGPT':
        return mm.MatchMaker(use_instances=False, use_gpt=True)


def run_benchmark(BENCHMARK='gdc_studies', DATASET='gdc_studies', ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/gdc'):

    HEADER = ['benchmark', 'dataset', 'source_table', 'target_table','ncols_src','ncols_tgt','nrows_src','nrows_tgt','nmatches','method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', BENCHMARK, DATASET)
    result_file = os.path.join(results_dir,  BENCHMARK + '_' + DATASET + '_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)

    target_file = os.path.join(ROOT, 'target-tables', 'gdc_unique_columns_concat_values.csv')
    
    df_target = pd.read_csv(target_file)

    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith('.csv'):

            print(f"Processing {gt_file}")

            # if gt_file != 'Krug.csv':
            #     continue
            source_file = os.path.join(studies_path, gt_file)
            df_source = pd.read_csv(source_file)

            gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
            gt_df.dropna(inplace=True)
            ground_truth = list(gt_df.itertuples(index=False, name=None))


            matchers = [ "MatchMaker",  "MatchMakerFT"]

            for matcher in matchers:
                # print(f"Matcher: {matcher}, Source: {source_file}, Target: {target_file}")

                method_name = matcher
                matcher = get_matcher(matcher)    

                start_time = time.time()
                matches = valentine_match(df_source, df_target, matcher)
                end_time = time.time()
                runtime = end_time - start_time
                
                mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
                
                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

                print('File: ' , gt_file, ' and ', method_name, " with MRR Score: ", mrr_score, ", RecallAtGT: ", recallAtGT, " and Runtime: ", runtime)    

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                ncols_src = str(df_source.shape[1])
                ncols_tgt = str(df_target.shape[1])
                nrows_src = str(df_source.shape[0])
                nrows_tgt = str(df_target.shape[0])

                nmatches = len(ground_truth)

                result = [BENCHMARK, DATASET, 'gdc_table', gt_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt,nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                      one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)
            print("\n")



if __name__ == '__main__':
    run_benchmark()
    
    
