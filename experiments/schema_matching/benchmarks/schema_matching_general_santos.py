import pprint
from valentine.algorithms import JaccardDistanceMatcher, Coma
from valentine import valentine_match
import os
import sys
import pandas as pd
import time
import datetime
import valentine.algorithms.matcher_results as matcher_results

project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.topk.indexed_similarity.indexed_similarity as indexed_similarity
import algorithms.schema_matching.topk.cl.cl as cl
from experiments.schema_matching.benchmarks.utils import compute_mean_ranking_reciprocal, create_result_file, record_result

HEADER = ['benchmark', 'dataset', 'source_table', 'target_table', 'method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
          'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']


pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)


def run_for_santos_benchmark():

    BENCHMARK = 'santos'

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', 'santos')
    result_file = os.path.join(results_dir,  'santos_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)
    root = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema matching data/santos_benchmark/'

    # interested = '311_calls_historic_data_a.csv'
    # interested = 'time_spent_watching_vcr_movies_a.csv'

    for file in os.listdir(root + "query"):

        # if file != interested:
        #     continue

        df_input = pd.read_csv(root + "query/" + file)
        print(f"Matching {file}")
        # print(df_input.head())

        if file.endswith("_a.csv"):
            table_prefix = file.replace("_a.csv", "")
        elif file.endswith("_b.csv"):
            table_prefix = file.replace("_b.csv", "")
        else:
            continue

        target_files = [f for f in os.listdir(
            root + "datalake") if f.startswith(table_prefix)]

        for target_file in target_files:
            df_target = pd.read_csv(root + "datalake/" + target_file)
            # print("To match with \n")
            # print(target_file)

            ground_truth = []
            for col in df_input.columns:
                if col in df_target.columns:
                    ground_truth.append((col, col))

            if len(ground_truth) == 0:
                continue

            # matchers = [indexed_fast.IndexedSimilarityMatcher(), Coma(), Coma(use_instances=True, java_xmx="10096m")]
            matchers = ["Coma", indexed_similarity.IndexedSimilarityMatcher(), cl.CLMatcher()]
            # matchers = ["Coma", "ComaInst", indexed_fast.IndexedSimilarityMatcher()]

            for matcher in matchers:
                print("Matcher: ", matcher)

                if matcher == "Coma":
                    method_name = "Coma"
                    matcher = Coma()
                elif matcher == "ComaInst":
                    method_name = "ComaInst"
                    matcher = Coma(use_instances=True, java_xmx="10096m")
                else:
                    method_name = str(matcher.__class__.__name__)

                start_time = time.time()

                try:
                    matches = valentine_match(df_input, df_target, matcher)
                except Exception as e:
                    print(f"Not able to run the matcher because of exception: {e}")
                    matches = matcher_results.MatcherResults({})

                end_time = time.time()
                runtime = end_time - start_time
                print(f"Runtime for valentine_match: {runtime:.4f} seconds")

                mrr_score = compute_mean_ranking_reciprocal(
                    matches, ground_truth)

                all_metrics = matches.get_metrics(ground_truth)

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                result = [BENCHMARK, table_prefix, file, target_file, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                          one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)
            # break


if __name__ == '__main__':
    # run_for_gdc()
    run_for_santos_benchmark()

    # run_for_gdc_alt()
