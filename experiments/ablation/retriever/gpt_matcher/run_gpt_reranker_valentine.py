
import pprint

from itertools import product
import os
import sys
import json
import pandas as pd
import time
import datetime
from valentine import valentine_match
from valentine.algorithms import Coma
import valentine.algorithms.matcher_results as matcher_results

import warnings
warnings.simplefilter('ignore', FutureWarning)


project_path = os.getcwd()
sys.path.append(os.path.join(project_path))


import algorithms.schema_matching.gpt_matcher.gpt_matcher as gpt_matcher
import algorithms.schema_matching.match_maker.match_maker as mm
from experiments.benchmarks.utils import compute_mean_ranking_reciprocal, create_result_file, record_result


pp = pprint.PrettyPrinter(indent=4)


def extract_matchings(json_data):
    data = json.loads(json_data)
    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings


def get_gpt_method(method):
    if method == 'GPTMatcher':
        return gpt_matcher.GPTMatcher()
    elif method == 'GPTMatcherExample':
        return gpt_matcher.GPTMatcher(include_example=True)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_match_maker_matcher(method):
    return mm.MatchMaker()


def get_matcher(method):
    if method.startswith('GPT'):
        return get_gpt_method(method)
    elif method.startswith('MatchMaker'):
        return get_match_maker_matcher(method)
    else:
        raise ValueError(f"Unknown method: {method}")


def run_valentine_benchmark_one_level(BENCHMARK='valentine', DATASET='musicians', ROOT='./data/valentine/Wikidata/Musicians'):

    # Extended header for grid search results
    HEADER = [
        'benchmark', 'dataset', 'source_table', 'target_table',
        'ncols_src', 'ncols_tgt', 'nrows_src', 'nrows_tgt', 'nmatches',
        'method', 'encoding_mode', 'sampling_mode', 'sampling_size',
        'strsim', 'emebedding', 'equal',
        'runtime', 'mrr',
        'All_Precision', 'All_F1Score', 'All_Recall',
        'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
        'One2One_Precision', 'One2One_F1Score', 'One2One_Recall',
        'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth'
    ]

    # Create results directory and file
    results_dir = os.path.join(
        project_path, 'results', 'ablations', 'gpt_reranker',
        BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_gpt_reranker_results_{
            datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    )
    create_result_file(results_dir, result_file, HEADER)

    for folder in os.listdir(ROOT):
        if folder == '.DS_Store' or folder == '.ipynb_checkpoints':
            continue

        source_file = os.path.join(ROOT, folder, folder+'_source.csv')
        target_file = os.path.join(ROOT, folder, folder+'_target.csv')
        mapping_file = os.path.join(ROOT, folder, folder+'_mapping.json')

        df_source = pd.read_csv(source_file, low_memory=False)
        df_target = pd.read_csv(target_file, low_memory=False)
        ground_truth = extract_matchings(open(mapping_file).read())

        ncols_src = str(df_source.shape[1])
        ncols_tgt = str(df_target.shape[1])
        nrows_src = str(df_source.shape[0])
        nrows_tgt = str(df_target.shape[0])
        nmatches = len(ground_truth)

        matchers = ["GPTMatcher", "GPTMatcherExample", "MatchMaker"]

        for matcher in matchers:

            print("Running matcher: ", matcher)

            method_name = matcher
            matcher = get_matcher(matcher)

            start_time = time.time()

            try:
                matches = valentine_match(df_source, df_target, matcher)
            except Exception as e:
                print(f"Not able to run the matcher because of exception: {e}")
                matches = matcher_results.MatcherResults({})
            # matches = valentine_match(df_source, df_target, matcher)

            end_time = time.time()
            runtime = end_time - start_time

            mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

            all_metrics = matches.get_metrics(ground_truth)

            recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

            print(method_name, " with MRR Score: ", mrr_score,
                  " and RecallAtGT: ", recallAtGT, " and runtime: ", runtime)

            matches = matches.one_to_one()
            one2one_metrics = matches.get_metrics(ground_truth)

        break


if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    run_valentine_benchmark_one_level()

    # Magellan

    # OpenData
    # run_valentine_benchmark_three_levels()

    # # ChEMBLc
    # DATASET='ChEMBL'
    # ROOT='./data/valentine/ChEMBL/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='./data/valentine/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
