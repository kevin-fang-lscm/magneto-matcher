import os
import sys
import pandas as pd
import time

project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.topk.indexed_similarity as isim
import algorithms.schema_matching.topk.cl.cl as cl

# from algorithms.schema_matching.topk.indexed_similarity import IndexedSimilarityMatcher

from valentine.metrics import F1Score, PrecisionTopNPercent
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher, Coma
import pprint
import sys

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)


def sort_matches(matches):

    sorted_matches = {entry[0][1]: [] for entry in matches}
    for entry in matches:
        sorted_matches[entry[0][1]] .append((entry[1][1], matches[entry]))

    # for key in sorted_matches:
    #     print(key, ' ', sorted_matches[key])
    return sorted_matches


def compute_mean_ranking_reciprocal(matches, ground_truth):
    # print("Matches: ", matches)
    # print("Ground Truth: ", ground_truth)

    ordered_matches = sort_matches(matches)
#
    total_score = 0
    for input_col, target_col in ground_truth:
        score = 0
        # print("Input Col: ", input_col)
        if input_col in ordered_matches:
            ordered_matches_list = [v[0] for v in ordered_matches[input_col]]
            position = -1
            if target_col in ordered_matches_list:
                position = ordered_matches_list.index(target_col)
                score = 1/(position + 1)
            else:
                print(f"1- Mapping {input_col} -> {target_col} not found")
                # for entry in ordered_matches[input_col]:
                #     print(entry)
        # else:
            # print(f"2- Mapping {input_col} -> {target_col} not found")
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score


def run_for_gdc_alt():
    print(project_path)
    gdc_input_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-discovery.csv')
    gdc_target_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-confirmatory.csv')

    gt_df = pd.read_csv(project_path+'/data/gdc_alt/gt.csv')
    gt_df.dropna(inplace=True)
    ground_truth = list(gt_df.itertuples(index=False, name=None))

    # print(ground_truth)

    # print(gdc_input_df.head())

    # print(gdc_target_df.head())
    # matchers = [ IndexedSimilarityMatcher(),  JaccardDistanceMatcher(), Coma()]
    # matchers = [IndexedSimilarityMatcher(), Coma(), Coma(
    #     use_instances=True, java_xmx="10096m")]

    # matchers = [isim.IndexedSimilarityMatcher()]
    matchers = [Coma(),isim.IndexedSimilarityMatcher(), cl.CLMatcher()]

    # Coma()

    for matcher in matchers:
        print("Matcher: ", matcher)

        start_time = time.time()

        matches = valentine_match(gdc_input_df, gdc_target_df, matcher)
        metrics = matches.get_metrics(ground_truth)



        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime for valentine_match: {runtime:.4f} seconds")

        score = compute_mean_ranking_reciprocal(matches, ground_truth)
        print("Mean Ranking Reciprocal Score: ", score)

        print("Normal Metrics:")
        pp.pprint(metrics)

        matches = matches.one_to_one()
        metrics = matches.get_metrics(ground_truth)
        print("One2One Metrics:")
        pp.pprint(metrics)
        print("\n\n")


if __name__ == '__main__':
    # run_for_gdc()
    # run_for_santos_benchmark()

    run_for_gdc_alt()
