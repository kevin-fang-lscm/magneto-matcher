import os
import sys
import pandas as pd
import time
import pprint

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)

project_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_path))

from experiments.benchmarks.benchmark_utils import (
    compute_mean_ranking_reciprocal_adjusted,
    create_result_file,
    record_result,
    calculate_recall_at_k,
)
from algorithms.magneto.magneto import Magneto


def get_matcher(
    topk = 20,
):
        if topk == 0:
            return Magneto(
                encoding_mode="header_values_repeat",
                embedding_model="mpnet",
            )
        else:
            return Magneto(
                encoding_mode="header_values_repeat",
                embedding_model="mpnet",
                llm_model="gpt-4o-mini",
                use_bp_reranker=False,
                use_gpt_reranker=True,
                topk=topk,
            )


def run_benchmark(
    BENCHMARK="gdc_studies",
    DATASET="gdc_studies",
    ROOT="data/gdc",
):

    HEADER = [
        "benchmark",
        "dataset",
        "source_table",
        "target_table",
        "ncols_src",
        "ncols_tgt",
        "nrows_src",
        "nrows_tgt",
        "nmatches",
        "method",
        "runtime",
        "mrr",
        "Recall@20",
        "All_Precision",
        "All_F1Score",
        "All_Recall",
        "All_PrecisionTop10Percent",
        "All_RecallAtSizeofGroundTruth",
        "One2One_Precision",
        "One2One_F1Score",
        "One2One_Recall",
        "One2One_PrecisionTop10Percent",
        "One2One_RecallAtSizeofGroundTruth",
    ]

    results_dir = os.path.join(
        project_path, "results", "benchmarks", BENCHMARK
    )
    result_file = os.path.join(
        results_dir,
        DATASET
        + "-"
        + "ablations"
        # + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        + ".csv",
    )
    print(result_file)

    create_result_file(results_dir, result_file, HEADER)

    target_file = os.path.join(
        ROOT, "target-tables", "gdc_unique_columns_concat_values.csv"
    )

    df_target = pd.read_csv(target_file, low_memory=False)

    studies_path = os.path.join(ROOT, "source-tables")
    gt_path = os.path.join(ROOT, "ground-truth")

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith(".csv"):

            print(f"Processing {gt_file}")

            source_file = os.path.join(studies_path, gt_file)
            df_source = pd.read_csv(source_file)

            gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
            gt_df.dropna(inplace=True)
            ground_truth = list(gt_df.itertuples(index=False, name=None))

            matcher_name = "MagnetoGPT"
            ks = [0, 3, 5, 10]

            for topk in ks:

                matcher = get_matcher(topk=topk)

                start_time = time.time()
                matches = matcher.get_matches(df_source, df_target)
                end_time = time.time()
                runtime = end_time - start_time

                # print("Matches: ", matches)

                mrr_score = compute_mean_ranking_reciprocal_adjusted(
                    matches, ground_truth
                )

                recall_at_k = calculate_recall_at_k(matches, ground_truth)

                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics["RecallAtSizeofGroundTruth"]

                print(
                    "File: ",
                    gt_file,
                    " and ",
                    matcher_name,
                    " with MRR Score: ",
                    mrr_score,
                    ", RecallAtGT: ",
                    recallAtGT,
                    ", Recall@20: ",
                    recall_at_k,
                    ", and Runtime: ",
                    runtime,
                )

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                ncols_src = str(df_source.shape[1])
                ncols_tgt = str(df_target.shape[1])
                nrows_src = str(df_source.shape[0])
                nrows_tgt = str(df_target.shape[0])

                nmatches = len(ground_truth)

                result = [
                    BENCHMARK,
                    DATASET,
                    "gdc_table",
                    gt_file,
                    ncols_src,
                    ncols_tgt,
                    nrows_src,
                    nrows_tgt,
                    nmatches,
                    matcher_name + f"_{topk}",
                    runtime,
                    mrr_score,
                    recall_at_k,
                    all_metrics["Precision"],
                    all_metrics["F1Score"],
                    all_metrics["Recall"],
                    all_metrics["PrecisionTop10Percent"],
                    all_metrics["RecallAtSizeofGroundTruth"],
                    one2one_metrics["Precision"],
                    one2one_metrics["F1Score"],
                    one2one_metrics["Recall"],
                    one2one_metrics["PrecisionTop10Percent"],
                    one2one_metrics["RecallAtSizeofGroundTruth"],
                ]

                record_result(result_file, result)
            print("\n")


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
