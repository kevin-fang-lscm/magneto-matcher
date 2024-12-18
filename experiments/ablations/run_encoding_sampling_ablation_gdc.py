import os
import sys
import pandas as pd
import time
import datetime
from itertools import product

project_path = os.getcwd()
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.magneto.magneto as mm
from valentine import valentine_match
from experiments.benchmarks.utils import (
    compute_mean_ranking_reciprocal,
    create_result_file,
    record_result,
)


def run_grid_search_experiment(
    BENCHMARK="gdc_studies", DATASET="gdc_studies", ROOT="./data/gdc"
):
    # Define parameter grid

    encoding_modes = [
        "header_values_default",
        "header_values_prefix",
        "header_values_repeat",
        "header_values_verbose",
        "header_only",
        "header_values_verbose_notype",
        "header_values_columnvaluepair_notype",
        "header_header_values_repeat_notype",
    ]

    # sampling_modes = ["random", "frequent", "mixed", "weighted", "priority_sampling", "consistent_sampling"]
    sampling_modes = ["frequent"]

    sampling_sizes = [10]

    # Extended header for grid search results
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
        "encoding_mode",
        "sampling_mode",
        "sampling_size",
        "runtime",
        "mrr",
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

    # Create results directory and file
    results_dir = os.path.join(
        project_path, "results", "ablations", "grid_search", BENCHMARK, DATASET
    )
    result_file = os.path.join(
        results_dir,
        f'{BENCHMARK}_{DATASET}_grid_search_results_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.csv',
    )
    create_result_file(results_dir, result_file, HEADER)

    # Load target data
    target_file = os.path.join(
        ROOT, "target-tables", "gdc_unique_columns_concat_values.csv"
    )
    df_target = pd.read_csv(target_file, low_memory=False)

    studies_path = os.path.join(ROOT, "source-tables")
    gt_path = os.path.join(ROOT, "ground-truth")

    # Process each ground truth file
    for gt_file in os.listdir(gt_path):
        if not gt_file.endswith(".csv"):
            continue

        # if gt_file!='Wang.csv':
        #     continue

        print(f"\nProcessing {gt_file}")

        # Load source data and ground truth
        source_file = os.path.join(studies_path, gt_file)
        df_source = pd.read_csv(source_file)

        gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
        gt_df.dropna(inplace=True)
        ground_truth = list(gt_df.itertuples(index=False, name=None))

        # Get dataset dimensions
        ncols_src = str(df_source.shape[1])
        ncols_tgt = str(df_target.shape[1])
        nrows_src = str(df_source.shape[0])
        nrows_tgt = str(df_target.shape[0])
        nmatches = len(ground_truth)

        # Grid search over all parameter combinations
        for encoding_mode, sampling_mode, sampling_size in product(
            encoding_modes, sampling_modes, sampling_sizes
        ):
            print(
                f"Testing configuration: {encoding_mode}, {sampling_mode}, {sampling_size}"
            )

            # Initialize matcher with current parameter combination
            matcher = mm.Magneto(
                encoding_mode=encoding_mode,
                sampling_mode=sampling_mode,
                sampling_size=sampling_size,
                include_strsim_matches=False,
                include_embedding_matches=True,
                include_equal_matches=False,
                use_bp_reranker=False,
                use_gpt_reranker=False,
            )

            # Run matching
            start_time = time.time()
            matches = valentine_match(df_source, df_target, matcher)
            runtime = time.time() - start_time

            # Calculate metrics
            mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
            all_metrics = matches.get_metrics(ground_truth)

            # Calculate one-to-one metrics
            one2one_matches = matches.one_to_one()
            one2one_metrics = one2one_matches.get_metrics(ground_truth)

            # Record result
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
                "Magneto",
                encoding_mode,
                sampling_mode,
                sampling_size,
                runtime,
                mrr_score,
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

            print(
                f"MRR: {mrr_score:.4f}, RecallAtGT {all_metrics['RecallAtSizeofGroundTruth']:.4f} Runtime: {runtime:.2f}s"
            )


if __name__ == "__main__":
    run_grid_search_experiment()
