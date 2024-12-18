import pprint
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
from tqdm import tqdm

warnings.simplefilter("ignore", FutureWarning)


project_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(os.path.join(project_path))


from experiments.benchmarks.utils import (
    compute_mean_ranking_reciprocal,
    create_result_file,
    record_result,
)
import algorithms.schema_matching.magneto.magneto as mm


pp = pprint.PrettyPrinter(indent=4)


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [
        (match["source_column"], match["target_column"]) for match in data["matches"]
    ]
    return matchings


def get_matcher(method, model_name=None, mode="header_values_default"):
    if method == "Coma":
        return Coma()
    elif method == "ComaInst":
        return Coma(use_instances=True, java_xmx="10096m")

    elif method == "Magneto":
        return mm.Magneto()
    elif method == "MagnetoGPT":
        return mm.Magneto(use_instances=False, use_gpt=True)

    elif method == "MagnetoFT":
        model_path = os.path.join(project_path, "models", model_name,)
        return mm.Magneto(encoding_mode=mode, embedding_model=model_path)
    elif method == "MagnetoGPT":
        return mm.Magneto(use_bp_reranker=False, use_gpt_reranker=True)
    elif method == "MagnetoFTGPT":
        model_path = os.path.join(project_path, "models", model_name,)
        return mm.Magneto(
            encoding_mode=mode,
            embedding_model=model_path,
            use_bp_reranker=False,
            use_gpt_reranker=True,
        )


def run_valentine_benchmark_one_level(
    BENCHMARK="valentine",
    DATASET="musicians",
    ROOT="data/valentine/Wikidata/Musicians",
    MODE="header_values_default",
):
    """
    Run the valentine benchmark for one level of the dataset (Magelan and Wikidata)
    """

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

    dataset_name = DATASET.lower()
    model_names = [
        f"mpnet-{dataset_name}-{MODE}-exact_semantic-16-0.5.pth",
        f"mpnet-{dataset_name}-{MODE}-exact-16-0.5.pth",
        f"mpnet-{dataset_name}-{MODE}-semantic-16-0.5.pth",
    ]

    for model_name in model_names:

        print(f"Model: {model_name}")

        results_dir = os.path.join(
            project_path, "results", "benchmarks", BENCHMARK, DATASET
        )
        result_file = os.path.join(
            results_dir,
            DATASET
            + "-"
            + model_name.replace("-16-0.5.pth", "").replace("mpnet-", "")
            + "_results"
            + ".csv",
        )
        print(result_file)

        create_result_file(results_dir, result_file, HEADER)

        for folder in os.listdir(ROOT):
            if folder == ".DS_Store":
                continue

            source_file = os.path.join(ROOT, folder, folder.lower() + "_source.csv")
            target_file = os.path.join(ROOT, folder, folder.lower() + "_target.csv")
            mapping_file = os.path.join(ROOT, folder, folder.lower() + "_mapping.json")

            df_source = pd.read_csv(source_file)
            df_target = pd.read_csv(target_file)
            ground_truth = extract_matchings(open(mapping_file).read())

            ncols_src = str(df_source.shape[1])
            ncols_tgt = str(df_target.shape[1])
            nrows_src = str(df_source.shape[0])
            nrows_tgt = str(df_target.shape[0])

            nmatches = len(ground_truth)

            # print(ground_truth)

            if len(ground_truth) == 0:
                continue

            matchers = ["Magneto", "MagnetoGPT", "MagnetoFT", "MagnetoFTGPT"]

            for matcher in matchers:

                print("Running matcher: ", matcher)

                method_name = matcher
                matcher = get_matcher(matcher, model_name, MODE)

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

                recallAtGT = all_metrics["RecallAtSizeofGroundTruth"]

                print(
                    method_name,
                    " with MRR Score: ",
                    mrr_score,
                    " and RecallAtGT: ",
                    recallAtGT,
                )

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                source_file = source_file.split("/")[-1]
                target_file = target_file.split("/")[-1]

                result = [
                    BENCHMARK,
                    DATASET,
                    source_file,
                    target_file,
                    ncols_src,
                    ncols_tgt,
                    nrows_src,
                    nrows_tgt,
                    nmatches,
                    method_name,
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


def run_valentine_benchmark_three_levels(
    BENCHMARK="valentine",
    DATASET="OpenData",
    ROOT="data/valentine/OpenData/",
    MODE="header_values_default",
):
    """
    Run the valentine benchmark for datasets split on Unionable, View-Unionable, Joinable, Semantically-Joinable
    """

    HEADER = [
        "benchmark",
        "dataset",
        "type",
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

    dataset_dict = {
        "OpenData": "opendata",
        "Magellan": "magellan",
        "ChEMBL": "chembl",
        "TPC-DI": "tpc",
        "Wikidata": "wikidata",
    }
    dataset_name = dataset_dict[DATASET]

    model_names = [
        # f"mpnet-{dataset_name}-{MODE}-exact_semantic-16-0.5.pth",
        # f"mpnet-{dataset_name}-{MODE}-exact-16-0.5.pth",
        f"mpnet-{dataset_name}-{MODE}-semantic-16-0.5.pth",
    ]

    for model_name in model_names:

        print(f"Model: {model_name}")

        results_dir = os.path.join(
            project_path, "results", "benchmarks", BENCHMARK, DATASET
        )
        result_file = os.path.join(
            results_dir,
            DATASET
            + "-"
            + model_name.replace("-16-0.5.pth", "").replace("mpnet-", "")
            + "_results_1"
            + ".csv",
        )
        print(result_file)

        if not os.path.exists(result_file):
            create_result_file(results_dir, result_file, HEADER)
            print("Created result file")

        table_count = 0

        for type in os.listdir(ROOT):
            if type == ".DS_Store":
                continue

            print("Type: ", type)
            for table_folder in os.listdir(os.path.join(ROOT, type)):

                if table_folder == ".DS_Store":
                    continue

                source_file = os.path.join(
                    ROOT, type, table_folder, table_folder + "_source.csv"
                )
                target_file = os.path.join(
                    ROOT, type, table_folder, table_folder + "_target.csv"
                )
                mapping_file = os.path.join(
                    ROOT, type, table_folder, table_folder + "_mapping.json"
                )

                ground_truth = extract_matchings(open(mapping_file).read())

                df_source = pd.read_csv(source_file)
                df_target = pd.read_csv(target_file)

                ncols_src = str(df_source.shape[1])
                ncols_tgt = str(df_target.shape[1])
                nrows_src = str(df_source.shape[0])
                nrows_tgt = str(df_target.shape[0])
                nmatches = len(ground_truth)

                if len(ground_truth) == 0:
                    continue

                table_count += 1

                matchers = ["Magneto", "MagnetoGPT", "MagnetoFT", "MagnetoFTGPT"]

                for matcher in matchers:
                    print("Running matcher: ", matcher)

                    # if a dataset with same type, source_table, target_table, and method exists in result file, skip
                    # if os.path.exists(result_file):
                    #     df = pd.read_csv(result_file)
                    #     if not df[(df['type'] == type) & (df['source_table'] == source_file.split("/")[-1]) & (df['target_table'] == target_file.split("/")[-1]) & (df['method'] == matcher)].empty:
                    #         print(f"Skipping {source_file.split('/')[-1]} and {target_file.split('/')[-1]} for {matcher}")
                    #         continue

                    method_name = matcher
                    matcher = get_matcher(matcher, model_name, MODE)

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

                    recallAtGT = all_metrics["RecallAtSizeofGroundTruth"]

                    print(
                        method_name,
                        " with MRR Score: ",
                        mrr_score,
                        " and RecallAtGT: ",
                        recallAtGT,
                    )

                    matches = matches.one_to_one()
                    one2one_metrics = matches.get_metrics(ground_truth)

                    source_file = source_file.split("/")[-1]
                    target_file = target_file.split("/")[-1]

                    result = [
                        BENCHMARK,
                        DATASET,
                        type,
                        source_file,
                        target_file,
                        ncols_src,
                        ncols_tgt,
                        nrows_src,
                        nrows_tgt,
                        nmatches,
                        method_name,
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

        print("Total tables: ", table_count)


def main():
    parser = argparse.ArgumentParser(description="Run the Valentine benchmark")

    parser.add_argument(
        "--benchmark",
        type=str,
        help="The benchmark to run. Default is Valentine",
        default="valentine",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to run the benchmark on. Default is OpenData",
        default="OpenData",
    )

    parser.add_argument(
        "--mode", type=str, help="serialization mode", default="header_values_default",
    )

    args = parser.parse_args()
    dataset_dict = {
        "opendata": "OpenData",
        "magellan": "Magellan",
        "chembl": "ChEMBL",
        "tpc": "TPC-DI",
        "wikidata": "Wikidata",
    }
    path_dict = {
        "OpenData": "data/valentine/OpenData/",
        "Magellan": "data/valentine/Magellan/",
        "ChEMBL": "data/valentine/ChEMBL/",
        "TPC-DI": "data/valentine/TPC-DI/",
        "Wikidata": "data/valentine/Wikidata/Musicians",
    }
    DATASET = dataset_dict[args.dataset]
    root = path_dict[DATASET]

    if DATASET in ["Wikidata", "Magellan"]:
        run_valentine_benchmark_one_level(args.benchmark, DATASET, root, args.mode)
    else:
        run_valentine_benchmark_three_levels(args.benchmark, DATASET, root, args.mode)


if __name__ == "__main__":
    # BENCHMARK = "valentine"

    # WIKIDATA musicians
    # run_valentine_benchmark_one_level()

    # Magellan
    # DATASET='Magellan'
    # ROOT='data/valentine/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # OpenData
    # run_valentine_benchmark_three_levels()

    # ChEMBLc
    # DATASET = "ChEMBL"
    # ROOT = "data/valentine/ChEMBL/"
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='data/valentine/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # Wikidata
    # DATASET='Wikidata'
    # ROOT='data/valentine/Wikidata/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    main()
