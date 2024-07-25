import os
import sys
import pandas as pd
import datetime
import time
import csv
import warnings


from valentine import valentine_match
project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_path))

from algorithms.schema_matching.topk.topk_metrics import RecallAtTopK
import algorithms.schema_matching.topk.ccsm.ccsm as ccsm
import algorithms.schema_matching.topk.cl.cl as cl
import algorithms.schema_matching.topk.era.era as era

TOP_K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
# TOP_K_LIST = [1, 2, 3, 10]
TOP_K = TOP_K_LIST[-1]

def infer_column_type(series):

    
    
    
    # Drop NaNs for type checking
    non_null_series = series.dropna()

    # Suppress warnings during type checks
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Check if the series can be converted to numeric
        try:
            pd.to_numeric(non_null_series)
            return 'numeric'
        except ValueError:
            pass

        # Check if the series can be converted to datetime
        try:
            pd.to_datetime(non_null_series)
            return 'date'
        except ValueError:
            pass

    # If neither, return 'string'
    return 'string'

def get_gdc_matchers():
    matchers = {}

    matchers['ContrastiveLearning'] = cl.CLMatcher(
        model_name='bdi-cl-v0.2', top_k=TOP_K)

    # matchers['CCSM'] = ccsm.CombinedColumnSimilarityMatcher(top_k=TOP_K)

    matchers['MPNetEmbedRetrieveAlign'] = era.EmbedRetrieveAlign(model_name='all-mpnet-base-v2', top_k=TOP_K)

    curr_dir = os.getcwd()
    gdc_model_path = os.path.join(curr_dir,  'model', 'fine_gdc')
    matchers['FineTunedEmbedRetrieveAlign'] = era.EmbedRetrieveAlign(model_name=gdc_model_path, top_k=TOP_K)

    return matchers


def config_experiment_for_dataset(dataset):
    global dataset_name, experiment_name, data_dir, result_folder, result_file, data_paths_list, matchers
    dataset_name = dataset
    curr_dir = os.getcwd()

    # for new datasets, make sure to populate the data_paths_list with triplets
    # (paths to GTruth, path to source table , path to target table)
    # see the folder for examples
    data_paths_list = []

    if dataset_name == 'gdc':

        experiment_name = 'schema_matching_precolposition_gdc'

        data_dir = os.path.join(curr_dir,  'data', 'gdc')

        result_folder = os.path.join(data_dir, 'results')
        result_file = os.path.join(result_folder, experiment_name + '_results_' +
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

        matchers = get_gdc_matchers()

        gdc_df_path = os.path.join(data_dir, 'target-tables',  'gdc_table.csv')

        source_dir = os.path.join(data_dir, 'source-tables')
        gt_dir = os.path.join(data_dir, 'ground-truth')

        for filename in os.listdir(gt_dir):
            if filename == '.DS_Store':
                continue

            # if filename != 'Krug.csv':
            #     continue

            gt_file_path = os.path.join(gt_dir, filename)
            if os.path.isfile(gt_file_path):
                source_file_path = os.path.join(source_dir, filename)
                eval_entry = (gt_file_path, source_file_path, gdc_df_path)
                data_paths_list.append(eval_entry)

    else:
        print(f"Dataset {dataset} not found")


def create_result_file():
    if not os.path.exists(result_folder):
        os.makedirs(result_file)
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)

        header = ['Matcher', 'Source File', 'Target File', 'Source Column', 'Target Column', 'Data Type','TopK',  'Target Column Index']
        writer.writerow(header)
        print(f"Result file created at {result_file}")


def record_result(result):
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)


def evaluate_matchers():
    print(f"Evaluating matchers on {dataset_name} dataset")

    create_result_file()
    for data_paths in data_paths_list:
        gt_file_path, source_file_path, target_file_path = data_paths
        gt_df = pd.read_csv(gt_file_path)
        ground_truth = list(gt_df.itertuples(index=False, name=None))
        source_df = pd.read_csv(source_file_path)
        target_df = pd.read_csv(target_file_path)

        source_file_name = os.path.basename(source_file_path)
        target_file_name = os.path.basename(target_file_path)

        for matcher_name, matcher in matchers.items():

            print("Using ", matcher_name, ' to match ',
                  source_file_name, ' to ',  target_file_name)

            start_time = time.time()
            matches = valentine_match(source_df, target_df, matcher)
            end_time = time.time()
            runtime = end_time - start_time

            #print(matches)
            for source_col, target_col in ground_truth:
                # print(source_col, target_col)
                candidate_matches = [ (match[1][1],matches[match]) for match in matches if match[0][1] == source_col]
                candidate_matches.sort(key=lambda x: x[1], reverse=True)

                target_col_index = next((index for index, match in enumerate(candidate_matches) if match[0] == target_col), -1)

                # print(candidate_matches)
                # print(target_col_index)
                # print('\n')

                data_type = infer_column_type(source_df[source_col])

                
                result_line = [matcher_name, source_file_name, target_file_name, source_col, target_col, data_type, TOP_K, target_col_index]

                record_result(result_line)
                
                


def main():

    config_experiment_for_dataset('gdc')

    evaluate_matchers()


if __name__ == '__main__':
    main()
