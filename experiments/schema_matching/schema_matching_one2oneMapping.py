from valentine import valentine_match
import datetime
import time
import csv
import pandas as pd
import os
import sys

project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_path))


import algorithms.schema_matching.cl.cl as cl
import algorithms.schema_matching.ccsm.ccsm as ccsm






def get_gdc_matchers():
    matchers = {}


    # matchers['ContrastiveLearning'] = cl.CLMatcher(
    #     model_name='cl-reducer-v0.1', top_k=1)

    matchers['CCSM'] = ccsm.CombinedColumnSimilarityMatcher()
    
    


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

        experiment_name = 'schema_matching_gdc_one2oneMapping'

        data_dir = os.path.join(curr_dir,  'data', 'gdc')

        result_folder = os.path.join(data_dir, 'results')
        result_file = os.path.join(result_folder, experiment_name + '_results_' +
                                   datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

        matchers = get_gdc_matchers()

        gdc_df_path = os.path.join(data_dir, 'target-tables',  'gdc_table.csv')

        source_dir = os.path.join(data_dir, 'source-tables')
        gt_dir = os.path.join(data_dir, 'ground-truth')

        for filename in os.listdir(gt_dir):

            if filename != 'Krug.csv':
                continue


            if filename == '.DS_Store':
                continue


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

        header = ['Matcher', 'Filenames', 'GTruthSize']
        header.append(f'PrecisionAtGTSize')
        header.append('Runtime (s)')
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

            metrics = matches.get_metrics(ground_truth)
    
            print(metrics)
            

            # result = [matcher_name, source_file_name +
            #           '_to_' + target_file_name, len(ground_truth)]
            


def main():
    
    config_experiment_for_dataset('gdc')
    
    evaluate_matchers()


if __name__ == '__main__':
    main()
