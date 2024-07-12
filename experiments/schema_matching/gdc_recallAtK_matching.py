import os
import sys
import pandas as pd
import csv
import time
import datetime

from valentine import valentine_match

project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.era.table2text as table2text
import algorithms.schema_matching.era.era as era
import algorithms.schema_matching.cl.cl as cl
from algorithms.schema_matching.additional_metrics import RecallAtTopK

current_dir = os.getcwd()
GDC_DIR = os.path.join(current_dir,  'data', 'gdc')

GDC_GT_DIR = os.path.join(GDC_DIR, 'ground-truth')
GDC_DATA_DIR = os.path.join(GDC_DIR, 'source-tables')

RESULT_FOLDER = os.path.join(GDC_DIR, 'results')
EXPERIMENT_NAME = 'gdc_recallAtK_matching'
RESULT_FILE = os.path.join(
    RESULT_FOLDER, EXPERIMENT_NAME + '_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

# MODEL_PATH = os.path.join(current_dir,  'model', 'fine_tablegpt')
MODEL_PATH = os.path.join(current_dir,  'model', 'fine_gdc')

target_files = ['Dou.csv']

TOP_K = 20

complexTransformer = table2text.Dataset2Text(
    num_context_columns=0, num_context_rows=0, col_summary_impl=table2text.ComplexColumnSummary())

matchers = {
    'ContrastiveLearning' : cl.CLMatcher(model_name='cl-reducer-v0.1', top_k=TOP_K)
    # 'EmbedRetrieveAlignTop20_MP': era.EmbedRetrieveAlign(model_name='all-mpnet-base-v2', top_k=TOP_K),
    # 'ComplexEmbedRetrieveAlignTop20_MP': era.EmbedRetrieveAlign(model_name='all-mpnet-base-v2', top_k=TOP_K, column_transformer=complexTransformer),
    # 'NewModelComplexEmbedRetrieveAlignTop20_MP': era.EmbedRetrieveAlign(model_name=MODEL_PATH, column_transformer=complexTransformer, top_k=TOP_K)
    
    # 'EmbedRetrieveAlignTop20_Mini': era.EmbedRetrieveAlign(model_name='all-MiniLM-L12-v2', top_k=TOP_K)
}


def record_result(result):
    with open(RESULT_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)


def create_result_file():
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    with open(RESULT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Matcher', 'Filename', 'GTruthSize', 'RecallAtK1', 'RecallAtK5',
                        'RecallAtK10', 'RecallAtK20',  'Runtime (s)'])


def evaluate_matchers_on_gdc():
    print("Evaluating matchers on GDC dataset")

    create_result_file()

    # gdc_table = pd.read_csv(os.path.join(
    #     GDC_DIR, 'target-tables',  'gdc_table.csv'))
    gdc_table = pd.read_csv(os.path.join(
        GDC_DIR, 'target-tables',  'gdc_table.csv'))

    for filename in os.listdir(GDC_GT_DIR):

        # if filename not in target_files:
        #     continue

        if os.path.isfile(os.path.join(GDC_GT_DIR, filename)):
            gt_df = pd.read_csv(os.path.join(GDC_GT_DIR, filename))
            ground_truth = list(gt_df.itertuples(index=False, name=None))

            df_input = pd.read_csv(os.path.join(GDC_DATA_DIR, filename))

            for matcher_name, matcher in matchers.items():
                print("Running ", matcher_name, ' on ', filename)

                start_time = time.time()
                matches = valentine_match(df_input, gdc_table, matcher)
                end_time = time.time()
                runtime = end_time - start_time

                # # Calculate RecallAtK for k=1, k=5, k=10, k=20
                # recall_at_k1 = RecallAtTopK(1).apply(matches, ground_truth)
                # recall_at_k5 = RecallAtTopK(5).apply(matches, ground_truth)
                # recall_at_k10 = RecallAtTopK(10).apply(matches, ground_truth)
                recall_at_k20 = RecallAtTopK(20).apply(matches, ground_truth)

                # # Record results
                # result = [
                #     matcher_name,
                #     filename,
                #     len(ground_truth),
                #     recall_at_k1,
                #     recall_at_k5,
                #     recall_at_k10,
                #     recall_at_k20,
                #     runtime
                # ]
                # record_result(result)
                # print(
                #     f"Recall at K1: {recall_at_k1}, K5: {recall_at_k5}, K10: {recall_at_k10}, K20: {recall_at_k20}\n")


def main():
    evaluate_matchers_on_gdc()


if __name__ == '__main__':
    main()
