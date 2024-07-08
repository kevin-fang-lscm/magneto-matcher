import os
import pandas as pd
from valentine.metrics import F1Score, PrecisionTopNPercent
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher, DistributionBased, Coma, Cupid, SimilarityFlooding
from valentine.algorithms import schema_only_algorithms, instance_only_algorithms, schema_instance_algorithms
import pprint
import csv
import time
import datetime
pp = pprint.PrettyPrinter(indent=4, sort_dicts=False)

current_dir = os.getcwd()
GDC_DIR = os.path.join(current_dir,  'data', 'gdc')

GDC_GT_DIR = os.path.join(GDC_DIR, 'ground-truth')
GDC_DATA_DIR = os.path.join(GDC_DIR, 'source-tables')

RESULT_FOLDER = os.path.join(GDC_DIR, 'results')
RESULT_FILE = os.path.join(
    RESULT_FOLDER, 'macher_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

target_files = ['Clark.csv']

matchers = {
    # 'Jaccard' : JaccardDistanceMatcher(),
    # 'DistributionBased': DistributionBased(),
    'Coma': Coma(),
    # 'ComaInstances': Coma(use_instances=True),
    # 'Cupid': Cupid(),
    # 'SimilarityFlooding': SimilarityFlooding(),
}


def get_matcher_type(matcher):
    if matcher in schema_only_algorithms:
        return 'Schema'
    elif matcher in instance_only_algorithms:
        return 'Instance'
    elif matcher in schema_instance_algorithms:
        return 'SchemaInstance'
    else:
        return 'Unknown'


def record_result(result):
    with open(RESULT_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)


def create_result_file():

    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)

    with open(RESULT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Matcher', 'MatcherType',  'Filename', 'GTSize', 'Precision', 'F1Score', 'Recall',
                        'RecallAtSizeOfGroundtruth', 'PrecisionTop10Percent', 'Runtime (s)'])


def evaluate_matchers_on_gdc():

    print("Evaluating matchers on GDC dataset")

    create_result_file()

    gdc_table = pd.read_csv(os.path.join(
        GDC_DIR, 'target-tables',  'gdc_table.csv'))

    for filename in os.listdir(GDC_GT_DIR):

        if filename not in target_files:
            continue

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

                metrics = matches.get_metrics(ground_truth)
                result = [
                    matcher_name,
                    get_matcher_type(type(matcher).__name__),
                    filename,
                    len(ground_truth),
                    metrics['Precision'],
                    metrics['F1Score'],
                    metrics['Recall'],
                    metrics['RecallAtSizeofGroundTruth'],
                    metrics['PrecisionTop10Percent'],
                    runtime
                ]

                record_result(result)


def main():
    evaluate_matchers_on_gdc()


if __name__ == '__main__':
    main()
