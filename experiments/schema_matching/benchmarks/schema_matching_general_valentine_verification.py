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


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import numpy as np


project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_path))


from experiments.schema_matching.benchmarks.utils import compute_mean_ranking_reciprocal, compute_mean_ranking_reciprocal_detail, create_result_file, record_result

import algorithms.schema_matching.topk.harmonizer.match_reranker as mr

import pprint
pp = pprint.PrettyPrinter(indent=4)




import numpy as np
import pandas as pd



from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose different models as well

def lsh_cosine_similarity(source, target, threshold=0.5):
    # Convert columns to unique values and drop NaNs
    source_values = source.dropna().astype(str).unique()
    target_values = target.dropna().astype(str).unique()

    # Check if either source or target is empty to avoid errors
    if len(source_values) == 0 or len(target_values) == 0:
        return 0.0, None

    # Encode the source and target values using Sentence-BERT
    source_embeddings = model.encode(source_values, convert_to_tensor=True)
    target_embeddings = model.encode(target_values, convert_to_tensor=True)

    # Move tensors to CPU if they are on a GPU
    source_embeddings = source_embeddings.cpu()
    target_embeddings = target_embeddings.cpu()

    # Compute cosine similarity between the source and target embeddings
    cosine_scores = cosine_similarity(source_embeddings, target_embeddings)

    # Calculate mean similarity, optionally apply a threshold
    mean_similarity = np.mean(cosine_scores > threshold)  # Proportion of pairs above the threshold
    return mean_similarity


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings





def run_valentine_benchmark_one_level(BENCHMARK='valentine', DATASET='musicians', ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/Wikidata/Musicians'):
    '''
    Run the valentine benchmark for one level of the dataset (Magelan and Wikidata)
    '''

    HEADER = ['benchmark', 'dataset', 'source_table', 'target_table','ncols_src','ncols_tgt','nrows_src','nrows_tgt','nmatches','method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', BENCHMARK, DATASET)
    result_file = os.path.join(results_dir,  BENCHMARK + '_' + DATASET + '_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)

    for folder in os.listdir(ROOT):
        if folder == '.DS_Store':
            continue

        source_file = os.path.join(ROOT, folder, folder+'_source.csv')
        target_file = os.path.join(ROOT, folder, folder+'_target.csv')
        mapping_file = os.path.join(ROOT, folder, folder+'_mapping.json')

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

      


def run_valentine_benchmark_three_levels(BENCHMARK='valentine', DATASET='OpenData', ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/OpenData/'):
    '''
    Run the valentine benchmark for datasets split on Unionable, View-Unionable, Joinable, Semantically-Joinable
    '''

    HEADER = ['benchmark', 'dataset', 'type', 'source_table', 'target_table', 'ncols_src','ncols_tgt','nrows_src','nrows_tgt', 'nmatches','method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', BENCHMARK, DATASET)
    result_file = os.path.join(results_dir,  BENCHMARK + '_' + DATASET + '_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)

    for type in os.listdir(ROOT):
        if type == '.DS_Store':
            continue

        print("Type: ", type)
        for table_folder in os.listdir(os.path.join(ROOT, type)):

            if table_folder == '.DS_Store':
                continue

            # print("Table: ", table_folder)  

            source_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_source.csv')
            target_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_target.csv')
            mapping_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_mapping.json')

            ground_truth = extract_matchings(open(mapping_file).read())

            # if len(ground_truth) < 2:
            #     continue

            df_source = pd.read_csv(source_file)
            df_target = pd.read_csv(target_file)

         
            # print("GroundTruth")
            # for gt in ground_truth:
            #     print(gt)
            # print("\n")

            # print(ground_truth)

            ncols_src = str(df_source.shape[1])
            ncols_tgt = str(df_target.shape[1])
            nrows_src = str(df_source.shape[0])
            nrows_tgt = str(df_target.shape[0])
            nmatches = len(ground_truth)

            if len(ground_truth) == 0:
                continue

            print("Aqui")
            for source_col, target_col in ground_truth:    
                
                similarity = lsh_cosine_similarity(df_source[source_col], df_target[target_col])
                print('Source:', source_col, 'Target:', target_col, ' Similarity:', similarity)
            print("\n\n")


        


if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    run_valentine_benchmark_one_level()

    # Magellan
    # DATASET='Magellan'
    # ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # OpenData
    # run_valentine_benchmark_three_levels()

    # ChEMBLc
    DATASET='ChEMBL'
    ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/ChEMBL/'
    run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
