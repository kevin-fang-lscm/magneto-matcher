import os
import sys
import pandas as pd
import time
import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import numpy as np


project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(os.path.join(project_path))



import algorithms.schema_matching.topk.indexed_similarity.indexed_similarity as indexed_similarity
import algorithms.schema_matching.topk.cl.cl as cl
import algorithms.schema_matching.topk.harmonizer.harmonizer as hm
import algorithms.schema_matching.topk.harmonizer.match_reranker as mr

import algorithms.schema_matching.topk.retrieve_match.retrieve_match as rema
import algorithms.schema_matching.topk.retrieve_match.retrieve_match_simpler as rema_simpler
import algorithms.schema_matching.topk.match_maker.match_maker as mm

import algorithms.schema_matching.topk.retrieve_match.retrieve_match as rema
from experiments.schema_matching.benchmarks.utils import compute_mean_ranking_reciprocal, compute_mean_ranking_reciprocal_detail, create_result_file, record_result,extract_matchings

# from algorithms.schema_matching.topk.indexed_similarity import IndexedSimilarityMatcher


from valentine import valentine_match
from valentine.algorithms import Coma
import pprint
import sys

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)



# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Or any other SentenceTransformer model

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


def normalize_binary(column):
    """Convert binary columns to a standard 1/0 format."""
    binary_map = {
        "yes": 1, "no": 0,
        "present": 1, "absent": 0,
        "y": 1, "n": 0,
        "1": 1, "0": 0,
        True: 1, False: 0
    }
    return column.map(lambda x: binary_map.get(str(x).lower(), x))

def preprocess_text(column):
    """Lowercase and remove extra spaces. Expand acronyms if necessary."""
    column = column.str.lower().str.strip()
    acronym_map = {
        "dept": "department",
        "mgr": "manager",
        # Add other known acronyms here if needed
    }
    column = column.replace(acronym_map, regex=True)
    return column

def fuzzy_column_similarity(source, target, threshold=80):
    """Calculate fuzzy similarity by checking close matches between unique values."""
    source_values = source.dropna().unique()
    target_values = target.dropna().unique()
    
    matches = 0
    for s in source_values:
        for t in target_values:
            if fuzz.ratio(s, t) >= threshold:
                matches += 1
                break  # Stop at the first match

    return matches / max(len(source_values), len(target_values))

def generate_embeddings(column):
    """Generate embeddings for each unique value in a column."""
    unique_values = column.dropna().astype(str).unique()
    embeddings = model.encode(unique_values, convert_to_tensor=True)
    return dict(zip(unique_values, embeddings))

def embedding_similarity(source, target):
    """Calculate embedding-based similarity using cosine similarity."""
    source_embeddings = generate_embeddings(source)
    target_embeddings = generate_embeddings(target)
    
    similarities = []
    for source_val, source_emb in source_embeddings.items():
        for target_val, target_emb in target_embeddings.items():
            # Move embeddings to CPU before converting to numpy
            source_emb_cpu = source_emb.cpu().numpy().reshape(1, -1)
            target_emb_cpu = target_emb.cpu().numpy().reshape(1, -1)
            similarity = cosine_similarity(source_emb_cpu, target_emb_cpu)[0][0]
            similarities.append(similarity)
    
    return np.mean(similarities)

def content_similarity(source, target):
    # Step 1: Normalize binary values
    source = normalize_binary(source)
    target = normalize_binary(target)

    # Step 2: Preprocess text values
    source = preprocess_text(source.astype(str))
    target = preprocess_text(target.astype(str))

    # # Step 3: Calculate Jaccard similarity for exact matches
    # source_values = set(source.dropna().unique())
    # target_values = set(target.dropna().unique())
    # jaccard_similarity = len(source_values.intersection(target_values)) / len(source_values.union(target_values))

    # Step 4: Calculate fuzzy similarity for approximate matches
    fuzzy_similarity = fuzzy_column_similarity(source, target)

    # Step 5: Calculate embedding similarity for semantic similarity
    embedding_based_similarity = embedding_similarity(source, target)

    # Step 6: Combine all similarity scores (adjust weights as needed)
    combined_similarity =  (0.4 * fuzzy_similarity) + (0.6 * embedding_based_similarity)
    return combined_similarity

def get_matcher(method):
    if method == 'Coma':
        return Coma()
    elif method == 'ComaInst':
        return Coma(use_instances=True, java_xmx="10096m")
    elif method == 'IndexedSimilarity':
        return indexed_similarity.IndexedSimilarityMatcher()
    elif method == 'Rema':
        return rema.RetrieveMatch('arctic',None, 'header_values_default', 'gpt-4o-mini')
    elif method == 'Harmonizer':
        return hm.Harmonizer()
    elif method == 'HarmonizerFine':
        #return hm.Harmonizer('./my_fine_tuned_model')
        #return hm.Harmonizer('sentence-transformers/all-mpnet-base-v2')
        return hm.Harmonizer('sentence-transformers/all-mpnet-base-v2')
    elif method == 'Rema':
        return rema.RetrieveMatch('arctic',None, 'header_values_default', 'gpt-4o-mini')
    elif method == 'RemaSimpler':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini')
    elif method == 'RemaBipartite':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini', True)
    elif method == 'Rema-BP+Basic':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini', True, True)
   
    elif method == 'MatchMaker':
        return mm.MatchMaker()
    elif method == 'MatchMakerInstance':
        return mm.MatchMaker(use_instances=True)
    
    
    elif method == 'HarmonizerInstance':
        return hm.Harmonizer(use_instances=True)
    
    elif method == 'HarmonizerMatcher':
        return mr.MatchReranker(use_instances=False)

    
    elif method == 'IndexedSimilarityInst':
        return indexed_similarity.IndexedSimilarityMatcher(use_instances=True)
    elif method == 'CL':
        return cl.CLMatcher()





def run_gdc_studies(BENCHMARK='gdc_studies', DATASET='gdc_studies', ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/gdc'):


    HEADER = ['benchmark', 'dataset', 'source_table', 'target_table','ncols_src','ncols_tgt','nrows_src','nrows_tgt','nmatches','method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', BENCHMARK, DATASET)
    result_file = os.path.join(results_dir,  BENCHMARK + '_' + DATASET + '_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)

    # target_file = os.path.join(ROOT, 'target-tables', 'gdc_table.csv')
    # target_file = os.path.join(ROOT, 'target-tables', 'gdc_full.csv'
    # target_file = os.path.join(ROOT, 'target-tables', 'gdc_all_columns_all_values.csv')
    target_file = os.path.join(ROOT, 'target-tables', 'gdc_unique_columns_concat_values.csv')
    
    df_target = pd.read_csv(target_file)

    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith('.csv'):

            # if gt_file != 'Krug.csv':
            #     continue

            source_file = os.path.join(studies_path, gt_file)
            df_source = pd.read_csv(source_file)

            gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
            gt_df.dropna(inplace=True)
            ground_truth = list(gt_df.itertuples(index=False, name=None))

            for source_col, target_col in ground_truth:
                
                similarity = lsh_cosine_similarity(df_source[source_col], df_target[target_col])
                print('Source:', source_col, 'Target:', target_col, ' Similarity:', similarity)
            print('\n')

            

          




if __name__ == '__main__':
    run_gdc_studies()
    

