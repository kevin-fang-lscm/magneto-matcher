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


project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(project_path))

import algorithms.schema_matching.topk.indexed_similarity.indexed_similarity as indexed_similarity
import algorithms.schema_matching.topk.retrieve_match.retrieve_match as rema
import algorithms.schema_matching.topk.retrieve_match.retrieve_match_simpler as rema_simpler
import algorithms.schema_matching.topk.cl.cl as cl
import algorithms.schema_matching.topk.harmonizer.harmonizer as hm
import algorithms.schema_matching.topk.match_maker.match_maker as mm
from experiments.schema_matching.benchmarks.utils import compute_mean_ranking_reciprocal, compute_mean_ranking_reciprocal_detail, create_result_file, record_result

import algorithms.schema_matching.topk.harmonizer.match_reranker as mr

import pprint
pp = pprint.PrettyPrinter(indent=4)

def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [(match['source_column'], match['target_column'])
                 for match in data['matches']]
    return matchings


def get_matcher(method):
    if method == 'Coma':
        return Coma()
    elif method == 'ComaInst':
        return Coma(use_instances=True, java_xmx="10096m")
    elif method == 'Harmonizer':
        return hm.Harmonizer()
    elif method == 'HarmonizerInstance':
        return hm.Harmonizer(use_instances=True)
    elif method == 'IndexedSimilarityInst':
        return indexed_similarity.IndexedSimilarityMatcher(use_instances=True)
    elif method == 'Rema':
        return rema.RetrieveMatch('arctic',None, 'header_values_default', 'gpt-4o-mini')
    elif method == 'RemaSimpler':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini')
    elif method == 'RemaBipartite':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini', True)
    elif method == 'Rema-BP+Basic':
        return rema_simpler.RetrieveMatchSimpler('arctic',None, 'header_values_default', 'gpt-4o-mini', True, True)
    elif method == 'CL':
        return cl.CLMatcher()
    elif method == 'HarmonizerMatcher':
        return mr.MatchReranker(use_instances=True)
    elif method == 'MatchMaker':
        return mm.MatchMaker()
    elif method == 'MatchMakerInstance':
        return mm.MatchMaker(use_instances=True)
    elif method == 'MatchMakerGPT':
        return mm.MatchMaker(use_instances=False, use_gpt=True)


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

        # matchers = [ "IndexedSimilarityNew", "Coma"]
        # matchers = [ "Harmonizer", "HarmonizerInstance", "Coma", "ComaInst", "CL"]
        # matchers = [ "HarmonizerInstance", "HarmonizerMatcher", "RemaSimpler", "RemaBipartite", "Rema"]
        # matchers = [ "MatchMaker", "MatchMakerInstance"]
        matchers = [ "MatchMaker",  "MatchMakerGPT", "MatchMakerInstance"]
        

        for matcher in matchers:

            # print("Matcher: ", matcher)

            method_name = matcher
            matcher = get_matcher(matcher)    

            start_time = time.time()

            # try:
            #     matches = valentine_match(df_source, df_target, matcher)
            # except Exception as e:
            #     print(f"Not able to run the matcher because of exception: {e}")
            #     matches = matcher_results.MatcherResults({})
            matches = valentine_match(df_source, df_target, matcher)

            end_time = time.time()
            runtime = end_time - start_time
            # print(f"Runtime for valentine_match: {runtime:.4f} seconds")

            # mrr_score = compute_mean_ranking_reciprocal(
            #     matches, ground_truth)
            # detail = f"Type: {type}, Table: {folder}, Matcher: {method_name}"
            # mrr_score = compute_mean_ranking_reciprocal_detail(matches, ground_truth, detail)
            mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)

            

            all_metrics = matches.get_metrics(ground_truth)

            recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

            print(method_name, " with MRR Score: ", mrr_score, " and RecallAtGT: ", recallAtGT)

            matches = matches.one_to_one()
            one2one_metrics = matches.get_metrics(ground_truth)

            source_file = source_file.split('/')[-1]
            target_file = target_file.split('/')[-1]

            result = [BENCHMARK, DATASET, source_file, target_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt,nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                      one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

            record_result(result_file, result)


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


            # matchers = [ "Rema"]#, "HarmonizerInstance", "Coma", "ComaInst", "CL"]
            # matchers = ["HarmonizerInstance", "HarmonizerMatcher"] #"HarmonizerInstance", "Rema"] # , "HarmonizerMatcher"]
            #matchers = [ "HarmonizerInstance", "HarmonizerMatcher", "RemaSimpler", "RemaBipartite", "Rema"]
            # matchers = [  "RemaSimpler", "RemaBipartite", "Rema-BP+Basic"]
            # matchers = [ "RemaBipartite",  "Rema-BP+Basic"]
            # matchers = [  "MatchMaker", "RemaSimpler", "Rema-BP+Basic",  "RemaBipartite"]
            # matchers = [ "MatchMaker",  "MatchMakerInstance"]
            matchers = [ "MatchMaker",  "MatchMakerGPT"]
            # matchers = [ "MatchMakerGPT"]

            


            for matcher in matchers:
                # print("Matcher: ", matcher)

                method_name = matcher
                matcher = get_matcher(matcher)    

                start_time = time.time()

                try:
                    matches = valentine_match(df_source, df_target, matcher)
                except Exception as e:
                    print(
                        f"Not able to run the matcher because of exception: {e}")
                    matches = matcher_results.MatcherResults({})
                # matches = valentine_match(df_source, df_target, matcher)

                # print("Matches: ", matches)
                

                end_time = time.time()
                runtime = end_time - start_time
                # print(f"Runtime for valentine_match: {runtime:.4f} seconds")

                # mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
                detail = f"Type: {type}, Table: {table_folder}, Matcher: {method_name}"
                mrr_score = compute_mean_ranking_reciprocal_detail(matches, ground_truth, detail)

                

                all_metrics = matches.get_metrics(ground_truth)

                recallAtGT = all_metrics['RecallAtSizeofGroundTruth']

                print(method_name, " with MRR Score: ", mrr_score, " and RecallAtGT: ", recallAtGT)


                # if mrr_score < 0.5 or recallAtGT < 0.5:
                #     print("GroundTruth: ")
                #     for gt in ground_truth:
                #         print(gt)    
                #     print("Matches\n")
                #     for i, match in enumerate(matches):
                #         if i >= len(ground_truth):
                #             break
                #         print(match, matches[match])



                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                source_file = source_file.split('/')[-1]
                target_file = target_file.split('/')[-1]

                result = [BENCHMARK, DATASET, type, source_file, target_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt, nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                          one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)
            print("\n")

            # break
            
        # break  x 
    # break
    


if __name__ == '__main__':
    BENCHMARK = 'valentine'

    # WIKIDATA musicians
    # run_valentine_benchmark_one_level()

    # Magellan
    # DATASET='Magellan'
    # ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/Magellan'
    # run_valentine_benchmark_one_level(BENCHMARK, DATASET, ROOT)

    # OpenData
    run_valentine_benchmark_three_levels()

    # ChEMBLc
    # DATASET='ChEMBL'
    # ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/ChEMBL/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)

    # TPC-DI
    # DATASET='TPC-DI'
    # ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/TPC-DI/'
    # run_valentine_benchmark_three_levels(BENCHMARK, DATASET, ROOT)
