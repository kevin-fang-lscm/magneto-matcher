import os
import sys
import pandas as pd
import time
import datetime


project_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.append(os.path.join(project_path))



import algorithms.schema_matching.topk.indexed_similarity.indexed_similarity as indexed_similarity
import algorithms.schema_matching.topk.indexed_similarity_new.indexed_similarity_new as indexed_similarity_new
import algorithms.schema_matching.topk.cl.cl as cl
from experiments.schema_matching.benchmarks.utils import compute_mean_ranking_reciprocal, compute_mean_ranking_reciprocal_detail, create_result_file, record_result,extract_matchings

# from algorithms.schema_matching.topk.indexed_similarity import IndexedSimilarityMatcher


from valentine import valentine_match
from valentine.algorithms import Coma
import pprint
import sys

pp = pprint.PrettyPrinter(indent=4, sort_dicts=True)











def get_matcher(method):
    if method == 'Coma':
        return Coma()
    elif method == 'ComaInst':
        return Coma(use_instances=True, java_xmx="10096m")
    elif method == 'IndexedSimilarity':
        return indexed_similarity.IndexedSimilarityMatcher()
    elif method == 'IndexedSimilarityNew':
        return indexed_similarity_new.IndexedSimilarityMatcherNew()
    elif method == 'IndexedSimilarityInst':
        return indexed_similarity.IndexedSimilarityMatcher(use_instances=True)
    elif method == 'CL':
        return cl.CLMatcher()



def run_dou_pair():
    print(project_path)
    gdc_input_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-discovery.csv')
    gdc_target_df = pd.read_csv(project_path+'/data/gdc_alt/Dou-ucec-confirmatory.csv')

    gt_df = pd.read_csv(project_path+'/data/gdc_alt/gt.csv')
    gt_df.dropna(inplace=True)
    ground_truth = list(gt_df.itertuples(index=False, name=None))

    # matchers = [ IndexedSimilarityMatcher(),  JaccardDistanceMatcher(), Coma()]
    matchers = [cl.CLMatcher(), Coma(
        use_instances=True, java_xmx="10096m")]

    # matchers = [isimnew.IndexedSimilarityMatcherNew(), isim.IndexedSimilarityMatcher(use_instances=True), Coma(), Coma(use_instances=True)]
    # matchers = [ isim.IndexedSimilarityMatcher(use_instances=True)]

    for matcher in matchers:
        print("Matcher: ", matcher)

        start_time = time.time()

        matches = valentine_match(gdc_input_df, gdc_target_df, matcher)
        metrics = matches.get_metrics(ground_truth)

        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime for valentine_match: {runtime:.4f} seconds")

        score = compute_mean_ranking_reciprocal(matches, ground_truth)
        print("Mean Ranking Reciprocal Score: ", score)

        print("Normal Metrics:")
        pp.pprint(metrics)

        matches = matches.one_to_one()
        metrics = matches.get_metrics(ground_truth)
        print("One2One Metrics:")
        pp.pprint(metrics)
        print("\n\n")

        ncols_src = str(gdc_input_df.shape[1])
        ncols_tgt = str(gdc_target_df.shape[1])
        nrows_src = str(gdc_input_df.shape[0])
        nrows_tgt = str(gdc_target_df.shape[0])
        nmatches = len(ground_truth)

        



def run_gdc_studies(BENCHMARK='gdc_studies', DATASET='gdc_studies', ROOT='/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/gdc'):




    HEADER = ['benchmark', 'dataset', 'source_table', 'target_table','ncols_src','ncols_tgt','nrows_src','nrows_tgt','nmatches','method', 'runtime', 'mrr',  'All_Precision', 'All_F1Score', 'All_Recall', 'All_PrecisionTop10Percent', 'All_RecallAtSizeofGroundTruth',
              'One2One_Precision', 'One2One_F1Score', 'One2One_Recall', 'One2One_PrecisionTop10Percent', 'One2One_RecallAtSizeofGroundTruth']

    results_dir = os.path.join(
        project_path,  'results', 'benchmarks', BENCHMARK, DATASET)
    result_file = os.path.join(results_dir,  BENCHMARK + '_' + DATASET + '_results_' +
                               datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.csv')

    create_result_file(results_dir, result_file, HEADER)

    target_file = os.path.join(ROOT, 'target-tables', 'gdc_table.csv')
    df_target = pd.read_csv(target_file)

    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith('.csv'):

            source_file = os.path.join(studies_path, gt_file)
            df_source = pd.read_csv(source_file)

            gt_df = pd.read_csv(os.path.join(gt_path, gt_file))
            gt_df.dropna(inplace=True)
            ground_truth = list(gt_df.itertuples(index=False, name=None))

            matchers = [  "CL"]
        

            for matcher in matchers:
                print(f"Matcher: {matcher}, Source: {source_file}, Target: {target_file}")

                method_name = matcher
                matcher = get_matcher(matcher)    

                start_time = time.time()

                matches = valentine_match(df_source, df_target, matcher)
                

                end_time = time.time()
                runtime = end_time - start_time
                print(f"Runtime for valentine_match: {runtime:.4f} seconds")

                mrr_score = compute_mean_ranking_reciprocal(matches, ground_truth)
                print("Mean Ranking Reciprocal Score: ", mrr_score)

                

                all_metrics = matches.get_metrics(ground_truth)

                matches = matches.one_to_one()
                one2one_metrics = matches.get_metrics(ground_truth)

                ncols_src = str(df_source.shape[1])
                ncols_tgt = str(df_target.shape[1])
                nrows_src = str(df_source.shape[0])
                nrows_tgt = str(df_target.shape[0])

                nmatches = len(ground_truth)

                result = [BENCHMARK, DATASET, 'gdc_table', gt_file, ncols_src, ncols_tgt, nrows_src, nrows_tgt,nmatches, method_name, runtime, mrr_score, all_metrics['Precision'], all_metrics['F1Score'], all_metrics['Recall'], all_metrics['PrecisionTop10Percent'], all_metrics['RecallAtSizeofGroundTruth'],
                      one2one_metrics['Precision'], one2one_metrics['F1Score'], one2one_metrics['Recall'], one2one_metrics['PrecisionTop10Percent'], one2one_metrics['RecallAtSizeofGroundTruth']]

                record_result(result_file, result)





if __name__ == '__main__':
    run_gdc_studies()
    # run_for_santos_benchmark()

    # run_dou_pair()
