import os
import pandas as pd
import json


def get_gdc_triplets():
    ROOT = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/gdc'

    target_file = os.path.join(ROOT, 'target-tables', 'gdc_full.csv')
    studies_path = os.path.join(ROOT, 'source-tables')
    gt_path = os.path.join(ROOT, 'ground-truth')

    triplets = []

    for gt_file in os.listdir(gt_path):
        if gt_file.endswith('.csv'):

            # if gt_file != 'Krug.csv':
            #     continue

            source_file = os.path.join(studies_path, gt_file)
            triplet = (source_file, target_file,
                       os.path.join(gt_path, gt_file))
            triplets.append(triplet)

    return triplets


def get_valentine_benchmark_three_levels(DATASET='OpenData'):

    ROOT = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema Matching Benchmarks/Valentine-datasets/'
    ROOT = os.path.join(ROOT, DATASET)

    triplets = []

    for type in os.listdir(ROOT):
        if type == '.DS_Store':
            continue

        # print("Type: ", type)
        for table_folder in os.listdir(os.path.join(ROOT, type)):

            if table_folder == '.DS_Store':
                continue

            source_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_source.csv')
            target_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_target.csv')
            mapping_file = os.path.join(
                ROOT, type, table_folder, table_folder+'_mapping.json')

            triplet = (source_file, target_file, mapping_file)
            triplets.append(triplet)

    return triplets


def get_triplets(dataset='GDC'):
    if dataset == 'GDC':
        return get_gdc_triplets()
    if dataset == 'OpenData':
        return get_valentine_benchmark_three_levels('OpenData')


def get_gt(gt):
        
    if gt.endswith('.csv'):
        gt_df = pd.read_csv(gt)
        gt_df.dropna(inplace=True)
        ground_truth = list(gt_df.itertuples(index=False, name=None))
    elif gt.endswith('.json'):

        data = json.loads(open(gt).read())
        matchings = [(match['source_column'], match['target_column'])
                        for match in data['matches']]
        ground_truth = [(match['source_column'], match['target_column'])
                        for match in data['matches']]
            
    return ground_truth
    

# if __name__ == '__main__':

#     # triplets = get_triplets('GDC')
#     triplets = get_triplets('OpenData')

#     # main loop for experiments with diferent datasets
#     for source, target, gt in triplets:

#         df_source = pd.read_csv(source)
#         df_target = pd.read_csv(target)

#         if gt.endswith('.csv'):
#             gt_df = pd.read_csv(gt)
#             gt_df.dropna(inplace=True)
#             ground_truth = list(gt_df.itertuples(index=False, name=None))
#         elif gt.endswith('.json'):

#             data = json.loads(open(gt).read())
#             matchings = [(match['source_column'], match['target_column'])
#                          for match in data['matches']]
#             ground_truth = [(match['source_column'], match['target_column'])
#                             for match in data['matches']]

#         print(df_source.columns)
#         print(df_target.columns)
#         print(ground_truth)
#         break
