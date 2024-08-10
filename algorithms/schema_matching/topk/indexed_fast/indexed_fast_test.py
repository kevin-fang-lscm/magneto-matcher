import os
import pandas as pd


from indexed_fast import IndexedSimilarityMatcher


def get_gdc_dataframes(file):
    # gdc_cat_path = './data/gdc/gdc_cat_ingt.csv'
    # gdc_cat_path = './data/gdc/gdc_cat.csv'
    gdc_cat_path = './data/gdc/gdc_num_cat.csv'

    df_gdc = pd.read_csv(gdc_cat_path, encoding='utf-8', engine='python')

    df_input = pd.read_csv(f'./data/gdc/source-tables/{file}')

    gt_df = pd.read_csv(
        f'./data/gdc/ground-truth/{file}', encoding='utf-8', engine='python')
    ground_truth = list(gt_df.itertuples(index=False, name=None))

    gt_gdc_cols = set([col[1] for col in ground_truth])
    gt_gdc_cols = set(gt_gdc_cols).intersection(df_gdc.columns)
    gt_gdc_cols_in_input_df = [col_pair[0]
                               for col_pair in ground_truth if col_pair[1] in gt_gdc_cols]

    df_input = df_input[gt_gdc_cols_in_input_df]

    return df_input, df_gdc, ground_truth


def run_for_gdc():

    files = ['Dou.csv', 'Krug.csv', 'Clark.csv',  'Vasaikar.csv',
             'Wang.csv', 'Satpathy.csv', 'Cao.csv', 'Huang.csv', 'Gilette.csv']
    # file = files[-1]

    files = ['Dou.csv']

    # positions_global = []
    results_global = []
    for file in files:
        print("File: ", file)

        df_input, df_gdc, ground_truth = get_gdc_dataframes(file)

        print(df_input.shape)
        print(df_gdc.shape)

        matcher = IndexedSimilarityMatcher()
        results = matcher.get_matches(df_input, df_gdc, ground_truth)


def run_for_santos_benchmark():

    root = '/Users/pena/Library/CloudStorage/GoogleDrive-em5487@nyu.edu/My Drive/NYU - GDrive/arpah/Schema matching data/santos_benchmark/'

    interested = '311_calls_historic_data_a.csv'

    for file in os.listdir(root + "query"):
        if file != interested:
            continue

        df_input = pd.read_csv(root + "query/" + file)
        print("Matching \n")
        print(df_input.head())

        if file.endswith("_a.csv"):
            table_prefix = file.replace("_a.csv", "")
        elif file.endswith("_b.csv"):
            table_prefix = file.replace("_b.csv", "")
        else:
            continue

        target_files = [f for f in os.listdir(
            root + "datalake") if f.startswith(table_prefix)]

        for target_file in target_files:
            df_target = pd.read_csv(root + "datalake/" + target_file)
            print("To match with \n")
            print(target_file)

            ground_truth = []
            for col in df_input.columns:
                if col in df_target.columns:
                    ground_truth.append((col, col))

            matcher = IndexedSimilarityMatcher()
            results = matcher.get_matches(df_input, df_target, ground_truth)

            break


def run_for_gdc_alt():
    gdc_input_df = pd.read_csv('./data/gdc_alt/Dou-ucec-discovery.csv')
    gdc_target_df = pd.read_csv('./data/gdc_alt/Dou-ucec-confirmatory.csv')

    gt_df = pd.read_csv('./data/gdc_alt/gt.csv')
    gt_df.dropna(inplace=True)

    print(gdc_input_df.head())
    ground_truth = list(gt_df.itertuples(index=False, name=None))

    gt = {}
    for gt_pair in ground_truth:
        gt[gt_pair[0]] = gt_pair[1]

    # print(ground_truth)

    # print(gdc_input_df.head())

    # print(gdc_target_df.head())

    matcher = IndexedSimilarityMatcher()

    results = matcher.get_matches(gdc_input_df, gdc_target_df, gt)


if __name__ == '__main__':
    # run_for_gdc()
    # run_for_santos_benchmark()

    run_for_gdc_alt()
