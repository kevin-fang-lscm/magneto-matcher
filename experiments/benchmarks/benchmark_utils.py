import os
import csv
import json


def sort_matches(matches):

    sorted_matches = {entry[0][1]: [] for entry in matches}
    for entry in matches:
        sorted_matches[entry[0][1]].append((entry[1][1], matches[entry]))

    # for key in sorted_matches:
    #     print(key, ' ', sorted_matches[key])
    return sorted_matches


def extract_matchings(json_data):

    data = json.loads(json_data)

    matchings = [
        (match["source_column"], match["target_column"]) for match in data["matches"]
    ]
    return matchings


def compute_mean_ranking_reciprocal(matches, ground_truth):

    ordered_matches = sort_matches(matches)
    total_score = 0
    for input_col, target_col in ground_truth:
        score = 0
        if input_col in ordered_matches:
            ordered_matches_list = [v[0] for v in ordered_matches[input_col]]

            if target_col in ordered_matches_list:
                position = ordered_matches_list.index(target_col)
                score = 1 / (position + 1)
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score


def compute_mean_ranking_reciprocal_adjusted(matches, ground_truth):

    # Group ground truth by its key (input_col)
    gt_per_input_col = {}
    for input_col, target_col in ground_truth:
        if input_col not in gt_per_input_col:
            gt_per_input_col[input_col] = set()
        gt_per_input_col[input_col].add(target_col)

    ordered_matches = sort_matches(matches)

    total_score = 0
    total_queries = 0
    for input_col in ordered_matches.keys():
        gt = gt_per_input_col.get(input_col, set())
        if len(gt) == 0:
            # If no ground truth, we ignore the source column.
            # We could throw an exception instead to find problems in the data.
            continue
        else:
            total_queries += 1

        for idx, (target_col, _) in enumerate(ordered_matches[input_col]):
            if target_col in gt:
                total_score += 1 / (idx + 1)
                break  # MRR only considers the first correct match

    return total_score / total_queries


def calculate_recall_at_k(matches, ground_truth):
    ground_truth_set = set(frozenset(pair) for pair in ground_truth)
    correct_matches = 0
    for ((_, source_col), (_, target_col)), _ in matches.items():
        match_pair = frozenset((source_col, target_col))
        if match_pair in ground_truth_set:
            correct_matches += 1
            ground_truth_set.remove(match_pair)

    total_ground_truth = len(ground_truth)
    recall = correct_matches / total_ground_truth if total_ground_truth > 0 else 0

    return recall


def sort_matches(matches):

    sorted_matches = {entry[0][1]: [] for entry in matches}
    for entry in matches:
        sorted_matches[entry[0][1]].append((entry[1][1], matches[entry]))

    # for key in sorted_matches:
    #     print(key, ' ', sorted_matches[key])
    return sorted_matches


def compute_mean_ranking_reciprocal_detail(matches, ground_truth, details):
    # print("Matches: ", matches)
    # print("Ground Truth: ", ground_truth)

    ordered_matches = sort_matches(matches)
    #
    total_score = 0
    for input_col, target_col in ground_truth:
        score = 0
        # print("Input Col: ", input_col)
        if input_col in ordered_matches:
            ordered_matches_list = [v[0] for v in ordered_matches[input_col]]
            # position = -1
            if target_col in ordered_matches_list:
                position = ordered_matches_list.index(target_col)
                score = 1 / (position + 1)
            else:
                # print(f"1- Mapping {input_col} -> {target_col} not found")
                # for entry in ordered_matches[input_col]:
                #     print(entry)

                s = "\n" + details
                s += f"\n{input_col} -> {target_col} not found"
                s += f"\n\tMethod Matches for {input_col}: {ordered_matches_list}\n"

                with open("log.txt", "a") as file:
                    file.write(s)

        # else:
        # print(f"2- Mapping {input_col} -> {target_col} not found")
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score


def create_result_file(result_folder, result_file, header):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_file, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(header)
        print(f"Result file created at {result_file}")


def record_result(result_file, result):
    with open(result_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(result)
