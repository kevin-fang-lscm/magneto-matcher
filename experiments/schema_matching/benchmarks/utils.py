
import os
import csv


def sort_matches(matches):

    sorted_matches = {entry[0][1]: [] for entry in matches}
    for entry in matches:
        sorted_matches[entry[0][1]] .append((entry[1][1], matches[entry]))

    # for key in sorted_matches:
    #     print(key, ' ', sorted_matches[key])
    return sorted_matches


def compute_mean_ranking_reciprocal(matches, ground_truth):
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
                score = 1/(position + 1)
            else:
                print(f"1- Mapping {input_col} -> {target_col} not found")
                for entry in ordered_matches[input_col]:
                    print(entry)
        # else:
            # print(f"2- Mapping {input_col} -> {target_col} not found")
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score


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
                score = 1/(position + 1)
            else:
                print(f"1- Mapping {input_col} -> {target_col} not found")
                for entry in ordered_matches[input_col]:
                    print(entry)

                s = "\n" + details 
                s += f"\n{input_col} -> {target_col} not found"
                s += f"\n\tMethod Matches for {input_col}: {ordered_matches_list}\n"

                with open('log.txt', 'a') as file:
                    file.write(s)

        # else:
            # print(f"2- Mapping {input_col} -> {target_col} not found")
        total_score += score

    final_score = total_score / len(ground_truth)
    return final_score


def create_result_file(result_folder, result_file, header):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    with open(result_file, 'w', newline='') as file:
        writer = csv.writer(file)

        writer.writerow(header)
        print(f"Result file created at {result_file}")


def record_result(result_file, result):
    with open(result_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(result)
