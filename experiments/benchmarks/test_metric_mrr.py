# To run this test, use the following command:
#   pytest experiments/benchmarks/test_metric_mrr.py -s
#
from valentine.metrics import *
from valentine.algorithms.matcher_results import MatcherResults

from utils import compute_mean_ranking_reciprocal_adjusted


def test_mean_reciprocal_rank():
    # Case 1 - Correct match at position 1
    matches = MatcherResults(
        {
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "Cited by")): 0.9374313,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )
    ground_truth = [
        ("Cited by", "Cited by"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0

    # Case 2 - Correct match at position 3
    matches = MatcherResults(
        {
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.9349037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8514057,
            (
                ("table_1", "Cited by"),
                ("table_2", "Cited by"),
            ): 0.8204313,  # correct match
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8115057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )

    ground_truth = [
        ("Cited by", "Cited by"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1 / 3

    # Case 3 - Multiple source columns at position 1
    matches = MatcherResults(
        {
            (
                ("table_1", "Cited by"),
                ("table_2", "Cited by"),
            ): 0.8374313,  # correct match
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
            (("table_1", "name"), ("table_2", "firstName")): 0.9349037,  # correct match
            (("table_1", "name"), ("table_2", "EID")): 0.8514057,
            (("table_1", "name"), ("table_2", "Cited by")): 0.8204313,
            (("table_1", "name"), ("table_2", "DUMMY1")): 0.8115057,
            (("table_1", "name"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )

    ground_truth = [
        ("Cited by", "Cited by"),
        ("name", "firstName"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0

    # Case 4 - Multiple source columns at different positions
    matches = MatcherResults(
        {
            (
                ("table_1", "Cited by"),
                ("table_2", "Cited by"),
            ): 0.8374313,  # correct match
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
            (("table_1", "name"), ("table_2", "Cited by")): 0.9349037,
            (("table_1", "name"), ("table_2", "EID")): 0.8514057,
            (("table_1", "name"), ("table_2", "firstName")): 0.8204313,  # correct match
            (("table_1", "name"), ("table_2", "DUMMY1")): 0.8115057,
            (("table_1", "name"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )

    ground_truth = [
        ("Cited by", "Cited by"),
        ("name", "firstName"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0 / 2.0 * (1.0 + 1.0 / 3.0)

    # Case 5 - Multiple correct matches, at positions 1 and 2
    matches = MatcherResults(
        {
            (("table_1", "Cited by"), ("table_2", "Cited by")): 0.8374313,
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )
    ground_truth = [
        ("Cited by", "Cited by"),
        ("Cited by", "Authors"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0

    # Case 5 - Multiple correct matches, at positions 2 and 3
    matches = MatcherResults(
        {
            (("table_1", "Cited by"), ("table_2", "EID")): 0.9214057,
            (("table_1", "Cited by"), ("table_2", "Cited by")): 0.8374313,
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )
    ground_truth = [
        ("Cited by", "Cited by"),
        ("Cited by", "Authors"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0 / 2.0

    # Case 6 - Edge-case with multiple source columns and multiple correct matches
    matches = MatcherResults(
        {
            (
                ("table_1", "Cited by"),
                ("table_2", "Cited by"),
            ): 0.8374313,  # correct match at rank 1
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
            (("table_1", "name"), ("table_2", "Cited by")): 0.9349037,
            (("table_1", "name"), ("table_2", "EID")): 0.8514057,
            (
                ("table_1", "name"),
                ("table_2", "firstName"),
            ): 0.8204313,  # correct match at rank 3
            (("table_1", "name"), ("table_2", "DUMMY1")): 0.8115057,
            (("table_1", "name"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )

    ground_truth = [
        ("Cited by", "Cited by"),
        ("name", "firstName"),
        ("name", "DUMMY1"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0 / 2.0 * (1 / 1 + 1 / 3)

    # Case 7 - Multiple correct matches at position 2 and 3
    matches = MatcherResults(
        {
            (
                ("table_1", "Cited by"),
                ("table_2", "Cited by"),
            ): 0.8374313,  # correct match at rank 1
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
            (("table_1", "name"), ("table_2", "Cited by")): 0.9349037,
            (("table_1", "name"), ("table_2", "EID")): 0.8514057,
            (("table_1", "name"), ("table_2", "firstName")): 0.8204313,
            (("table_1", "name"), ("table_2", "DUMMY1")): 0.8115057,
            (("table_1", "name"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )
    ground_truth = [
        ("Cited by", "Cited by"),
        ("Cited by", "Authors"),
        # no ground truth for name -> firstName
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 1.0  # we ignore the sourcce column name since it has no ground truth

    # Case 8 - Correct match is not found
    matches = MatcherResults(
        {
            (("table_1", "Cited by"), ("table_2", "LastName")): 0.9374313,
            (("table_1", "Cited by"), ("table_2", "Authors")): 0.83498037,
            (("table_1", "Cited by"), ("table_2", "EID")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY1")): 0.8214057,
            (("table_1", "Cited by"), ("table_2", "DUMMY2")): 0.8114057,
        }
    )
    ground_truth = [
        ("Cited by", "Cited by"),
    ]

    mrr = compute_mean_ranking_reciprocal(matches, ground_truth)
    assert mrr == 0.0
