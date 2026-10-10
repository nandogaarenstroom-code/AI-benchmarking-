import csv
import tempfile
import unittest
from pathlib import Path

from tools.calibration import CalibrationDataError, analyze, load_responses


FIELDS = [
    "protocol_id", "run_config_id", "task_set_version", "evaluator_version",
    "category", "item_id", "item_instance_id", "item_fingerprint", "tier",
    "cohort_type", "cohort_label", "respondent_id", "repeat_index",
    "evaluator_pass", "score", "max_score", "reviewer1_pass", "reviewer2_pass",
    "reviewer1_ambiguous", "reviewer2_ambiguous",
]


def response(**overrides):
    value = {
        "protocol_id": "protocol-2026-01", "run_config_id": "default-config",
        "task_set_version": "pilot-1", "evaluator_version": "pilot-1",
        "category": "Reasoning", "item_id": "route-family-1",
        "item_instance_id": "route-family-1@seed1", "item_fingerprint": "abc123",
        "tier": "Hard",
        "cohort_type": "human", "cohort_label": "domain-reviewers",
        "respondent_id": "anon-1", "repeat_index": 1,
        "evaluator_pass": "true", "score": "10", "max_score": "10",
        "reviewer1_pass": "true", "reviewer2_pass": "true",
        "reviewer1_ambiguous": "false", "reviewer2_ambiguous": "false",
    }
    value.update(overrides)
    return value


class CalibrationTests(unittest.TestCase):
    def write_csv(self, rows):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "responses.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_analyzer_keeps_versions_categories_and_cohorts_separate(self):
        rows = [
            response(respondent_id="h1"),
            response(respondent_id="h2", evaluator_pass="false", score="0",
                     reviewer1_pass="false", reviewer2_pass="false"),
            response(cohort_type="model", cohort_label="model-a@v1", respondent_id="m1"),
            response(task_set_version="pilot-2", respondent_id="h3"),
            response(run_config_id="model-config-2", cohort_type="model",
                     cohort_label="model-a@v1", respondent_id="m1"),
            response(category="Coding", item_id="code-1", item_instance_id="code-1@seed1",
                     item_fingerprint="def456", respondent_id="h4"),
        ]
        report = analyze(load_responses(self.write_csv(rows)))
        groups = report["groups"]
        self.assertEqual(len(groups), 5)
        self.assertEqual({group["cohort_type"] for group in groups}, {"human", "model"})
        self.assertEqual({group["task_set_version"] for group in groups}, {"pilot-1", "pilot-2"})
        hard = next(group for group in groups if group["task_set_version"] == "pilot-1" and
                    group["cohort_type"] == "human" and group["category"] == "Reasoning")
        tier = next(item for item in hard["tier_results"] if item["tier"] == "Hard")
        self.assertEqual(tier["pass_rate"], 0.5)
        self.assertEqual(tier["mean_score_fraction"], 0.5)
        self.assertGreater(tier["pass_rate_interval"]["low"], 0)
        self.assertEqual(tier["respondent_count"], 2)
        self.assertEqual(tier["pass_rate_interval"]["method"],
                         "95% respondent-cluster bootstrap (2000 resamples)")
        self.assertEqual(hard["status"], "preliminary; no automatic tier certification")

    def test_repeatability_and_reviewer_agreement_are_reported_separately(self):
        rows = [
            response(respondent_id="h1"),
            response(respondent_id="h2", evaluator_pass="false", score="0",
                     reviewer1_pass="false", reviewer2_pass="false"),
            response(respondent_id="h1", repeat_index=2, evaluator_pass="false", score="0",
                     reviewer1_pass="", reviewer2_pass="", reviewer1_ambiguous="",
                     reviewer2_ambiguous=""),
        ]
        report = analyze(load_responses(self.write_csv(rows)))["groups"][0]
        self.assertEqual(report["repeatability"]["paired_repeat_outcomes"], 1)
        self.assertEqual(report["repeatability"]["agreement_with_first_response"], 0)
        self.assertEqual(report["reviewer_agreement"]["paired_reviews"], 2)
        self.assertEqual(report["reviewer_agreement"]["exact_agreement"], 1)
        self.assertEqual(report["reviewer_agreement"]["evaluator_vs_unanimous_review"], 1)
        item = report["item_results"][0]
        self.assertEqual(item["repeat_pairs"], 1)
        self.assertEqual(item["repeat_agreement"], 0)

    def test_unscored_responses_and_undefined_intervals_are_not_zero(self):
        report = analyze(load_responses(self.write_csv([
            response(evaluator_pass="", score="", reviewer1_pass="", reviewer2_pass="",
                     reviewer1_ambiguous="", reviewer2_ambiguous=""),
        ])))["groups"][0]
        self.assertEqual(report["unscored_first_responses"], 1)
        tier = next(item for item in report["tier_results"] if item["tier"] == "Hard")
        self.assertIsNone(tier["pass_rate"])
        self.assertIsNone(tier["pass_rate_interval"])
        self.assertIsNone(tier["mean_score_fraction"])

    def test_loader_rejects_duplicate_observation_identity(self):
        with self.assertRaisesRegex(CalibrationDataError, "duplicate response identity"):
            load_responses(self.write_csv([response(), response()]))

    def test_loader_rejects_invalid_outcomes_and_keeps_unknowns_blank(self):
        with self.assertRaisesRegex(CalibrationDataError, "evaluator_pass"):
            load_responses(self.write_csv([response(evaluator_pass="maybe")]))
        records = load_responses(self.write_csv([response(evaluator_pass="", score="")]))
        self.assertTrue(records[0]["unscored"])

    def test_item_instances_are_not_merged_and_discrimination_is_reported(self):
        rows = []
        for index, result in enumerate((True, True, False, False)):
            rows.append(response(
                respondent_id=f"person-{index}",
                evaluator_pass=str(result).lower(),
                score="10" if result else "0",
                reviewer1_pass=str(result).lower(), reviewer2_pass=str(result).lower(),
            ))
        for index, result in enumerate((True, False, True, False)):
            rows.append(response(
                item_instance_id="route-family-1@seed2", item_fingerprint="xyz987",
                respondent_id=f"person-{index}",
                evaluator_pass=str(result).lower(),
                score="10" if result else "0",
                reviewer1_pass=str(result).lower(), reviewer2_pass=str(result).lower(),
            ))
        report = analyze(load_responses(self.write_csv(rows)))["groups"][0]
        self.assertEqual(len(report["item_results"]), 2)
        self.assertEqual({item["item_instance_id"] for item in report["item_results"]},
                         {"route-family-1@seed1", "route-family-1@seed2"})
        self.assertTrue(all(item["pass_rate_interval"]["method"].startswith("95% Wilson")
                            for item in report["item_results"]))
        self.assertTrue(all(item["item_rest_correlation"] is not None
                            for item in report["item_rest_discrimination"]))


if __name__ == "__main__":
    unittest.main()
