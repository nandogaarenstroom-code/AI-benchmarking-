"""Analyze versioned human and model calibration-pilot response CSV files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


TIERS = ("Easy", "Medium", "Hard", "Uncalibrated")
REQUIRED_COLUMNS = {
    "protocol_id", "run_config_id", "task_set_version", "evaluator_version",
    "category", "item_id", "item_instance_id", "item_fingerprint", "tier",
    "cohort_type", "cohort_label", "respondent_id", "repeat_index",
    "evaluator_pass", "score", "max_score", "reviewer1_pass", "reviewer2_pass",
    "reviewer1_ambiguous", "reviewer2_ambiguous",
}
BOOLEAN_FIELDS = (
    "evaluator_pass", "reviewer1_pass", "reviewer2_pass",
    "reviewer1_ambiguous", "reviewer2_ambiguous",
)
BOOTSTRAP_SAMPLES = 2000


class CalibrationDataError(ValueError):
    """Raised when calibration observations cannot be safely analyzed."""


def _parse_bool(value, field, line):
    value = value.strip().casefold()
    if not value:
        return None
    if value in ("1", "true", "yes"):
        return True
    if value in ("0", "false", "no"):
        return False
    raise CalibrationDataError(f"line {line}: {field} must be true/false or blank")


def load_responses(path):
    """Load and validate response records; blank outcome cells remain unscored."""
    records = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise CalibrationDataError("CSV must include a header row")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise CalibrationDataError(
                "CSV is missing required columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, start=2):
            record = {name: (row.get(name) or "").strip() for name in REQUIRED_COLUMNS}
            for field in ("protocol_id", "run_config_id", "task_set_version",
                          "evaluator_version", "category", "item_id", "item_instance_id",
                          "item_fingerprint",
                          "cohort_label", "respondent_id"):
                if not record[field]:
                    raise CalibrationDataError(f"line {line}: {field} cannot be blank")
            if record["tier"] not in TIERS:
                raise CalibrationDataError(
                    f"line {line}: tier must be one of {', '.join(TIERS)}")
            if record["cohort_type"] not in ("human", "model"):
                raise CalibrationDataError(f"line {line}: cohort_type must be human or model")
            try:
                record["repeat_index"] = int(record["repeat_index"])
                if record["repeat_index"] < 1:
                    raise ValueError
            except (TypeError, ValueError):
                raise CalibrationDataError(f"line {line}: repeat_index must be a positive integer")
            identity = tuple(record[field] for field in (
                "protocol_id", "run_config_id", "task_set_version", "evaluator_version",
                "category", "item_id", "item_instance_id",
                "cohort_type", "cohort_label", "respondent_id", "repeat_index"))
            if identity in seen:
                raise CalibrationDataError(f"line {line}: duplicate response identity")
            seen.add(identity)
            for field in BOOLEAN_FIELDS:
                record[field] = _parse_bool(record[field], field, line)
            try:
                record["max_score"] = float(record["max_score"])
                if not math.isfinite(record["max_score"]) or record["max_score"] <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise CalibrationDataError(f"line {line}: max_score must be a positive number")
            try:
                record["score"] = float(record["score"]) if record["score"] else None
                if record["score"] is not None and (
                        not math.isfinite(record["score"]) or
                        not 0 <= record["score"] <= record["max_score"]):
                    raise ValueError
            except (TypeError, ValueError):
                raise CalibrationDataError(
                    f"line {line}: score must be blank or between zero and max_score")
            if record["evaluator_pass"] is None:
                record["unscored"] = True
            else:
                record["unscored"] = False
            records.append(record)
    return records


def _cluster_interval(respondent_values, seed_text):
    """Percentile interval over respondents, keeping a person's items clustered."""
    values = list(respondent_values.values())
    if not values:
        return None
    if len(values) == 1:
        return {"low": round(values[0], 4), "high": round(values[0], 4),
                "method": "single respondent; interval not estimable"}
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"low": round(means[int(.025 * (len(means) - 1))], 4),
            "high": round(means[int(.975 * (len(means) - 1))], 4),
            "method": f"95% respondent-cluster bootstrap ({BOOTSTRAP_SAMPLES} resamples)"}


def _cohen_kappa(pairs):
    if not pairs:
        return None
    count = len(pairs)
    observed = sum(left == right for left, right in pairs) / count
    left_yes = sum(left for left, _ in pairs) / count
    right_yes = sum(right for _, right in pairs) / count
    expected = left_yes * right_yes + (1 - left_yes) * (1 - right_yes)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def _correlation(pairs):
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(variance_x * variance_y)
    return round(covariance / denominator, 4) if denominator else None


def analyze(records):
    """Return item and tier evidence without pooling cohorts, categories or versions."""
    groups = defaultdict(list)
    for record in records:
        scope = (record["protocol_id"], record["run_config_id"],
                 record["task_set_version"], record["evaluator_version"],
                 record["category"], record["cohort_type"], record["cohort_label"])
        groups[scope].append(record)

    reports = []
    for scope, group in sorted(groups.items()):
        (protocol_id, run_config_id, task_version, evaluator_version, category,
         cohort_type, cohort_label) = scope
        scope_label = "|".join(scope)
        first = [row for row in group if row["repeat_index"] == 1]
        all_by_identity = defaultdict(list)
        reviewers = []
        evaluator_consensus = []
        item_rows = defaultdict(list)
        for row in first:
            item_rows[(row["tier"], row["item_instance_id"])].append(row)
        for row in group:
            all_by_identity[(row["tier"], row["item_instance_id"],
                             row["respondent_id"])].append(row)
            if row["reviewer1_pass"] is not None and row["reviewer2_pass"] is not None:
                pair = (row["reviewer1_pass"], row["reviewer2_pass"])
                reviewers.append(pair)
                if row["repeat_index"] == 1 and row["evaluator_pass"] is not None:
                    consensus = pair[0] if pair[0] == pair[1] else None
                    if consensus is not None:
                        evaluator_consensus.append((row["evaluator_pass"], consensus))

        item_reports = []
        for (tier, instance_id), rows in sorted(item_rows.items()):
            scored = [row for row in rows if row["evaluator_pass"] is not None]
            pass_count = sum(row["evaluator_pass"] for row in scored)
            reviewer_rows = [row for row in rows if row["reviewer1_pass"] is not None and
                             row["reviewer2_pass"] is not None]
            score_fractions = [row["score"] / row["max_score"] for row in rows
                               if row["score"] is not None]
            item_reports.append({
                "item_id": rows[0]["item_id"], "item_instance_id": instance_id,
                "item_fingerprint": rows[0]["item_fingerprint"],
                "tier": tier, "response_count": len(rows),
                "unique_respondents": len({row["respondent_id"] for row in rows}),
                "scored_first_responses": len(scored),
                "unscored_first_responses": len(rows) - len(scored),
                "pass_rate": round(pass_count / len(scored), 4) if scored else None,
                "reviewer_pairs": len(reviewer_rows),
                "reviewer_agreement": round(sum(
                    row["reviewer1_pass"] == row["reviewer2_pass"]
                    for row in reviewer_rows) / len(reviewer_rows), 4) if reviewer_rows else None,
                "ambiguous_reviews": sum(
                    row["reviewer1_ambiguous"] is True or
                    row["reviewer2_ambiguous"] is True for row in rows),
                "mean_score_fraction": round(sum(score_fractions) / len(score_fractions), 4)
                if score_fractions else None,
            })

        tier_reports = []
        for tier in TIERS:
            tier_rows = [row for row in first if row["tier"] == tier]
            scored = [row for row in tier_rows if row["evaluator_pass"] is not None]
            by_respondent = defaultdict(list)
            for row in scored:
                by_respondent[row["respondent_id"]].append(
                    1.0 if row["evaluator_pass"] else 0.0)
            respondent_values = {person: sum(values) / len(values)
                                 for person, values in by_respondent.items()}
            successes = sum(row["evaluator_pass"] for row in scored)
            tier_scores = [row["score"] / row["max_score"] for row in tier_rows
                           if row["score"] is not None]
            tier_reports.append({
                "tier": tier, "item_count": len({row["item_instance_id"] for row in tier_rows}),
                "first_response_count": len(tier_rows),
                "scored_response_count": len(scored),
                "respondent_count": len(respondent_values),
                "pass_rate": round(successes / len(scored), 4) if scored else None,
                "respondent_cluster_pass_rate": round(
                    sum(respondent_values.values()) / len(respondent_values), 4)
                    if respondent_values else None,
                "pass_rate_interval": _cluster_interval(
                    respondent_values, f"{scope_label}|{tier}") if respondent_values else None,
                "mean_score_fraction": round(sum(tier_scores) / len(tier_scores), 4)
                if tier_scores else None,
            })

        tier_item_respondent_outcomes = defaultdict(dict)
        for (tier, instance_id), rows in item_rows.items():
            per_respondent = defaultdict(list)
            for row in rows:
                if row["evaluator_pass"] is not None:
                    per_respondent[row["respondent_id"]].append(
                        float(row["evaluator_pass"]))
            tier_item_respondent_outcomes[tier][instance_id] = {
                person: sum(values) / len(values)
                for person, values in per_respondent.items()
            }
        discrimination = []
        for (tier, instance_id), rows in sorted(item_rows.items()):
            target = tier_item_respondent_outcomes[tier][instance_id]
            comparison = []
            for respondent, target_score in target.items():
                rest = [respondents[respondent]
                        for other_id, respondents in tier_item_respondent_outcomes[tier].items()
                        if other_id != instance_id and respondent in respondents]
                if rest:
                    comparison.append((target_score, sum(rest) / len(rest)))
            discrimination.append({
                "item_id": rows[0]["item_id"], "item_instance_id": instance_id,
                "item_rest_correlation": _correlation(comparison),
                "respondents_with_rest_score": len(comparison),
            })

        repeat_sets = [rows for rows in all_by_identity.values() if len(rows) > 1]
        repeat_pairs = []
        for rows in repeat_sets:
            ordered = sorted(rows, key=lambda row: row["repeat_index"])
            first_outcome = ordered[0]["evaluator_pass"]
            later = [row["evaluator_pass"] for row in ordered[1:]
                     if row["evaluator_pass"] is not None and first_outcome is not None]
            repeat_pairs.extend((first_outcome, outcome) for outcome in later)

        reviewer_agreement = sum(left == right for left, right in reviewers) / len(reviewers) \
            if reviewers else None
        reports.append({
            "protocol_id": protocol_id, "run_config_id": run_config_id,
            "task_set_version": task_version, "evaluator_version": evaluator_version,
            "category": category, "cohort_type": cohort_type, "cohort_label": cohort_label,
            "status": "preliminary; no automatic tier certification",
            "first_response_count": len(first),
            "repeat_observation_count": sum(row["repeat_index"] > 1 for row in group),
            "unscored_first_responses": sum(row["unscored"] for row in first),
            "tier_results": tier_reports, "item_results": item_reports,
            "item_rest_discrimination": discrimination,
            "repeatability": {
                "paired_repeat_outcomes": len(repeat_pairs),
                "agreement_with_first_response": round(
                    sum(left == right for left, right in repeat_pairs) / len(repeat_pairs), 4)
                    if repeat_pairs else None,
                "note": "Repeated model generations are correlated observations, not independent models."
                         if cohort_type == "model" else
                         "Repeated responses are correlated observations, not independent participants.",
            },
            "reviewer_agreement": {
                "paired_reviews": len(reviewers),
                "exact_agreement": round(reviewer_agreement, 4) if reviewer_agreement is not None else None,
                "cohen_kappa": _cohen_kappa(reviewers),
                "evaluator_vs_unanimous_review": round(
                    sum(left == right for left, right in evaluator_consensus) /
                    len(evaluator_consensus), 4) if evaluator_consensus else None,
                "evaluator_review_pairs": len(evaluator_consensus),
            },
            "interpretation": (
                "Results are confined to this cohort, category, task/evaluator version and protocol. "
                "They are not numerically comparable to external benchmark scores.")
        })
    return {
        "report_version": "calibration-analysis-1",
        "analysis_method": {
            "difficulty": "first response per respondent/item; tier intervals bootstrap respondents",
            "interval": "percentile cluster bootstrap; descriptive, not a certification rule",
            "discrimination": "Pearson item-rest correlation within tier; omitted when undefined",
            "repeatability": "first response compared with later scored repeats",
            "cohort_policy": "protocols, run configurations, cohorts, categories and versions are never pooled",
        },
        "groups": reports,
        "limitations": [
            "This report analyzes only supplied observations; it does not collect or invent calibration data.",
            "Human and model cohorts are reported separately and are not universal reference populations.",
            "External benchmark scores are not merged or treated as equivalent constructs.",
            "A calibration study requires independent item review and a pre-registered decision protocol.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path, help="CSV using the schema in CALIBRATION_STUDY.md")
    parser.add_argument("--output", type=Path, help="Write JSON report here; default is stdout")
    args = parser.parse_args(argv)
    try:
        report = analyze(load_responses(args.responses))
    except (OSError, CalibrationDataError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
