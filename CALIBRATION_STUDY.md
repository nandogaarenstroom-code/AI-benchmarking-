# Original Task Calibration Pilot

## Purpose and boundaries

This is a protocol and local analysis tool for calibrating the application's **original**
Reasoning, Coding, and Summarization task families. It does not assert that their current tiers are
hard, representative, or validated. Until observations have been collected and reviewed, all tiers
remain **Uncalibrated**.

The study compares evidence within one category, task-set version, evaluator version, and cohort.
It does not combine scores across categories, participant groups, external benchmarks, or task
versions. Published HELM, BIG-bench, MMLU, GPQA, and SWE-bench results may inform study design and
construct descriptions, but are not a common scale. This project does not copy or import their
questions, datasets, patches, or code.

The current task-set version is `pilot-3`; the evaluator version is `pilot-2`. Freeze both definitions
for each data collection wave. Any change to prompts, answer keys, scoring rules, or task generation
requires a new task-set version; evaluator changes require a new evaluator version. Analyze each
version combination separately and do not pool checkpointed runs across versions.

## Construct map

| Local pilot | Intended narrow construct | Useful external design context | Explicit non-claim |
| --- | --- | --- | --- |
| Reasoning | Find a route satisfying varied interacting ordering constraints, including satisfiable and unsatisfiable instances. | BIG-bench's task diversity and human/model comparison practices. | Not general reasoning, formal proof, or a BIG-bench score. |
| Coding | Implement deterministic inventory-event reconciliation or apply adjustments to opening stock, using generated assertions. | SWE-bench's principle of executing task-specific tests. | Not repository-level issue resolution or comparable to SWE-bench. |
| Summarization | Produce a faithful incident handoff across original upload, checkout, and notification scenarios, preserving impact, cause, recovery, hypotheses, and unknowns. | HELM's scenario-specific reporting and MMLU/GPQA's explicit domain/construct boundaries. | Not broad summarization quality or an official HELM metric. |

The other application categories retain their existing narrow tests; this pilot does not calibrate
them. A graduate-science study inspired by GPQA would need an independently authored and reviewed
science task bank. It is not part of this pilot.

## Preregister before collection

Complete and timestamp a study record before viewing outcome data. At minimum record:

- study identifier, date, responsible researchers, and approvals/consent procedure
- a protocol ID and immutable run-configuration ID; archive the full manifest, including exact
  model/runtime versions and normalized generation settings, and do not pool different config IDs
- frozen task-set/evaluator versions, seed policy, included categories and excluded items
- exact generated item instances and prompt/evaluator fingerprints (the app records an
  `item_instance_id` and SHA-256 `item_fingerprint` for each generated prompt/check set)
- each construct and its operational definition; outcome and partial-credit interpretation
- inclusion/recruitment criteria for domain-relevant and less-specialized human participants
- model names and exact versions, runtime/server versions, prompt template, generation settings,
  repeat policy, and execution environment
- how reviewers are trained, blinded, assigned, and how disagreements/adjudication are handled
- planned sample-size rationale, missing-response and exclusion rules, analysis, and tier-change
  criteria
- privacy, retention, access, and deletion plan

Do not infer difficulty cutoffs after seeing the results. Set provisional tier criteria for the
intended cohort in advance. Treat a small pilot as a way to identify unclear tasks and broken
evaluators, not as representative population calibration.

## Collection protocol

1. Two independent reviewers inspect each item before model/human testing. They check task clarity,
   answer-key validity, edge-case coverage, and whether the evaluator accepts plausible correct
   answers while rejecting contradictions. Record their independent binary judgments and ambiguity
   flags in the item-review CSV schema below; adjudicate before the main response study.
2. Recruit separate cohorts: domain-relevant human participants and less-specialized participants.
   Record no names, contact details, or unnecessary demographics in the response file. Obtain
   informed consent and follow applicable institutional/organizational review requirements.
3. Present original tasks without showing answer keys or evaluator rubric details. Randomize item
   order where feasible; keep task version and presentation protocol fixed. Collect one primary
   response per participant/item. If repeats are needed, record each with a distinct repeat index.
4. Run a varied, explicitly identified model cohort using the frozen prompt and recorded settings.
   Use fresh generated seeds for task variants where available and repeats for stochastic behavior.
   A model label must identify the actual model/version, not only a provider or family.
5. Apply the frozen deterministic evaluator. Have reviewers independently assess the same sampled
   answers using the preregistered rubric. Preserve evaluator output, both reviews, ambiguity flags,
   and adjudication separately; never replace a disagreement with an undocumented label.
6. Export the following CSV schema. `respondent_id` should be a random study pseudonym, not a
   reversible identifier. Use a distinct random `response_id` to join reviewed answers to a protected
   response archive; do not place raw answers in the analysis CSV. Leave missing assessments blank
   rather than entering zero.

```text
protocol_id,run_config_id,task_set_version,evaluator_version,category,subcategory,item_id,item_instance_id,item_fingerprint,response_id,tier,cohort_type,cohort_label,respondent_id,repeat_index,evaluator_pass,score,max_score,reviewer1_pass,reviewer2_pass,reviewer1_ambiguous,reviewer2_ambiguous
```

`protocol_id` identifies the preregistered study protocol. `run_config_id` identifies one exact
human presentation protocol or model/runtime/settings manifest; keep the full manifest outside the
CSV and archive it with the study. `cohort_type` is `human` or `model`. `cohort_label` names a
preregistered human group or an exact model/version. Outcomes are `true`/`false` or `1`/`0`. Scores are numeric from zero through
`max_score`; `evaluator_pass` records the evaluator's binary outcome separately from a partial score.
Reviewer outcome and ambiguity fields are independently completed. Each
`task/evaluator/category/subcategory/item-instance/cohort/respondent/repeat_index` combination must be unique.
`item_id` identifies the task family; `item_instance_id` and `item_fingerprint` identify the exact
seeded prompt and evaluator checks so different generated variants are never mistaken for the same
item. Reuse the exact stable identifiers from the run artifact; do not reconstruct them from item
position or a truncated hash.

Before response collection, save independent item-level reviewer judgments in a second CSV:

```text
protocol_id,task_set_version,evaluator_version,category,subcategory,item_id,item_instance_id,item_fingerprint,tier,reviewer_id,answer_key_valid,wording_clear,evaluator_valid,construct_fit,ambiguous
```

Use true/false for each judgment and a distinct pseudonymous reviewer ID. The analyzer reports each
item's independent judgments and pairwise agreement by dimension; disagreements must be adjudicated
and documented before collecting scored responses. Keep free-text reviewer notes in the
access-controlled study archive, not in the analysis data.

## Analysis and decision rules

Run from the repository root:

```bash
python tools/calibration.py path/to/responses.csv --output path/to/calibration-report.json
```

Include the pre-study review CSV with `--item-reviews path/to/item-reviews.csv`.

The dependency-free analyzer reports, separately for each version/category/subcategory/cohort:

- per-item-instance first-response counts, pass rates with Wilson intervals, unscored counts,
  reviewer disagreement, evaluator agreement with unanimous reviews, repeat consistency, and
  normalized partial-score means
- tier pass rates and respondent-cluster bootstrap intervals (95%, 2,000 resamples)
- normalized score means (score divided by that response's max score)
- per-item item-rest Pearson correlation within a tier when enough variable data exists
- repeat agreement with the first scored response, reported as correlated repeats
- reviewer exact agreement, Cohen's kappa when defined, and evaluator agreement with unanimous
  reviewer pairs
- ambiguity counts and missing outcome counts

Intervals are descriptive pilot uncertainty, not evidence that a tier is certified. The bootstrap resamples respondents as clusters when summarizing tier responses; the additional
response-level Wilson interval does not account for correlation among multiple items answered by one
person and is only a descriptive reference. Neither interval establishes generalizability beyond the
observed cohort. Item-rest correlations are undefined for small or
constant samples, and are not a standalone validity test. Repeated generations do not count as
independent models or participants.

Review item wording and evaluator disagreements before interpreting pass rates. A negative or weak
item-rest relationship may indicate ambiguity, evaluator defects, narrow construct coverage, or
sampling noise; it is not automatically evidence that a question is “too hard.” Tier changes require
the preregistered criteria, evidence of intended within-category ordering, reviewer agreement on
correctness, and a new task/evaluator version. If evidence is insufficient, retain
**Uncalibrated**.

No numeric score from this pilot should be compared with HELM, BIG-bench, MMLU, GPQA, or SWE-bench.
A SWE-bench-inspired repository task track would require a separate agent interface, pinned
repository snapshots, safe isolated execution of repository tests, and its own validation protocol.
It must not be represented by the current function-level Coding category.

## Limitations

- The tool analyzes submitted observations; it does not recruit participants, run model cohorts,
  verify informed consent, or independently judge semantic correctness.
- The current acceptable-answer summaries are finite alternatives. Reviewers should flag plausible
  paraphrases the evaluator rejects and contradictions it accepts before using scores for claims.
- Calibration results apply only to the observed tasks, versions, participants/models, and protocol.
- Synthetic test fixtures validate analysis code only. They are never calibration evidence.
