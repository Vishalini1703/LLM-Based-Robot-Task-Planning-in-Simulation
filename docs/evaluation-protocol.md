# Frozen Evaluation Protocol

## Design

The technical experiment uses the 30 cases in `evaluation/commands.json`: 6 valid single-step, 8 valid multi-step, 6 invalid-precondition, 5 unsupported, and 5 ambiguous commands. Five independent Groq calls per command produce 150 trials. `config/kitchen.json` is reloaded before every trial.

The frozen settings are Groq `llama-3.3-70b-versatile`, temperature 0, maximum 1,200 completion tokens, a 30-second API timeout, and at most one validator-guided correction. The final evaluation uses deterministic in-memory execution so all 150 repetitions can use exactly the same declared state. Three separate Webots runs confirm physical adapter behaviour.

## Collection procedure

1. Run all offline tests.
2. Run one excluded pilot case from every category.
3. Freeze hashes with `python -m robot_planner.evaluation_cli freeze`.
4. Collect with `python -m robot_planner.evaluation_cli run`.
5. Resume only missing trial IDs after interruption; never overwrite completed trials silently.
6. Analyse with `python -m robot_planner.evaluation_cli analyse`.
7. Reconcile and generate confidence intervals/figure with `python tools/extended_analysis.py`.

Each raw record retains its command, expected outcome, all planner attempts, raw responses, validation, execution state, timings, and assessment. Pilot files are marked `included_in_final_dataset: false`.

## Measures

- Structured-response rate: contract-bearing planner results / completed trials.
- Plan-validity rate: validator-accepted ready plans / all ready plans.
- Task-success rate: expected executable trials reaching the expected final state / expected executable trials.
- Correct-rejection rate: correctly rejected invalid, unsupported, or ambiguous trials / rejection-expected trials.
- Invalid-step rate: validator-rejected generated steps / generated steps.
- Hallucination rate: trials referencing an unknown action, object, or location / all trials.
- Execution-error rate: runtime failures / validated executions started.
- Correction-attempt rate: trials using a second attempt / all trials.
- Correction-success rate: accepted second attempts / second attempts.
- Response time: mean, median, population standard deviation, minimum, maximum, and nearest-rank 95th percentile.

Failures are classified as malformed response, schema error, unsupported action, missing entity, precondition failure, sequence inconsistency, incorrect acceptance, incorrect rejection, execution mismatch, simulator failure, or API/planner failure.
