# Standard Participant Task Script

Run the same four-part demonstration for every participant.

1. Explain that the LLM proposes a plan but the deterministic validator controls whether it may execute.
2. Run `robot-task "Put the apple in the basket"` and show the structured `find`, `pick`, `navigate`, and `place` steps plus the successful outcome.
3. Run `robot-task "Cook the apple"` and show the rejection explanation; explain that cooking is not in the fixed action library.
4. Show one Webots screenshot sequence from a previously verified run. Do not expose the `.env` file, API key, raw participant records, or unrelated machine files.
5. Ask the participant to complete the questionnaire independently.

The demonstrator must not coach ratings or describe a preferred answer.
