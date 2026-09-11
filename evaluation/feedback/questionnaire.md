# Anonymous Feedback Questionnaire

Anonymous participant code: `P___`

Rate each statement from 1 (strongly disagree) to 5 (strongly agree).

1. The system's result was clear. (`clarity`)
2. The command-and-result workflow appeared easy to use. (`usability`)
3. I could understand the generated plan and outcome. (`understandability`)
4. The prototype appeared useful for exploring robot task planning. (`usefulness`)
5. I was confident that the rejected command explanation made sense. (`rejection_confidence`)

Optional free text:

- What was the clearest feature?
- What was the most confusing feature?
- What one improvement would you suggest?

Transfer answers to `responses.csv` using only the anonymous code. Aggregate approved responses with:

```powershell
python -m robot_planner.feedback_cli evaluation/feedback/responses.csv --invited 10
```

Replace `10` with the actual approved number invited. The command validates identifiers and rating ranges, rejects duplicate codes, and writes only aggregate ratings and anonymous comments to `evaluation/feedback/summary.json`.
