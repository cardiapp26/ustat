# uSTAT 10×20 statistical audit fixture

`dataset.csv` is a deterministic synthetic dataset for read-only validation of
uSTAT's Tests and Table tabs.

- Shape: 10 rows × 20 columns
- Continuous variables: age, biomarkers, outcomes, and repeated scores
- Categorical variables: binary, three-level, stratified, and ordered groups
- Missing values: `biomarker_normal`, `biomarker_skew`, `category_three`, and
  `followup_score`
- No patient or governed data
- Random seed: none; values are fixed in source control

The dataset is intentionally small. Methods whose own sample-size guard exceeds
10 rows must reject it rather than be described as validated. Repeated-measures
long-format checks may reshape the three wide score columns deterministically;
the source artifact remains 10×20.

Run from the repository root:

```bash
.venv/bin/python qa/statistical_audit_10x20/audit.py
```

`audit.py` prints a JSON result document. Package versions, dataset SHA-256,
product-source content hashes, working-tree diff hash, requests, endpoint
outputs, independent calculations, differences, and explicit non-runnable
cases are included.
