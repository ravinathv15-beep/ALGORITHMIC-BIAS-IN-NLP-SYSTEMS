# Algorithmic Bias in NLP Systems
### Analysis of Real-World Cases and Mitigation Strategies

Mini project (Text Analytics) · B.E. CSE (AI & ML) · EASA College of Engineering & Technology, Anna University

## Overview
Text-based ML models now score sentiment, moderate toxic content and screen
resumes, yet documented cases (Amazon's recruiting tool, toxicity classifiers
flagging identity terms, hate-speech detectors over-flagging African-American
English) show they can learn and amplify bias in their training data.

This project reproduces those failure mechanisms in two controlled, synthetic
experiments, audits them with fairness metrics, and compares three mitigation
strategies.

## Case Studies
1. **Sentiment / toxicity classifier**: identity terms (e.g. *muslim*, *gay*)
   co-occur with negative labels in training data.
2. **Resume screener**: historical hiring labels discriminate against
   qualified women.

## Method
- **Model:** TF-IDF (unigrams + bigrams, sublinear TF) + Logistic Regression
- **Fairness metrics:** FPR gap, disparate impact ratio, equal opportunity
  difference, counterfactual flip rate
- **Mitigation (pre-processing):**
  - Identity masking / anonymisation
  - Counterfactual Data Augmentation (CDA)
  - Reweighing (Kamiran & Calders, 2012)

## Key Results (mean of 5 seeds)
| Task | Metric | Baseline | Best mitigated |
|---|---|---|---|
| Sentiment | Accuracy | 88.1% | 92.2% |
| Sentiment | FPR gap | 0.178 | 0.006 – 0.010 |
| Resume | Accuracy (vs. merit) | 86.0% | 88.9% |
| Resume | Disparate impact | 0.621 | 0.986 (anonymisation) |

**Takeaway:** high accuracy does not imply fairness. The biased baselines
looked accurate while failing the four-fifths rule. CDA and Reweighing only
partly fixed the resume bias because the biased labels remained.

## Run It
Requires Python 3, `numpy`, `scikit-learn`. No external data or internet needed.

```bash
pip install numpy scikit-learn
python nlp_bias_audit.py
```

The script prints a single-run table (seed 42), 5-seed mean tables, and the
standard deviation of accuracy.

## Limitations
Datasets are synthetic and template-based; results are not measurements of any
commercial system. Only binary gender and a small set of identity terms are
covered, and the fairness metrics cannot all be satisfied simultaneously.

## Future Work
Real corpora (Jigsaw, IMDB), transformer models (BERT), intersectional groups,
adversarial debiasing, and an interactive bias-audit dashboard.

## Author
**Ravinath V** · Register No. E720524AM021
Supervisor: Mr. C G Kirubhanandhan, Asst. Professor

## References
Dastin (2018); Bolukbasi et al. (2016); Caliskan et al. (2017);
Kiritchenko & Mohammad (2018); Dixon et al. (2018); Sap et al. (2019);
Blodgett et al. (2020); Hardt et al. (2016); Kamiran & Calders (2012);
Mehrabi et al. (2021).
