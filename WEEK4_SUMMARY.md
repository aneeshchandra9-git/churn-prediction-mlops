# Week 4: Model Optimization — Summary

## Overview
Completed three optimization techniques: hyperparameter tuning, feature engineering, and threshold tuning. Results show model has fundamental limitations that optimization alone cannot overcome.

## Week 4a: Hyperparameter Tuning

**Best Hyperparameters Found:**
- C: 0.001 (strong regularization)
- solver: liblinear
- max_iter: 200

**Performance:**
- F1: 0.3211 (vs Week 3: 0.3210)
- Precision: 0.2475
- Recall: 0.4569
- AUC-ROC: 0.4550

**Finding:** Hyperparameter tuning yielded minimal improvement (+0.04%), suggesting hyperparameters were not the bottleneck.

## Week 4b: Feature Engineering

**Features Created:**
- Interaction: ChargePerMonth, TenureCharge, ChargePerTenureYear
- Polynomial: tenure_squared, MonthlyCharges_squared
- Domain: IsHighValue, IsEarlyCustomer, IsMonthToMonth

**Performance:**
- F1: 0.3012 (-6.19% vs Week 4a)
- Precision: 0.2347
- Recall: 0.4204
- AUC-ROC: 0.4533

**Finding:** Feature engineering degraded performance. Model lacks discriminative power.

## Week 4c: Threshold Tuning

**Probability Distribution:**
- Min: 0.4487
- Max: 0.5541
- Mean: 0.5001

**Key Finding:** Model probabilities clustered around 0.5 indicate poor discrimination between churners and non-churners.

**Optimal Threshold:** 0.10 (F1=0.4275)
- Catches 100% of churners
- But reaches out to everyone (impractical)

**Practical Threshold:** 0.50 (default, F1=0.3211)
- Recall: 45.69%
- Precision: 24.75%
- Manageable for production

## Production Model

**Selected:** Logistic Regression (Week 4a)
- Best overall performance
- Simple and deployable
- Threshold: 0.50

**Performance:**
- F1: 0.3211
- Precision: 0.2475
- Recall: 0.4569
- AUC-ROC: 0.4550

## Key Learnings

1. **Hyperparameter tuning limited** — model needs better features, not just better knobs
2. **Feature engineering requires care** — bad features hurt worse than no features
3. **Model lacks discriminative power** — probabilities cluster at 0.5
4. **Optimization has limits** — fundamental issues need fundamental solutions

## Next Steps

**Weeks 5-6:** Deploy to production (FastAPI + Docker)
**Week 7:** Production monitoring
**Week 8:** LLM integration for churn insights

## Files Generated

### Models
- `models/logistic_regression_tuned.pkl` — Best model

### Artifacts
- `week4_artifacts/week4_hyperparameter_tuning_summary.json`
- `week4_artifacts/week4_feature_engineering_summary.json`
- `week4_artifacts/week4_threshold_tuning_summary.json`
- `week4_artifacts/precision_recall_curve.png`
- `week4_artifacts/roc_curve.png`

### MLflow Experiments
- `week4_hyperparameter_tuning`
- `week4_feature_engineering`
- `week4_threshold_tuning`

View at: `http://localhost:5000`