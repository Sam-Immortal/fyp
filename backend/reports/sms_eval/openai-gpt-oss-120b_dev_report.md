# SMS Evaluation Report

**Model:** openai-gpt-oss-120b
**Split:** dev
**Threshold:** 0.75
**Total rows:** 30
**Completed:** 30
**Failed:** 0

## Score-only Rule (score >= threshold)

| | Pred Scam | Pred Legit |
|---|---|---|
| **Actual Scam** | 14 | 1 |
| **Actual Legit** | 1 | 14 |

- **Recall (TPR):** 0.9333
- **False Positive Rate:** 0.0667
- **Precision:** 0.9333
- **Accuracy:** 0.9333

## App Rule (risk_level in high/critical OR score >= threshold)

| | Pred Scam | Pred Legit |
|---|---|---|
| **Actual Scam** | 14 | 1 |
| **Actual Legit** | 1 | 14 |

- **Recall (TPR):** 0.9333
- **False Positive Rate:** 0.0667
- **Precision:** 0.9333
- **Accuracy:** 0.9333

## Overall Metrics

- **Mean score (scam):** 0.8587
- **Mean score (legit):** 0.2467
- **ROC-AUC:** 0.9867

## Error Rate by Category

| Category | Error Rate |
|---|---|
| appointments | 0.00% |
| bank_alerts | 0.00% |
| bank_kyc | 0.00% |
| courier_customs | 0.00% |
| fraud_awareness | 0.00% |
| govt_services | 0.00% |
| job_task | 0.00% |
| loan_invest | 0.00% |
| lottery_prize | 0.00% |
| orders_delivery | 0.00% |
| otp_phishing | 0.00% |
| personal | 0.00% |
| prompt_injection | 0.00% |
| telecom | 0.00% |
| upi_cashback | 0.00% |
| urgent_bills | 0.00% |
| utility_fine | 0.00% |

## Wrong Predictions (score-only rule)

| ID | Category | Score | Risk Level | Explanation |
|---|---|---|---|---|
| 14 | urgent_bills | 0.7800 | high | Impersonates LIC, adds urgency to pay via link and threatens policy lapse; no credential request or ... |
| 59 | job_task | 0.5500 | medium | The SMS offers payment for online reviews, which is a reward-like promise, but lacks urgency, author... |

## Wrong Predictions (app rule)

| ID | Category | Score | Risk Level | Explanation |
|---|---|---|---|---|
| 14 | urgent_bills | 0.7800 | high | Impersonates LIC, adds urgency to pay via link and threatens policy lapse; no credential request or ... |
| 59 | job_task | 0.5500 | medium | The SMS offers payment for online reviews, which is a reward-like promise, but lacks urgency, author... |
