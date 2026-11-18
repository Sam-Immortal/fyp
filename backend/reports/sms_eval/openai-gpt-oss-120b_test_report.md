# SMS Evaluation Report

**Model:** openai-gpt-oss-120b
**Split:** test
**Threshold:** 0.75
**Total rows:** 70
**Completed:** 70
**Failed:** 0

## Score-only Rule (score >= threshold)

| | Pred Scam | Pred Legit |
|---|---|---|
| **Actual Scam** | 34 | 1 |
| **Actual Legit** | 5 | 30 |

- **Recall (TPR):** 0.9714
- **False Positive Rate:** 0.1429
- **Precision:** 0.8718
- **Accuracy:** 0.9143

## App Rule (risk_level in high/critical OR score >= threshold)

| | Pred Scam | Pred Legit |
|---|---|---|
| **Actual Scam** | 35 | 0 |
| **Actual Legit** | 5 | 30 |

- **Recall (TPR):** 1.0000
- **False Positive Rate:** 0.1429
- **Precision:** 0.8750
- **Accuracy:** 0.9286

## Overall Metrics

- **Mean score (scam):** 0.8517
- **Mean score (legit):** 0.3086
- **ROC-AUC:** 0.9567

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
| 1 | orders_delivery | 0.8600 | high | Impersonates Flipkart and asks for OTP, a credential request, indicating a phishing attempt. |
| 61 | urgent_bills | 0.8500 | critical | The message pretends to be from the Income Tax Department (authority impersonation), imposes a deadl... |
| 69 | govt_services | 0.7800 | high | Message pretends to be from UIDAI, a government authority, and includes a link to a site for checkin... |
| 76 | courier_customs | 0.6800 | high | Impersonates a postal service and includes a suspicious link to confirm address, a common phishing t... |
| 80 | bank_alerts | 0.7800 | high | Message pretends to be from SBI, includes a suspicious link to renew a fixed deposit, a classic phis... |
| 85 | bank_alerts | 0.7800 | high | Message pretends to be from SBI, reports a debit and asks the user to call a number, showing authori... |

## Wrong Predictions (app rule)

| ID | Category | Score | Risk Level | Explanation |
|---|---|---|---|---|
| 1 | orders_delivery | 0.8600 | high | Impersonates Flipkart and asks for OTP, a credential request, indicating a phishing attempt. |
| 61 | urgent_bills | 0.8500 | critical | The message pretends to be from the Income Tax Department (authority impersonation), imposes a deadl... |
| 69 | govt_services | 0.7800 | high | Message pretends to be from UIDAI, a government authority, and includes a link to a site for checkin... |
| 80 | bank_alerts | 0.7800 | high | Message pretends to be from SBI, includes a suspicious link to renew a fixed deposit, a classic phis... |
| 85 | bank_alerts | 0.7800 | high | Message pretends to be from SBI, reports a debit and asks the user to call a number, showing authori... |
