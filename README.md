# Time-Based Anomaly Detection in Credit Card Transactions

![screenshot-localhost_8888-2025 04 06-14_22_00](https://github.com/user-attachments/assets/a92148ab-1302-4323-afde-2fc2d332872a)

### Project Overview

This project implements three unsupervised/semi-supervised machine learning models to detect fraudulent credit card transactions:

1. Autoencoder (Neural Network)
2. Isolation Forest
3. One-Class SVM
4. Ensemble Model (Combining all three)

The goal is to identify anomalies in transaction data while balancing fraud detection (recall) and false alarms (precision).

### Dataset

The project uses the creditcard.csv dataset, which contains:

- Time: Transaction timestamp (converted to datetime)
- Amount: Transaction value
- V1-V28: Anonymized PCA-transformed features
- Class: Fraud label (0 = normal, 1 = fraud)

Class Imbalance:

- 284,315 normal transactions
- 492 fraudulent transactions (~0.17% of data)

### Methodology

1. Data Preprocessing
- Normalize numerical features (excluding the fraud label).
- Convert Time to datetime and sort chronologically.

2. Anomaly Detection
- Compute a rolling sum of transaction amounts over a 5-transaction window.
- Flag anomalies where the rolling sum exceeds the 99th percentile threshold.

3. Visualization
- Plot rolling sums over time with anomalies highlighted in red.

### Results

1. High Recall Models (Autoencoder/Ensemble):
- Catch 84% of fraud but generate many false alarms.
- Use case: When missing fraud is costlier than manual reviews (e.g., high-risk transactions).

2. Balanced Model (One-Class SVM):
- 79% precision/recall—optimal for most business needs.

3. Isolation Forest:
- Least effective for this dataset (low recall).

### Conclusion

This project demonstrates how unsupervised learning can detect fraud in highly imbalanced datasets. The One-Class SVM offers the best trade-off, while the Ensemble provides the highest anomaly detection capability.

### Source

[Synthetic Bank Transfers Dataset from Kaggle](https://www.kaggle.com/datasets/nyingsha/synthetic-bank-transfers)

