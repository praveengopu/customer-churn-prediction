# Customer Churn Prediction

A reproducible machine-learning portfolio project demonstrating a binary customer-churn classification workflow using Python, Pandas, NumPy, and Scikit-learn.

> **Important:** This project uses a synthetic dataset created for learning and portfolio demonstration. The results are not real customer, company, or production results.

## Objective

Build and evaluate machine-learning models that classify whether a customer is likely to churn based on customer attributes.

## Dataset

The dataset contains 300 synthetic customer records with:

- `tenure_months`
- `monthly_charges`
- `support_tickets`
- `contract_type`
- `payment_method`
- `churn` — target variable

The dataset was generated with a fixed random seed so the experiment can be reproduced.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook / Python scripts
- GitHub

## Methodology

1. Load the synthetic dataset.
2. Separate features from the churn target.
3. Split the data into training and test sets using an 80/20 stratified split.
4. Impute missing numerical values using the median.
5. Standardize numerical features.
6. Impute missing categorical values using the most frequent value.
7. One-hot encode categorical features.
8. Train Logistic Regression and Random Forest classifiers.
9. Evaluate the models using accuracy, precision, recall, and F1-score.

## Model Results

Results generated from the reproducible experiment:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.683 | 0.655 | 0.679 | 0.667 |
| Random Forest | 0.750 | 0.724 | 0.750 | 0.737 |

These metrics describe performance on the synthetic test set only. They should not be interpreted as production performance.

## Project Structure

```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── churn_data.csv
├── train_model.py
└── results/
    └── model_metrics.csv
```

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python train_model.py
```

The script trains both models and generates the evaluation metrics.

## Skills Demonstrated

- Python programming
- Pandas and NumPy
- Data preprocessing
- Numerical feature scaling
- Categorical feature encoding
- Train-test splitting
- Classification
- Logistic Regression
- Random Forest
- Model evaluation
- Precision, recall, F1-score, and accuracy
- Reproducible data science workflow

## Limitations

This is a portfolio-learning project using synthetic data. The dataset does not represent a real company's customers, and the model has not been deployed in a production environment.

Future improvements could include testing on a public real-world churn dataset, hyperparameter tuning, cross-validation, feature-importance analysis, and model comparison across additional algorithms.

## Author

**Praveen Kumar**

Computer Science and Data Science student at Vellore Institute of Technology.
