# Machine Learning Assignment 2

# ###############################
# Ankit Nainwal  ################
# 2025AC05971    ################
# ###############################

## a) Problem statement
In this assignment, I have built a binary classification system to classify breast cancer tumors as Malignant (M) or Benign (B). I implemented all required ML models on the same dataset and evaluated them using the required metrics.

## b) Dataset description
- Dataset name: Breast Cancer Wisconsin (Diagnostic) Data Set
- Source: Kaggle - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- Total records: 569
- Features used: 30 numeric features
- Target column: `diagnosis` (M = malignant, B = benign)
- Problem type: Binary Classification

Files used:
- `data.csv` for model training and evaluation
- `test_data.csv` for Streamlit upload/testing

## c) Github Repository Link
- Repository: https://github.com/2025ac05971/ML_Assignment_2

Mandatory files in repo:
- `app.py`
- `requirements.txt`
- `README.md`
- `test_data.csv`
- `model/` (model files and training script)

## d) Models used
Implemented models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)

### Comparison Table (Required Metrics)
| Model               | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|---------------------|----------|--------|-----------|--------|--------|--------|
| Random Forest       | 0.9649   | 0.9970 | 1.0000    | 0.9048 | 0.9500 | 0.9258 |
| Logistic Regression | 0.9649   | 0.9960 | 0.9750    | 0.9286 | 0.9512 | 0.9245 |
| KNN                 | 0.9561   | 0.9823 | 0.9744    | 0.9048 | 0.9383 | 0.9058 |
| Decision Tree       | 0.9298   | 0.9246 | 0.9048    | 0.9048 | 0.9048 | 0.8492 |
| Naive Bayes         | 0.9211   | 0.9891 | 0.9231    | 0.8571 | 0.8889 | 0.8292 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong performance with balanced precision and recall. Good general baseline model. |
| Decision Tree | Easy to interpret, but lower performance than other models on this dataset. |
| KNN | Performs well after feature scaling, but slightly lower than Logistic Regression and Random Forest. |
| Naive Bayes | Good AUC score, but lower recall and MCC compared to other models. |
| Random Forest (Ensemble) | Best overall balance with highest AUC, best precision, and top MCC score. |
| Overall Winner for your dataset? | **Random Forest (Ensemble)** |

## Project structure
```
ML_Assignment_2/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- data.csv
|-- model_metrics.csv
|-- model/
|   |-- train_models.py
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest.pkl
|   |-- feature_columns.json
```

## Streamlit app features implemented
- CSV upload option for test data
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix
- Classification report
- Results of different models visible in app using model comparison and model selection


## Streamlit Deployment Link
- https://2025ac05971.streamlit.app/


### Streamlit App Features
- CSV file upload with validation
- Model comparison table (all 5 models, 6 metrics)
- Model selection with default model and cancel button
- Individual model metrics display
- Confusion matrix visualization
- Classification report with labels
- Multi-model comparison functionality
- Model ranking by performance
- Dataset information display
	- Training data: 569 instances
	- Test data: 114 instances (20% split)
	- Features count: 30
	- Class distribution: Benign/Malignant


## Screenshot section (for final PDF)
- Added in pdf file.
