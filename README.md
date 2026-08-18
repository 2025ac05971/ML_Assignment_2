# Machine Learning Assignment 2

Name: Ankit Nainwal  
BITS ID: 2025AC05971

## a) Problem statement
This project builds a binary classification system to classify breast cancer tumors as Malignant (M) or Benign (B). Five machine learning models are trained and evaluated on the same dataset using the required metrics.

## b) Dataset description
- Dataset name: Breast Cancer Wisconsin (Diagnostic) Data Set
- Source: Kaggle - https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- `data.csv`: 569 total records (357 Benign, 212 Malignant)
- `test_data.csv`: 114 test records (72 Benign, 42 Malignant)
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
- `model/` (trained model files and training notebook)

## d) Models used
Implemented models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (kNN)
4. Naive Bayes Classifier (GaussianNB)
5. Random Forest (Ensemble Model)

### Comparison Table (Required Metrics)
| ML Model Name                          | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|----------------------------------------|---------:|----------:|----------:|-------:|---------:|----------:|
| Logistic Regression                    |   0.9649 |    0.9960 |    0.9750 | 0.9286 |   0.9512 |    0.9245 |
| Decision Tree Classifier               |   0.9298 |    0.9246 |    0.9048 | 0.9048 |   0.9048 |    0.8492 |
| K-Nearest Neighbor Classifier (kNN)    |   0.9561 |    0.9823 |    0.9744 | 0.9048 |   0.9383 |    0.9058 |
| Naive Bayes Classifier (GaussianNB)    |   0.9211 |    0.9891 |    0.9231 | 0.8571 |   0.8889 |    0.8292 |
| Random Forest (Ensemble)               |   0.9649 |    0.9970 |    1.0000 | 0.9048 |   0.9500 |    0.9258 |

### Observations on Model Performance
| ML Model Name                       | Observation about model performance |
|-------------------------------------|-------------------------------------|
| Logistic Regression                 | **Highly reliable linear baseline.** It achieves joint-best accuracy (0.9649), the highest recall (0.9286), and the highest F1 score (0.9512). This shows strong malignant-case detection with only a small tradeoff in precision compared with Random Forest. Its simplicity also makes the model easier to explain. |
| Decision Tree Classifier            | **Most interpretable but weakest general performer.** The model gives readable decision rules, but its lower AUC (0.9246) and MCC (0.8492) show weaker class separation than the other models. This suggests a single tree is more sensitive to dataset splits and may not generalize as well as ensemble methods. |
| K-Nearest Neighbor Classifier (kNN) | **Competitive distance-based model.** After scaling, kNN performs strongly with 0.9561 accuracy, 0.9383 F1, and 0.9058 MCC. However, it remains slightly behind Logistic Regression and Random Forest, and its performance depends heavily on feature scaling and the selected value of k. |
| Naive Bayes Classifier (GaussianNB) | **Good probabilistic baseline with recall limitation.** Its AUC (0.9891) is high, meaning it ranks benign and malignant cases well, but it has the lowest recall (0.8571) and MCC (0.8292). For a cancer diagnosis task, lower recall is a concern because missed malignant cases are more serious. |
| Random Forest (Ensemble)            | **Best overall model with ensemble robustness.** It has the highest AUC (0.9970), perfect precision (1.0000), highest MCC (0.9258), and joint-best accuracy (0.9649). The ensemble of decision trees reduces overfitting compared with a single Decision Tree and gives the most stable overall performance. |
| Overall Winner for this dataset     | **Random Forest (Ensemble)** is the best performing model for this dataset because it leads in AUC, precision, and MCC while matching the top accuracy. Logistic Regression is the closest alternative because it gives slightly better recall and F1, which is important for detecting malignant tumors. |

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
|   |-- train_models.ipynb
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest.pkl
|   |-- feature_columns.json
```

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
