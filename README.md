# Predictive Modeling Using Machine Learning

## Overview

This project demonstrates the application of Machine Learning techniques to predict passenger survival on the Titanic dataset. The project involves data preprocessing, model training, evaluation, and visualization of model performance using a Random Forest Classifier.

## Objective

The objective of this project is to build a predictive model that can determine whether a passenger survived based on features such as passenger class, age, gender, fare, and embarkation point.

## Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

The project uses the Titanic dataset (`train.csv`), which contains demographic and travel information about passengers aboard the Titanic.

## Data Preprocessing

The following preprocessing steps were performed:

* Handled missing values in the Age column using the median value.
* Filled missing values in the Embarked column using the mode value.
* Converted categorical variables into numerical values.
* Selected relevant features for model training.

## Machine Learning Model

Algorithm Used:

* Random Forest Classifier

Why Random Forest?

* Handles classification tasks effectively.
* Reduces overfitting through ensemble learning.
* Provides strong predictive performance on structured datasets.

## Model Training

The dataset was split into training and testing sets using an 80:20 ratio.

Training Process:

1. Feature selection
2. Data preprocessing
3. Train-test split
4. Model training using Random Forest
5. Prediction on test data

## Model Evaluation

The model performance was evaluated using:

### Accuracy Score

Measures the percentage of correctly classified instances.

### Confusion Matrix

Displays the number of correct and incorrect predictions for each class.

### ROC Curve

Illustrates the model's ability to distinguish between classes at different threshold values.

## Outputs

The following visualizations are generated and saved in the outputs folder:

* confusion_matrix.png
* roc_curve.png

## Project Structure

Predictive_Modeling_Project/

├── train.csv

├── main.py

├── README.md

├── requirements.txt

└── outputs/

```
├── confusion_matrix.png

└── roc_curve.png
```

## Key Features

* Data preprocessing and cleaning
* Feature engineering
* Supervised machine learning
* Random Forest Classification
* Model evaluation using Accuracy Score
* Confusion Matrix visualization
* ROC Curve visualization

## Results

The Random Forest model successfully predicts passenger survival based on available features. Evaluation metrics and visualizations provide insights into the model's predictive performance and classification capability.

## Conclusion

This project demonstrates the complete workflow of a supervised machine learning classification problem, including data preprocessing, model training, prediction, and performance evaluation. It provides practical experience in predictive analytics and machine learning model assessment.
