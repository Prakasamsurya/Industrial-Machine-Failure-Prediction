# Industrial Machine Failure Prediction — Complete Project Explanation
Our project uses Machine Learning to predict whether an industrial machine is likely to experience a failure based on its sensor readings.

The goal is to help industries identify potential problems earlier, reduce unexpected breakdowns, and plan maintenance more effectively.
1. What is the purpose of the data, and why did we choose it?
We used the AI4I 2020 Predictive Maintenance dataset.

Imagine a factory with hundreds of machines. If a machine suddenly breaks down, production may stop, repairs may cost money, and delivery may be delayed.

Instead of waiting for a machine to break, we can use its sensor readings to train a Machine Learning model to identify patterns associated with failures.

Why we chose this dataset:
- It contains 10,000 records, making it manageable for learning.
- It includes machine measurements such as temperature, rotational speed, torque, and tool wear.
- It contains machine failure labels, so we can train a supervised classification model.
- It allows us to compare different ML algorithms and explore predictive maintenance.
2. What is inside the dataset?
Our dataset has 10,000 rows and 14 columns.

Each row represents one observation of a machine, and each column describes an identifier, machine characteristic, measurement, or failure label.
Column	Meaning
UDI	Unique row identifier
Product ID	Machine/product identifier
Type	Product type: H, L, or M
Air temperature [K]	Air temperature in Kelvin
Process temperature [K]	Process temperature in Kelvin
Rotational speed [rpm]	Machine rotation speed
Torque [Nm]	Rotational force
Tool wear [min]	Accumulated tool wear time
Machine failure	Target: 0 = no failure, 1 = failure
TWF	Tool wear failure indicator
HDF	Heat dissipation failure indicator
PWF	Power failure indicator
OSF	Overstrain failure indicator
RNF	Random failure indicator

The five measurements plus product type became our six input features. We excluded the identifiers and failure-mode indicators from the model inputs.

Our target distribution was:
No machine failure (0)
9,661



Machine failure (1)
339



Only 3.39% of observations contain a machine failure. This is an imbalanced dataset.

This imbalance matters because a model could predict “no failure” for almost every row and still achieve high accuracy while missing actual failures.
3. Why do we clean data before Machine Learning?
Data cleaning and preprocessing help us prepare data in a format that a model can use reliably.

We performed several checks and transformations:
- Missing values: We checked whether values were missing. Every column had 10,000 non-null values.
- Duplicate records: We checked for duplicate rows and found zero.
- Data types: We inspected numerical and categorical columns.
- Feature selection: We removed identifiers and failure-mode indicators to prevent irrelevant inputs and target leakage.
- Categorical encoding: We converted Type into three binary columns: Type_H, Type_L, and Type_M.
- Train-test splitting: We separated 80% of the records for training and 20% for testing, using stratification to preserve the failure proportion.
We did not need to remove missing or duplicate rows because none were found.

We also used StandardScaler for KNN and SVM because these models are sensitive to differences in feature scales. Tree-based models such as XGBoost do not generally require this scaling.

Important: Cleaning does not guarantee a perfect model. It reduces data-related problems and helps make training and evaluation more reliable.
4. Which ML models did we use, and what were the results?
We compared ten model configurations. Because failure is the minority class, we focused on the F1-score for class 1 (machine failure) rather than accuracy alone.
Model	Precision	Recall	F1-score	Accuracy
XGBoost (original)	0.904	0.691	0.783	0.987
XGBoost (tuned)	0.887	0.691	0.777	0.986
Decision Tree (tuned)	0.712	0.691	0.701	0.980
Random Forest (original)	0.698	0.647	0.672	0.978
Random Forest (tuned)	0.589	0.779	0.671	0.974
Gradient Boosting	0.324	0.868	0.472	0.934
Decision Tree (original)	0.311	0.882	0.460	0.930
KNN	0.833	0.294	0.435	0.974
SVM (balanced)	0.279	0.912	0.428	0.917
Logistic Regression (balanced)	0.144	0.824	0.245	0.827

What does each model do?

Logistic Regression

A simple classification model that estimates the probability of failure. We used it as a baseline.



Decision Tree

Makes predictions through a series of rules, such as checking whether torque or tool wear crosses certain values.



Random Forest

Combines predictions from many decision trees to produce a more robust prediction.



K-Nearest Neighbors (KNN)

Predicts based on the labels of similar training examples. Feature scaling is important.



Support Vector Machine (SVM)

Finds a boundary that separates the classes. We used class weighting to give more attention to failures.



Gradient Boosting

Builds trees sequentially, with later trees trying to correct errors made by earlier ones.



XGBoost

An optimized boosting algorithm that builds trees sequentially to improve predictions. It achieved the highest test F1-score in our comparison.



We also used balanced class weights or sample weights for selected models to address the imbalance between failure and non-failure examples.
Why did we choose XGBoost?
Our original XGBoost model achieved:
- Precision: 90.4%
- Recall: 69.1%
- F1-score: 0.783
- Accuracy: 98.7%
It provided the best F1-score among our tested configurations. However, the results are specific to this dataset and evaluation setup; further independent validation is needed before real-world use.
5. Why did we perform hyperparameter tuning?
Hyperparameters are settings we choose before training a model. They control how the model learns.

For example, in XGBoost:
- n_estimators: Number of boosting trees.
- max_depth: Maximum depth of each tree.
- learning_rate: How strongly each new tree contributes to the model.
We used GridSearchCV to test combinations of these settings using five-fold stratified cross-validation. We selected the combination with the highest cross-validation F1-score.

Our best XGBoost settings were:
{    "learning_rate": 0.05,    "max_depth": 5,    "n_estimators": 200}



The tuned model achieved a cross-validation F1-score of approximately 0.714, but its test F1-score was 0.777, slightly below the original model's 0.783.

Why is tuning still useful? It helps us systematically search for better settings rather than guessing. It does not guarantee that a tuned model will outperform the original one on every test set.
6. Why do we evaluate the model?
Training performance alone is not enough. We need to measure how well the model predicts examples it did not train on.

We used four key metrics:
Metric	Meaning in our project
Accuracy	Overall proportion of correct predictions
Precision	Of the predicted failures, how many were actual failures?
Recall	Of the actual failures, how many did the model detect?
F1-score	A balance between precision and recall

For machine failure prediction, missing a real failure may be costly. However, too many false alarms can also lead to unnecessary inspections and maintenance.

That's why we examined the confusion matrix and the individual failure-class metrics instead of relying only on accuracy.

One limitation: we reused the test set while comparing models, so it is no longer a completely independent final evaluation. A fresh holdout set or nested validation procedure would give a more reliable estimate of future performance.
7. Why is feature engineering needed?
Feature engineering means creating, transforming, or selecting input features to help a model learn useful patterns.

In our project, we mainly performed feature selection and categorical encoding. We selected six input features and converted the Type column into binary columns.

We did not create many new mathematical features, so advanced feature engineering remains a possible improvement.

Examples of future experiments:
- Temperature difference: Process temperature - Air temperature.
- Explore interactions between torque and rotational speed.
- Compare models using different feature subsets.
- Use domain knowledge to create meaningful machine-condition indicators.
We should test whether these additions improve cross-validation performance rather than assuming they will.
8. What did SHAP tell us?
We used SHAP to explain our trained XGBoost model.

Our mean absolute SHAP results ranked the features as follows:
Rank	Feature	Mean absolute SHAP
1	Torque	1.4718
2	Tool wear	1.2717
3	Air temperature	0.7894
4	Rotational speed	0.5817
5	Process temperature	0.3558

Torque and tool wear had the greatest average impact on predictions in the 500 training examples we analyzed.

SHAP helps us understand model behavior, but it does not prove that a feature causes machine failure.
9. Final project conclusion
We built a supervised Machine Learning pipeline that predicts machine failure from sensor readings, compares seven model families across ten configurations, tunes selected models, evaluates their performance, and explains predictions using SHAP.

Our original XGBoost configuration achieved the highest test F1-score of 0.783 among the configurations tested. We also explored a threshold of 0.32 to balance precision and recall.

For the threshold-adjusted test predictions, the model detected 52 of 68 actual failures, produced 13 false alarms, and missed 16 failures. These results show why evaluating recall and false alarms is important for predictive maintenance.

The next step toward a more reliable project is independent validation, followed by testing whether additional feature engineering improves performance.
