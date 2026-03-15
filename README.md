# Credit Card Fraud
###  Project Goal
#### To build a machine learning system capable of detecting credit card fraud while minimizing the number of innocent customers whose transactions are wrongly blocked.

## The Solution: Random Forest Classifier
#### After testing multiple models (Logistic Regression, XGBoost, and Random Forest), the Random Forest emerged as the winner. It was the most robust at handling the complex, non-linear patterns of fraudulent behavior.

## Key Findings & Results
#### Highly Reliable: Our Cross-Validation showed an average ROC-AUC of 98.1%, proving the model is highly capable of distinguishing between fraud and legitimate activity across different datasets.

#### Feature Intelligence: Using Mutual Information and Feature Importance analysis, we identified that features like V14, V10, and V12 are the "smoking guns" of fraud in this dataset.

#### Strategic Balance: By adjusting the classification threshold to 0.7, we optimized the model for the real world:

#### Caught 89% of Frauds (87 out of 98).

#### Reduced False Alarms by 72%, dropping from 2,204 down to just 618.

## Final Verdict
#### The model is ready for a production environment. It provides a massive security upgrade over traditional methods while maintaining a high level of customer satisfaction by keeping "false blocks" at a manageable level.
