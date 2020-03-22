# Credit_Risk
Machine Learning

# Objectives
- Implement machine learning models: 
	- required resampling algorithms models: Oversampling, Undersampling and Combination (Over and Under) Sampling.
	- optional ensemble algorithm models: Balanced Random Forest Classifier and Easy Ensemble AdaBoost classifier.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.


# Resources
- Data Sourse: LoanStats_2019Q1.csv
- Software: Python 3.6.9:: Anaconda, Inc., Jupyter Notebook, 6.0.2, Visual Studio Code, 1.40.2. 

# Challenge Summary
- Credit Risk Resampling (the analysis is in credit_risk_resampling_analysis.md file)
	- Imported the data source using pandas.
	- The counts for low risk loans is 68470 and high risk loans is 347. 
 	- Splited the data into Training and Testing samples
	- Performed several algorithms using resampling algorithms models: Oversampling, Undersampling and Combination (Over and Under) Sampling.
	- Used Logistic Regression to create the model and fit the resamplted data into the model.
	- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.
	- Found the accuracy score for all the models.
	- Printed the classification report.
- Credit Risk Ensemble (the analysis is in credit_risk_ensemble_analysis.md file)
	- Imported the data source using pandas.
	- The counts for low risk loans is 68470 and high risk loans is 347. 
 	- Splited the data into Training and Testing samples
	- Performed several algorithms using Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier
	- Used Logistic Regression to create the model and fit the resamplted data into the model.
	- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.
	- Found the accuracy score for all the models.
	- Printed the classification report.
	- For the Balanced Random Forest Classifier onely, printed the feature importance sorted in descending order. 