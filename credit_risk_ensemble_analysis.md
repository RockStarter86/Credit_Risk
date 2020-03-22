Credit Risk Ensemble Analysis

# Objectives
- Implement machine learning models: Balanced Random Forest Classifier and Easy Ensemble AdaBoost classifier.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.


# Resources
- Data Sourse: LoanStats_2019Q1.csv
- Software: Python 3.6.9:: Anaconda, Inc., Jupyter Notebook, 6.0.2, Visual Studio Code, 1.40.2. 

# Analysis
- Credit Risk Ensemble
	- Imported the data source using pandas.
	- The counts for low credit risk is 68470 and high credit risk is 347. 
 	- Splited the data into Training and Testing samples.
	- Ensemble Learners
		- Balanced Random Forest Classifier
			- Used BalancedRandomForestClassifier to resample the data.
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| RandomOverSampling | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 68    | FALSE NEGATIVE: 33   |
				| Actually False     | FALSE POSITIVE: 1749 | TRUE NEGATIVE: 15355 |

			- The accuracy score for Balanced Random Forest Classifier model is 0.7855052723466922.
				- this model may not be best to model to predict credit risk because the model's accuracy, 0.785, is low, and the precision and recall are not good enough to state that the model will be good at classifying credit risk.
				- the model was correct 78.5% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.04 | 0.67 | 0.90 | 0.07 | 0.78 | 0.59 | 101   |
				| low_risk              | 1.00 | 0.90 | 0.67 | 0.95 | 0.78 | 0.62 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.90 | 0.67 | 0.94 | 0.78 | 0.62 | 17205 |

			- Sorted the model by feature importance in descending order.
			- The precision for prediction of the high_risk and low_risk are not in line with each other. 
			- The recall for predicting low_risk is much higher than it is for predicting credit high_risk.
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk compared to  low_risk, however low_risk average (0.95) is closer to 1.0.

		- Easy Ensemble AdaBoost Classifier
			- Used EasyEnsembleClassifier to resample the data.
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| RandomOverSampling | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 93    | FALSE NEGATIVE: 8    |
				| Actually False     | FALSE POSITIVE: 983  | TRUE NEGATIVE: 16121 |

			- The accuracy score for Balanced Random Forest Classifier model is 0.9316600714093861.
				- this model may be used to model to predict credit risk because the model's accuracy, 0.931, the model was correct 93.1% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.09 | 0.92 | 0.94 | 0.16 | 0.93 | 0.87 | 101   |
				| low_risk              | 1.00 | 0.94 | 0.92 | 0.97 | 0.93 | 0.87 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.94 | 0.92 | 0.97 | 0.93 | 0.87 | 17205 |

			- The precision for prediction of the high_risk and low_risk are not in line with each other. 
			- The recall for predicting low_risk and high_risk are in line with each other.
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk compared to  low_risk, however low_risk average (0.97) is closer to 1.0.
