Credit Risk Resampling Analysis

# Objectives
- Implement machine learning models: Oversampling, Undersampling and Combination (Over and Under) Sampling.
- Use resampling to attempt to address class imbalance.
- Evaluate the performance of machine learning models.


# Resources
- Data Sourse: LoanStats_2019Q1.csv
- Software: Python 3.6.9:: Anaconda, Inc., Jupyter Notebook, 6.0.2, Visual Studio Code, 1.40.2. 

# Analysis
- Credit Risk Resampling
	- Imported the data source using pandas.
	- The counts for low credit risk is 68470 and high credit risk is 347. 
 	- Splited the data into Training and Testing samples
	- Oversampling
		- Naive Random Oversampling
			- Used RandomOverSampler to resample the data
			- Used Logistic Regression to create the model and fit the resamplted data into the model
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| RandomOverSampling | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 77    | FALSE NEGATIVE: 24   |
				| Actually False     | FALSE POSITIVE: 6984 | TRUE NEGATIVE: 10120 |

			- The accuracy score for RandomOverSampling model is 0.6770253498689438.
				- this model may not be best to model to predict credit risk because the model's accuracy, 0.677, is low, and the precision and recall are not good enough to state that the model will be good at classifying credit risk.
				- the model was correct 67.7% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.01 | 0.76 | 0.59 | 0.02 | 0.67 | 0.46 | 101   |
				| low_risk              | 1.00 | 0.59 | 0.76 | 0.74 | 0.67 | 0.44 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.59 | 0.76 | 0.74 | 0.67 | 0.44 | 17205 |

			- The precision for prediction of the high_risk and low_risk are not in line with each other. 
			- The recall for predicting low_risk is much lower than it is for predicting credit high_risk.
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk compared to low_risk, however low_risk average (0.74) is closer to 1.0.

		- SMOTE Oversampling
			- Used SMOTE to resample the data
			- Used Logistic Regression to create the model and fit the resamplted data into the model
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| SMOTE OverSampling | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 64    | FALSE NEGATIVE: 37   |
				| Actually False     | FALSE POSITIVE: 5284 | TRUE NEGATIVE: 11820 |

			- The accuracy score for SMOTE model is 0.6623648917744909
				- this model may not be best to model to predict credit risk because the model's accuracy, 0.662, is low, and the precision and recall are not good enough to state that the model will be good at classifying credit risk.
				- the model was correct 66.2% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.01 | 0.63 | 0.69 | 0.02 | 0.66 | 0.44 | 101   |
				| low_risk              | 1.00 | 0.69 | 0.63 | 0.82 | 0.66 | 0.44 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.69 | 0.63 | 0.81 | 0.66 | 0.44 | 17205 |

			- The precision for prediction of the high_risk and low_risk are not in line with each other. 
			- The recall for predicting low_risk and high_risk are in line. 
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk compared to low_risk, however low_risk average (0.82) is closer to 1.0.

	- Undersampling
		- Cluster Centroids
			- Used Cluster Centroids algorithm to undersample the data.
			- Used Logistic Regression to create the model and fit the resamplted data into the model
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| Cluster Centroids  | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 67    | FALSE NEGATIVE: 34   |
				| Actually False     | FALSE POSITIVE: 10217| TRUE NEGATIVE: 6887  |

			- The accuracy score for Cluster Centroids model is 0.6623648917744909
				- this model may not be best to model to predict credit risk because the model's accuracy, 0.662, is low, and the precision and recall are not good enough to state that the model will be good at classifying credit risk.
				- the model was correct 66.2% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.01 | 0.66 | 0.40 | 0.01 | 0.52 | 0.27 | 101   |
				| low_risk              | 1.00 | 0.40 | 0.66 | 0.57 | 0.52 | 0.26 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.40 | 0.66 | 0.57 | 0.52 | 0.26 | 17205 |

			- The precision for prediction of the high_risk and low_risk are not in line with each other. 
			- The recall for predicting low_risk is much lower than it is for predicting credit high_risk.
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk and low average for low_risk.

	- Combination (Over and Under) Sampling
		- SMOTEENN
			- Used SMOTEENN algorithm to resample the data
			- Used Logistic Regression to create the model and fit the resamplted data into the model
			- Used confusion matrix to display table of true positives, false positives, true negatives, and false negatives.

				| Cluster Centroids  | Predicted True       | Predicted False      |
				|--------------------|----------------------|----------------------|
				| Actually True      | TRUE POSITIVE: 71    | FALSE NEGATIVE: 30   |
				| Actually False     | FALSE POSITIVE: 6929 | TRUE NEGATIVE: 10175 |

			- The accuracy score for SMOTEENN model is 0.5330103432466726
				- this model may not be best to model to predict credit risk because the model's accuracy, 0.533, is low, and the precision and recall are not good enough to state that the model will be good at classifying credit risk.
				- the model was correct 53.3% of the time.
			- The classification report is:

				| classification report | pre  | rec  | spe  | f1   | geo  | iba  | sup   |
				|-----------------------|------|------|------|------|------|------|-------|
				|                       |      |      |      |      |      |      |       |
				| high_risk             | 0.01 | 0.70 | 0.59 | 0.02 | 0.65 | 0.42 | 101   |
				| low_risk              | 1.00 | 0.59 | 0.70 | 0.75 | 0.65 | 0.41 | 17104 |
				|                       |      |      |      |      |      |      |       |
				| avg / total           | 0.99 | 0.60 | 0.70 | 0.74 | 0.65 | 0.41 | 17205 |

			- The precision for prediction of the high_risk and low_risk are not in line with each other.
			- The recall for predicting low_risk is much lower than it is for predicting credit high_risk.
			- The f1 score, which is a weighted average of the true positive rate (recall) and precision, show very low average for high_risk and low average for low_risk.

