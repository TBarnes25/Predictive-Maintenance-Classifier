# Predictive Maintenance Classifier

A machine learning project predicting machine failure from sensor readings 
(air/process temperature, rotational speed, torque, tool wear)

## Project goals
- Explore and understand a real-world-style industrial sensor dataset
- Handle severe class imbalance properly (not just accuracy-chasing)
- Build and compare baseline vs. stronger classifiers
- Package the pipeline cleanly using scikit-learn

**Status: In progress.** : sklearn pipeline

## Dataset & EDA
- Source: [AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository]
- shape [10,000 x 14], 3.39% failure rate (severe class imbalance)
- No missing values in the data and no obviously erroneous data
- Severe data leakage from the columns TWF/HDF/PWF/OSF/RNF which are post-hoc labels added after the failure to diagnose the failure, using them for training would make the model trivially perfect, so they were droppped from training data
- ProductID/ UID also dropped as it was a human abstraction and has no effect on machine failure.
- Verified consistency between the failure-mode flags and the Machine failure label: found 348 rows (3.48%) where they disagreed — some failures with no mode flag set, and some flags set with no recorded failure. Not investigated further given project scope.
- Types (L/M/H):
- - Affecting the OSF threshold, converted using ordinal encoding, as the types represented continuous incremental increase in the OSF threshold before machine failure
  - Tool wear accumulates faster by type L:+2, M:+3, H:+5
  - Both relationships have a near constant magnitude difference between types, so I chose ordinal encoding to preserve the relationship
- Column renaming for ease of addressing

## Feature Engineering
- Engineered features:
- - power (torque x rotational speed)
  - overstrain (torque x tool_wear (type normalised))
  - temperature_difference machine_temp - ambient_temp
- Overstrain closely recontructs the original OSF formula, domain informed engineering. However inflates the performance on OSF failures 


## Evaluation Strategy
-Accuracy is an ineffective metric as the model could predict no failure and have an accuracy of 96.6%, but not prevent any damage or harm caused by machine failure thus a different metric is more appropriate.
-Primary metric using F-beta specifically F2 to weight recall more than precision as broken machines have a higher cost than an unnescessary technician visit.
- Secondary metric PR-auc, represents precision/recall statistics across every range of thresholds (0 to 1) rather than committing to a specific threshold like f2 (0.5).
- Rejected ROC-AUC as it uses true negatives in the denominator fp/(fp+tn)  due to the class imbalance the metric becomes overly optimistic (near 1)
- Stratified train/test split - testing and training data each contain 3.39% failure cases
- StratifiedkFold (k=5) divide training data into 5 quanta, train on 4 quanta test on the 5th
  evaluate the chosen metrics for each variation. This gives a more unbiased view of the models performance reducing the dependency of the results on the training set.


##Modelling & Results:
-Baseline: Dummyclassifier, (F2 = 0), truest baseline, emphasing the metrics the models will be evaluated by 
-Logistic Regression Unbalanced (F2 = 0.237)
-Logistic Regression Balanced (F2 = 0.487) Balancing increase the probability of predicting a failure as the errors have been made artificially more expensive

-Observation: Although f2 increased from Unbalanced -> Balanced, PR-Auc decreased (0.495 -> 0.439) .
Balancing reweights the loss function to penalise false negatives more heavily, shifting predocted probabilties upwards for the minority class. This improved the performance at the 0.5 threshold specifically, but slightly reduces hwo well the model ranks true positives above true negatives across all thresholds. Tracking both metrics revealed this. 


Random Forest:
Chosen over logistic regression tuning as the engineered features (overstrain and power) represent threshold-style relationships multiplicative relationships, (failure may be caused by a combination of high power and torque), a tree based model is better suited for catching the relationships without manually engineering the interactions.

-Random Forest Balanced Subsample n_estimators = 300 (f2 = 0.789) balanced as recall is more important than precision 

-Random Forest (class weight 1:20) n_estimators = 300 (f2 = 0.815) penalising false negatives more heavily improved the f2 score

-Feature importance check: A precision of 1.0 on the 1:20 rf model raised concern that overstrain was dominating causing leaking. However this was disproved, overstrain 0.121, is the medina feature importance.
-The manually-weighted (1:20) configuration was carried forward as the primary model for threshold tuning and final cross-validated evaluation below, as it produced the stronger F2.

## False Negative/Failure-Mode Analysis
-To investigate the false negatives added the flag columns back
- 9/14 of the false negatives had flag TWF, it is documented as having a randomly assigned outcome even within the danger window (200/240)
- 2/14 were flagless, the inconsistency spotted in EDA
  -Conclusion: the recall has a ceiling due to irreducible random nature of TWF feature, there is no guarantee more complexity would improve performance

  
##Threshold Tuning
- Swept thresholds (probability of machine failure) from 0.1 to 0.5 in 0.05 increments
- Chosen threshold 0.4, F2=0.828, precision=1, recall=0.794 - selected as it had the highest f2 metric, which was the primary metric from the evaluation strategy
-Threshold selected using testing data not a subset of training data


##Final Cross-Validated Result
- Used statifiedkfold k=5 on training data. F2 mean of 0.79, range 0.71-0.86. F2 sd 0.0515
- A more useful metric then the intiial  0.828 as it removes bias from selecting specific failure cases to train on.
  




