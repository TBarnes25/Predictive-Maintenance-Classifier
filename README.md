# Predictive Maintenance Classifier

A machine learning project predicting machine failure from sensor readings 
(air/process temperature, rotational speed, torque, tool wear)

**Status: in progress.** Currently at the EDA / feature engineering stage.

## Dataset & EDA
- Source: [AI4I 2020 Predictive Maintenance Dataset, UCI ML Repository]
- shape [10,000 x 14], 3.39% failure rate (severe class imbalance)
- No missing values in the data and no obviously erroneous data
- Severe data leakage from the columns TWF/HDF/PWF/OSF/RNF which are post-hoc labels added after the failure to diagnose the failure, using them for training would make the model trivially perfect, so they were droppped from training data
- UID also dropped as it was a human abstraction and has no affect on machine failure
- Checked consistency of reported vs actual failure rate confirmed (3.48%)
- Types (L/M/H) affecting the OSF threshold, converted using ordinal encoding 

## Project goals
- Explore and understand a real-world-style industrial sensor dataset
- Handle severe class imbalance properly (not just accuracy-chasing)
- Build and compare baseline vs. stronger classifiers
- Package the pipeline cleanly using scikit-learn
