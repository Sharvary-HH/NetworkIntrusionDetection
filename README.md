#NSL-KDD Intrusion Detection System
This repository implements an intrusion detection system (IDS) for the NSL-KDD dataset using K-Nearest Neighbors, Logistic Regression, Naive Bayes, and Decision Tree classifiers. The project features preprocessing, sampling to address class imbalance, feature selection, model training, evaluation, and clear visualization of attack distributions.

##Features
Clean end-to-end pipeline: Data loading, preprocessing, feature engineering, class balancing, model training, evaluation and visualization.

Multiple ML classifiers: KNN, Logistic Regression, Naive Bayes, Decision Tree.

Handles class imbalance: Uses Random Oversampling.

Feature selection: Selects top important features using Random Forest with RFE.

Train/Test split: Uses standard NSL-KDD dataset splits.

Performance reports: Accuracy, confusion matrix, and detailed classification metrics.

Attack mapping: Maps raw attack labels to broader categories (DoS, Probe, R2L, U2R, Normal).

Easy to extend.

##Project Structure
text
├── main.py           # Main code: data processing, training & evaluation
├── requirements.txt  # All Python dependencies
├── KDDTrain+.txt     # Training dataset (expected at repo root or given path)
├── KDDTest+.txt      # Test dataset (expected at repo root or given path)
###Installation
Clone the repository

text
git clone https://github.com/yourusername/NSL-KDD-IDS.git
cd NSL-KDD-IDS
Install the dependencies

Recommended: Create a new Python virtual environment.

text
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Place the NSL-KDD dataset files

Download KDDTrain+.txt and KDDTest+.txt from NSL-KDD dataset source.

Place them in the root folder, or update the paths as needed in main.py.

###Requirements
The main imports and Python packages you need (see requirements.txt):

text
pandas>=1.5.0
numpy>=1.24.0
matplotlib
seaborn
scikit-learn>=1.3.0
imbalanced-learn
How to Run
After setting up, execute the main script. It will load data, preprocess, train all models, perform evaluation and print results.

text
python main.py
Update data file paths in main.py if your dataset location is different:

python
train_file_path = "path/to/KDDTrain+.txt"
test_file_path = "path/to/KDDTest+.txt"
The script prints progress, stats, and metrics to the console, and shows attack distribution plots.

##Detailed Pipeline
Data Loading & Preprocessing: Loads NSL-KDD training and test sets, removes unused columns, checks missing values.

Attack Class Mapping: Maps detailed attack types into five broad classes (DoS, Probe, R2L, U2R, Normal).

Feature Scaling & Encoding: Scales numerical features, applies label and one-hot encoding to categorical features.

Resampling: Uses random oversampling to handle class imbalance in the training set.

Feature Selection: Selects top 20 most important features using Recursive Feature Elimination (RFE) with Random Forest.

Model Training: Trains four models: KNN, Logistic Regression, Naive Bayes, Decision Tree.

Evaluation: For each model: prints training/test accuracy, cross-validation mean, confusion matrix, and classification report.

Visualization: Displays bar chart comparing attack type distribution in train/test sets.

##Extending the Project
Add new classifiers or feature engineering techniques.

Integrate different balancing or selection methods from scikit-learn/imbalanced-learn.

Tune model hyperparameters for better results.

Experiment with more visualizations.

##Troubleshooting
If data files are missing, download from official [NSL-KDD source].

Edit file paths in main.py as needed.

Install missing Python packages via pip.

##License
MIT License. See LICENSE.md for details.