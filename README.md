# NSL-KDD Intrusion Detection System

This repository implements a **Network Intrusion Detection System (IDS)** using the **NSL-KDD dataset**.  
The system applies **K-Nearest Neighbors, Logistic Regression, Naive Bayes, and Decision Tree** classifiers.  

It includes a complete **data pipeline**: preprocessing, class balancing, feature selection, model training, evaluation, and visualization of attack distributions.

---

## 🚀 Features

- **Clean end-to-end pipeline**  
  Data loading → preprocessing → feature engineering → class balancing → model training → evaluation → visualization.

- **Multiple ML classifiers**  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Naive Bayes  
  - Decision Tree  

- **Handles class imbalance**  
  Uses **Random Oversampling** to improve minority class detection.  

- **Feature selection**  
  Uses **Random Forest with Recursive Feature Elimination (RFE)** to select top important features.  

- **Attack mapping**  
  Maps raw attack labels to broader categories:  
  - **DoS**, **Probe**, **R2L**, **U2R**, **Normal**  

- **Performance reports**  
  Accuracy, confusion matrix, classification report (precision, recall, F1-score).  

- **Easy to extend**  
  Add new models, balancing strategies, or visualization methods.  

---

## 📂 Project Structure

├── main.py # Main code: data processing, training & evaluation
├── requirements.txt # Python dependencies
├── KDDTrain+.txt # Training dataset (expected in repo root)
├── KDDTest+.txt # Test dataset (expected in repo root)


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/NSL-KDD-IDS.git
cd NSL-KDD-IDS

```
Create and activate a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Download the dataset:

NSL-KDD Dataset Source

Place KDDTrain+.txt and KDDTest+.txt in the root folder (or update paths in main.py).

📦 Requirements

Main dependencies (see requirements.txt):

pandas>=1.5.0
numpy>=1.24.0
matplotlib
seaborn
scikit-learn>=1.3.0
imbalanced-learn

▶️ How to Run

Run the main script:

python main.py


If your dataset files are in a different location, update paths in main.py:

train_file_path = "path/to/KDDTrain+.txt"
test_file_path = "path/to/KDDTest+.txt"


The script will:

Load and preprocess the dataset

Train all models

Print evaluation metrics

Show attack distribution plots

🔎 Detailed Pipeline

Data Loading & Preprocessing

Load train/test sets

Remove unused columns (num_outbound_cmds)

Handle missing values

Attack Class Mapping

Convert raw attack names into five categories: DoS, Probe, R2L, U2R, Normal

Feature Scaling & Encoding

Standard scaling for numeric features

Label encoding / One-hot encoding for categorical features

Resampling

Apply random oversampling to balance attack classes

Feature Selection

Select top 20 features using Random Forest + RFE

Model Training

Train KNN, Logistic Regression, Naive Bayes, Decision Tree

Evaluation

Accuracy, cross-validation mean score

Confusion matrix

Classification report (precision, recall, F1)

Visualization

Bar plots of attack class distribution (train vs test)

🔧 Extending the Project

Add new ML models (SVM, Random Forest, XGBoost, etc.)

Experiment with feature engineering techniques

Try different balancing methods from imbalanced-learn

Perform hyperparameter tuning

Add advanced visualizations

🛠️ Troubleshooting

Missing dataset → Download from NSL-KDD source

Wrong file path → Update dataset path in main.py

Missing packages → Run pip install -r requirements.txt

📜 License

This project is licensed under the MIT License.
See LICENSE.md
 for details.


---

Would you like me to also create a **`requirements.txt`** file and a **starter `main.py` template** so your intern can directly run the project without guessing?
