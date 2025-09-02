# NSL-KDD Intrusion Detection System
# Clean implementation with KNN, Logistic Regression, Naive Bayes, and Decision Tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=== NSL-KDD Intrusion Detection System ===")
print("Loading and preprocessing data...\n")

# Define dataset field names
datacols = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "last_flag"
]

# Function to load and preprocess data
def load_and_preprocess_data(train_file_path="KDDTrain+.txt", test_file_path="KDDTest+.txt"):
    """Load and preprocess NSL-KDD dataset from actual files"""
    
    try:
        print("Loading NSL-KDD dataset files...")
        
        # Load training dataset
        print(f"Loading training data from: {train_file_path}")
        dfkdd_train = pd.read_csv(train_file_path, names=datacols, header=None)
        
        # Load test dataset  
        print(f"Loading test data from: {test_file_path}")
        dfkdd_test = pd.read_csv(test_file_path, names=datacols, header=None)
        
        # Remove the last unnecessary column if it exists
        if 'last_flag' in dfkdd_train.columns:
            dfkdd_train = dfkdd_train.drop(['last_flag'], axis=1)
        if 'last_flag' in dfkdd_test.columns:
            dfkdd_test = dfkdd_test.drop(['last_flag'], axis=1)
            
        # Remove 'num_outbound_cmds' if it exists (usually all zeros)
        if 'num_outbound_cmds' in dfkdd_train.columns:
            dfkdd_train = dfkdd_train.drop(['num_outbound_cmds'], axis=1)
        if 'num_outbound_cmds' in dfkdd_test.columns:
            dfkdd_test = dfkdd_test.drop(['num_outbound_cmds'], axis=1)
            
        print(f"Successfully loaded datasets!")
        print(f"Training set shape: {dfkdd_train.shape}")
        print(f"Test set shape: {dfkdd_test.shape}")
        
        # Display basic info about the datasets
        print("\nTraining data sample:")
        print(dfkdd_train.head(3))
        
        print("\nTest data sample:")
        print(dfkdd_test.head(3))
        
        # Check for missing values
        print(f"\nMissing values in training set: {dfkdd_train.isnull().sum().sum()}")
        print(f"Missing values in test set: {dfkdd_test.isnull().sum().sum()}")
        
        return dfkdd_train, dfkdd_test
        
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. {e}")
        print("Please make sure the NSL-KDD dataset files are in the correct location.")
        print("Expected files: KDDTrain+.txt, KDDTest+.txt")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def map_attack_classes(df):
    """Map specific attacks to broader categories"""
    mapping = {
        'ipsweep': 'Probe', 'satan': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'saint': 'Probe', 'mscan': 'Probe',
        'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS', 'back': 'DoS',
        'neptune': 'DoS', 'smurf': 'DoS', 'mailbomb': 'DoS', 'udpstorm': 'DoS',
        'apache2': 'DoS', 'processtable': 'DoS',
        'perl': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R', 'buffer_overflow': 'U2R',
        'xterm': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'httptunnel': 'U2R',
        'ftp_write': 'R2L', 'phf': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L',
        'warezclient': 'R2L', 'imap': 'R2L', 'spy': 'R2L', 'multihop': 'R2L',
        'named': 'R2L', 'snmpguess': 'R2L', 'worm': 'R2L', 'snmpgetattack': 'R2L',
        'xsnoop': 'R2L', 'xlock': 'R2L', 'sendmail': 'R2L',
        'normal': 'Normal'
    }
    
    df['attack_class'] = df['attack'].map(mapping)
    df = df.drop(['attack'], axis=1)
    return df

def preprocess_features(train_df, test_df):
    """Preprocess numerical and categorical features"""
    
    # Separate numerical and categorical features
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_df.select_dtypes(include=['object']).columns
    
    # Remove 'attack_class' from categorical features for encoding
    categorical_features = categorical_features.drop('attack_class')
    
    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Scale numerical features
    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_df[numerical_features])
    test_num_scaled = scaler.transform(test_df[numerical_features])
    
    # Encode categorical features
    # First, use LabelEncoder for categorical features
    label_encoders = {}
    train_cat_encoded = train_df[categorical_features].copy()
    test_cat_encoded = test_df[categorical_features].copy()
    
    for col in categorical_features:
        le = LabelEncoder()
        # Fit on combined data to ensure consistent encoding
        combined_values = pd.concat([train_df[col], test_df[col]])
        le.fit(combined_values)
        
        train_cat_encoded[col] = le.transform(train_df[col])
        test_cat_encoded[col] = le.transform(test_df[col])
        label_encoders[col] = le
    
    # One-hot encode categorical features
    try:
        # For newer versions of scikit-learn (>=1.2)
        encoder = OneHotEncoder(sparse_output=False)
    except TypeError:
        # For older versions of scikit-learn (<1.2)
        encoder = OneHotEncoder(sparse=False)
    
    train_cat_onehot = encoder.fit_transform(train_cat_encoded)
    test_cat_onehot = encoder.transform(test_cat_encoded)
    
    # Combine numerical and categorical features
    X_train = np.concatenate([train_num_scaled, train_cat_onehot], axis=1)
    X_test = np.concatenate([test_num_scaled, test_cat_onehot], axis=1)
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['attack_class'])
    y_test = label_encoder.transform(test_df['attack_class'])
    
    print(f"Final feature shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Target shapes - y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoder

def apply_sampling(X_train, y_train):
    """Apply random oversampling to handle class imbalance"""
    print("Original class distribution:")
    print(Counter(y_train))
    
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    print("\nResampled class distribution:")
    print(Counter(y_train_resampled))
    
    return X_train_resampled, y_train_resampled

def feature_selection(X_train, y_train, n_features=20):
    """Select important features using Random Forest"""
    print(f"\nSelecting top {n_features} features...")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(rf, n_features_to_select=n_features)
    X_train_selected = rfe.fit_transform(X_train, y_train)
    
    print(f"Selected {X_train_selected.shape[1]} features")
    return rfe, X_train_selected

def train_models(X_train, y_train):
    """Train the four specified models"""
    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20)
    }
    
    trained_models = {}
    
    print("\n=== Training Models ===")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_train, y_train, X_test, y_test, label_encoder):
    """Evaluate all models on both training and test sets"""
    
    print("\n=== Model Evaluation Results ===")
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"{name} Results")
        print(f"{'='*60}")
        
        # Training set evaluation
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Test set evaluation
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Cross-Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        print(f"\nTest Set Confusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        print(f"\nTest Set Classification Report:")
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_test_pred, 
                                     target_names=class_names, 
                                     zero_division=0)
        print(report)

def visualize_attack_distribution(train_df, test_df):
    """Visualize attack class distribution"""
    
    # Calculate distributions
    train_dist = train_df['attack_class'].value_counts(normalize=True) * 100
    test_dist = test_df['attack_class'].value_counts(normalize=True) * 100
    
    # Create comparison dataframe
    dist_df = pd.DataFrame({
        'Train': train_dist,
        'Test': test_dist
    }).fillna(0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    dist_df.plot(kind='bar', ax=plt.gca())
    plt.title('Attack Class Distribution (Train vs Test)')
    plt.xlabel('Attack Class')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main execution
def main(train_file="KDDTrain+.txt", test_file="KDDTest+.txt"):
    """Main function to run the complete pipeline"""
    
    # Load data from actual files
    train_df, test_df = load_and_preprocess_data(train_file, test_file)
    
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Map attack classes
    train_df = map_attack_classes(train_df)
    test_df = map_attack_classes(test_df)
    
    # Visualize attack distribution
    print("\nAttack class distribution:")
    print(train_df['attack_class'].value_counts())
    
    # Preprocess features
    X_train, X_test, y_train, y_test, label_encoder = preprocess_features(train_df, test_df)
    
    print(f"\nFeature matrix shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Apply sampling to handle class imbalance
    X_train_resampled, y_train_resampled = apply_sampling(X_train, y_train)
    
    # Feature selection
    feature_selector, X_train_selected = feature_selection(X_train_resampled, y_train_resampled)
    X_test_selected = feature_selector.transform(X_test)
    
    # Train models
    trained_models = train_models(X_train_selected, y_train_resampled)
    
    # Evaluate models
    evaluate_models(trained_models, X_train_selected, y_train_resampled, 
                   X_test_selected, y_test, label_encoder)
    
    print("\n=== Analysis Complete ===")
    
    return trained_models, feature_selector, label_encoder

# Run the complete analysis
if __name__ == "__main__":
    # Update these paths to match your file locations
    train_file_path = "/datasets/KDDTrain+.txt"  # Update this path
    test_file_path = "/datasets/KDDTest+.txt"    # Update this path
    
    # If your files are in a different location, modify the paths:
    # train_file_path = "/path/to/your/KDDTrain+.txt"
    # test_file_path = "/path/to/your/KDDTest+.txt"
    
    models, feature_selector, label_encoder = main(train_file_path, test_file_path)