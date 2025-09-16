import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import datetime
import json

def load_data(datasets=['OMXS30', 'SP500']):
    """
    Load training, validation, and test datasets.
    
    Parameters:
    datasets (list): List of datasets to load. Options: ['OMXS30', 'SP500'] or ['OMXS30'] or ['SP500']
    
    Returns:
    tuple: (train_df, val_df, test_df)
    """
    print(f"Loading datasets: {datasets}...")
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for dataset in datasets:
        # Load the split datasets for each market
        train_df = pd.read_csv(f'training/{dataset}_train.csv')
        val_df = pd.read_csv(f'training/{dataset}_val.csv')
        test_df = pd.read_csv(f'test/{dataset}_test.csv')
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        test_dfs.append(test_df)
        
        print(f"{dataset} - Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    # Combine datasets if multiple are provided
    if len(datasets) > 1:
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"\nCombined datasets:")
    else:
        train_df = train_dfs[0]
        val_df = val_dfs[0]
        test_df = test_dfs[0]
    
    print(f"Final Training set: {train_df.shape}")
    print(f"Final Validation set: {val_df.shape}")
    print(f"Final Test set: {test_df.shape}")
    
    # Print target distribution
    train_target_dist = train_df['Target'].value_counts().sort_index()
    print(f"Target distribution - Class 0: {train_target_dist[0]}, Class 1: {train_target_dist[1]} (ratio: {train_target_dist[0]/train_target_dist[1]:.2f}:1)")
    
    return train_df, val_df, test_df

def prepare_features(df, selected_features=None, drop_columns=None):
    """
    Prepare features and target variables from the dataset.
    
    Parameters:
    df (DataFrame): Input dataset
    selected_features (list): Specific features to use (if None, use all except drop_columns)
    drop_columns (list): Columns to exclude from features
    
    Returns:
    tuple: (X, y) where X is features and y is target
    """
    if drop_columns is None:
        drop_columns = ['Date', 'Target']
    
    # Separate features and target
    if selected_features is not None:
        X = df[selected_features]
    else:
        X = df.drop(columns=drop_columns)
    
    y = df['Target']
    
    # Remove rows with NaN values (typically first few rows due to technical indicators)
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
    
    return X, y


def compute_class_weights(y_train):
    """
    Compute class weights for multiclass imbalanced dataset.
    
    Parameters:
    y_train (Series): Training targets
    
    Returns:
    dict: Class weights for each class
    """
    # Analyze class distribution
    class_counts = y_train.value_counts().sort_index()
    total_samples = len(y_train)
    n_classes = len(class_counts)
    
    # Compute balanced class weights
    class_weights = {}
    for class_label in class_counts.index:
        class_weights[class_label] = total_samples / (n_classes * class_counts[class_label])
    
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Class weights: {class_weights}")
    
    return class_weights

def train_xgboost(X_train, y_train, class_weights=None, **xgb_params):
    """
    Train an XGBoost classifier.
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training targets
    class_weights (dict or str): Class weights to handle imbalanced data
    **xgb_params: XGBoost parameters
    
    Returns:
    XGBClassifier: Trained model
    """
    # Default parameters for XGBoost
    default_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 1.0,
        'min_child_weight': 50,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',  # Changed to mlogloss for multiclass
        'objective': 'multi:softprob',  # Changed to softprob for multiclass
        'num_class': 3  # Added for 3-class classification (0=sell, 1=hold, 2=buy)
    }
    
    # Handle class imbalance (for multiclass, we'll use class_weight parameter)
    if class_weights is not None and isinstance(class_weights, dict):
        # For multiclass, XGBoost doesn't use scale_pos_weight
        # Instead, we can handle imbalance through sample_weight during fit
        print("Note: Class weights will be handled through sample weighting for multiclass")
    
    # Update with any provided parameters
    default_params.update(xgb_params)
    
    # Create and train the model
    xgb_model = XGBClassifier(**default_params)
    
    # Handle class imbalance through sample weights if provided
    if class_weights is not None and isinstance(class_weights, dict):
        sample_weights = [class_weights.get(label, 1.0) for label in y_train]
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        xgb_model.fit(X_train, y_train)
    
    print(f"Model trained for multiclass classification (3 classes)")
    
    return xgb_model

def evaluate_model(model, X, y, dataset_name="Dataset"):
    """
    Evaluate the model and print key metrics.
    
    Parameters:
    model: Trained model
    X (DataFrame): Features
    y (Series): True targets
    dataset_name (str): Name of the dataset for reporting
    
    Returns:
    dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)  # All class probabilities for multiclass
    
    # Calculate metrics (using 'weighted' average for multiclass)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    
    print(f"{dataset_name}: Acc={accuracy:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance from the trained XGBoost model.
    
    Parameters:
    model: Trained XGBClassifier
    feature_names (list): List of feature names
    top_n (int): Number of top features to display
    """
    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def log_training_results(model, metrics, selected_features, class_weights, model_filename, datasets_used=['OMXS30', 'SP500']):
    """
    Log training results to log.json file.
    
    Parameters:
    model: Trained XGBClassifier
    metrics (dict): Dictionary containing train/val/test metrics
    selected_features (list): List of features used in training
    class_weights (dict): Class weights used in training
    model_filename (str): Name of the saved model file
    datasets_used (list): List of datasets used for training
    """
    # Create log entry
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': 'XGBoost',
        'model_filename': model_filename,
        'datasets_used': datasets_used,
        'parameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'learning_rate': model.learning_rate,
            'subsample': model.subsample,
            'colsample_bytree': model.colsample_bytree,
            'colsample_bylevel': model.colsample_bylevel,
            'min_child_weight': model.min_child_weight,
            'reg_alpha': model.reg_alpha,
            'reg_lambda': model.reg_lambda,
            'random_state': model.random_state,
            'eval_metric': model.eval_metric,
            'objective': model.objective,
            'num_class': getattr(model, 'num_class', 3)  # Default to 3 for multiclass
        },
        'features': {
            'selected_features': selected_features,
            'feature_count': len(selected_features)
        },
        'metrics': {
            'training_set': {
                'accuracy': round(metrics['train']['accuracy'], 4),
                'precision': round(metrics['train']['precision'], 4),
                'recall': round(metrics['train']['recall'], 4),
                'f1_score': round(metrics['train']['f1_score'], 4)
            },
            'validation_set': {
                'accuracy': round(metrics['val']['accuracy'], 4),
                'precision': round(metrics['val']['precision'], 4),
                'recall': round(metrics['val']['recall'], 4),
                'f1_score': round(metrics['val']['f1_score'], 4)
            },
            'test_set': {
                'accuracy': round(metrics['test']['accuracy'], 4),
                'precision': round(metrics['test']['precision'], 4),
                'recall': round(metrics['test']['recall'], 4),
                'f1_score': round(metrics['test']['f1_score'], 4)
            }
        }
    }
    
    # Load existing log or create new one
    log_file = 'log.json'
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
    except (json.JSONDecodeError, FileNotFoundError):
        logs = []
    
    # Add new entry
    logs.append(log_entry)
    
    # Save updated log
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n✅ Training results logged to '{log_file}'")
    print(f"📊 Log entry #{len(logs)} saved with timestamp: {log_entry['timestamp']}")
    print(f"📈 Datasets used: {', '.join(datasets_used)}")

def main():
    """
    Main function to train the optimal XGBoost model.
    """
    print("=" * 60)
    print("XGBoost Model Training for Stock Price Prediction")
    print("=" * 60)
    
    # Use all available features instead of a selected subset
    selected_features = None  # This will use all features except those in drop_columns
    
    
    try:
        # 1. Load data from both OMXS30 and SP500
        train_df, val_df, test_df = load_data(datasets=['OMXS30'])
        
        # 2. Prepare features and targets with all available features
        print("\nPreparing data with all available features...")
        X_train, y_train = prepare_features(train_df, selected_features=selected_features)
        X_val, y_val = prepare_features(val_df, selected_features=selected_features)
        X_test, y_test = prepare_features(test_df, selected_features=selected_features)
        
        # 3. Compute optimal class weights
        class_weights = compute_class_weights(y_train)
        
        # 4. Train the optimal model
        print("\nTraining XGBoost with optimal configuration...")
        xgb_model = train_xgboost(X_train, y_train, class_weights=class_weights)
        
        # 5. Evaluate the model
        print("\n=== Model Evaluation ===")
        train_metrics = evaluate_model(xgb_model, X_train, y_train, "Training Set")
        val_metrics = evaluate_model(xgb_model, X_val, y_val, "Validation Set")
        test_metrics = evaluate_model(xgb_model, X_test, y_test, "Test Set")
        
        # 6. Show feature importance
        print("\n=== Feature Importance ===")
        feature_importance = plot_feature_importance(xgb_model, X_train.columns, top_n=15)  # Show top 15 features
        
        # Get the actual features used for logging
        actual_features_used = list(X_train.columns)
        
        # 7. Save the model
        model_filename = f'xgboost_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        joblib.dump(xgb_model, "models/" + model_filename)
        print(f"\n✅ Model saved as '{model_filename}'")
        
        # 8. Log training results
        print(f"\n=== LOGGING RESULTS ===")
        log_training_results(xgb_model, {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }, actual_features_used, class_weights, model_filename, datasets_used=['OMXS30'])
        
        # 9. Summary
        print(f"\n=== FINAL MODEL SUMMARY ===")
        print(f"Model Type: XGBoost")
        print(f"Features: {len(actual_features_used)}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        
        return {
            'model': xgb_model,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'feature_importance': feature_importance,
            'selected_features': actual_features_used,  # Now contains all features
            'class_weights': class_weights
        }
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
