import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    Compute manual ratio-based class weights for imbalanced dataset.
    
    Parameters:
    y_train (Series): Training targets
    
    Returns:
    dict: Manual ratio-based class weights
    """
    # Analyze class distribution
    class_counts = y_train.value_counts().sort_index()
    class_0_count = class_counts[0]
    class_1_count = class_counts[1]
    
    # Manual ratio-based weighting for minority class
    manual_weights = {
        0: 1.0,  # Majority class baseline
        1: class_0_count / class_1_count  # Ratio-based weight
    }
    
    print(f"Class weights: {manual_weights}")
    
    return manual_weights

def train_random_forest(X_train, y_train, class_weights=None, **rf_params):
    """
    Train a Random Forest classifier.
    
    Parameters:
    X_train (DataFrame): Training features
    y_train (Series): Training targets
    class_weights (dict or str): Class weights to handle imbalanced data
    **rf_params: Random Forest parameters
    
    Returns:
    RandomForestClassifier: Trained model
    """
    # Default parameters for Random Forest
    default_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 100,
        'min_samples_leaf': 50,
        'max_features': 0.5,
        'max_samples': 0.8,
        'bootstrap': True,
        'oob_score': True,
        # 'class_weight': class_weights if class_weights is not None else 'balanced',
        'class_weight': {0: 1.0, 1: 4.0},
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Update with any provided parameters
    default_params.update(rf_params)
    
    # Create and train the model
    rf_model = RandomForestClassifier(**default_params)
    rf_model.fit(X_train, y_train)
    
    print(f"Model trained. OOB Score: {rf_model.oob_score_:.4f}")
    
    return rf_model

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
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
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
    Plot feature importance from the trained Random Forest model.
    
    Parameters:
    model: Trained RandomForestClassifier
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
    plt.title(f'Top {top_n} Feature Importances - Random Forest')
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
    model: Trained RandomForestClassifier
    metrics (dict): Dictionary containing train/val/test metrics
    selected_features (list): List of features used in training
    class_weights (dict): Class weights used in training
    model_filename (str): Name of the saved model file
    datasets_used (list): List of datasets used for training
    """
    # Create log entry
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_filename': model_filename,
        'datasets_used': datasets_used,
        'parameters': {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'max_features': model.max_features,
            'max_samples': model.max_samples,
            'bootstrap': model.bootstrap,
            'class_weight': dict(model.class_weight) if hasattr(model.class_weight, 'items') else model.class_weight,
            'random_state': model.random_state,
            'oob_score': model.oob_score_
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
    Main function to train the optimal Random Forest model.
    """
    print("=" * 60)
    print("Random Forest Model Training for Stock Price Prediction")
    print("=" * 60)
    
    # Define best performing features
    selected_features = [
        'Return_1d',        # Most important
        'MACD_histogram',   # Important and unique
        'RSI_14',          # Important momentum indicator  
        'BB_width',        # Volatility measure
        'Volume',          # Keep volume - it helps despite missing data
        'SMA_5',
        'Rolling_Std_20',   # Volatility
        
        # Enhanced momentum features
        # 'Return_5d_lag',    # 5-day momentum (lagged)
        # 'Return_10d_lag',   # 10-day momentum (lagged)
        
        # # Enhanced volatility features
        
        # # Enhanced trend features
        # 'SMA_ratio',        # Price relative to SMA
        # 'Price_change_20d'  # 20-day price change
    ]
    
    try:
        # 1. Load data from both OMXS30 and SP500
        train_df, val_df, test_df = load_data(datasets=['OMXS30', 'SP500'])
        
        # 2. Prepare features and targets with selected features
        print("\nPreparing data with selected features...")
        X_train, y_train = prepare_features(train_df, selected_features=selected_features)
        X_val, y_val = prepare_features(val_df, selected_features=selected_features)
        X_test, y_test = prepare_features(test_df, selected_features=selected_features)
        
        # 3. Compute optimal class weights
        class_weights = compute_class_weights(y_train)
        
        # 4. Train the optimal model
        print("\nTraining Random Forest with optimal configuration...")
        rf_model = train_random_forest(X_train, y_train, class_weights=class_weights)
        
        # 5. Evaluate the model
        print("\n=== Model Evaluation ===")
        train_metrics = evaluate_model(rf_model, X_train, y_train, "Training Set")
        val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation Set")
        test_metrics = evaluate_model(rf_model, X_test, y_test, "Test Set")
        
        # 6. Show feature importance
        print("\n=== Feature Importance ===")
        feature_importance = plot_feature_importance(rf_model, X_train.columns, top_n=len(selected_features))
        
        # 7. Save the model
        model_filename = f'random_forest_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        joblib.dump(rf_model, "models/" + model_filename)
        print(f"\n✅ Model saved as '{model_filename}'")
        
        # 8. Log training results
        print(f"\n=== LOGGING RESULTS ===")
        log_training_results(rf_model, {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }, selected_features, class_weights, model_filename, datasets_used=['OMXS30', 'SP500'])
        
        # 9. Summary
        print(f"\n=== FINAL MODEL SUMMARY ===")
        print(f"Features: {len(selected_features)}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        
        return {
            'model': rf_model,
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'feature_importance': feature_importance,
            'selected_features': selected_features,
            'class_weights': class_weights
        }
        
    except Exception as e:
        print(f"\n❌ Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
