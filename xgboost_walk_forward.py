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
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

def get_available_datasets():
    """
    Get list of available datasets in the data directory.
    
    Returns:
    list: List of available dataset names (without .csv extension)
    """
    data_dir = 'data'
    if not os.path.exists(data_dir):
        return []
    
    datasets = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            datasets.append(file.replace('.csv', ''))
    
    return sorted(datasets)

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
    
    # Add 'Dataset' to drop_columns if it exists (from combined datasets)
    if 'Dataset' in df.columns and 'Dataset' not in drop_columns:
        drop_columns = drop_columns + ['Dataset']
    
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

def create_walk_forward_splits(df, initial_train_months=24, validation_months=3, test_months=3, step_months=3):
    """
    Create chronological walk-forward validation splits for time series data.
    
    Parameters:
    df (DataFrame): Full dataset with Date column
    initial_train_months (int): Initial training period in months
    validation_months (int): Validation period in months
    test_months (int): Test period in months  
    step_months (int): Step size between validation periods in months
    
    Returns:
    list: List of dictionaries containing train/val/test splits with metadata
    """
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Walk-forward config: {initial_train_months}m train, {validation_months}m val, {test_months}m test, {step_months}m step")
    
    splits = []
    split_num = 1
    
    # Calculate first split dates
    first_val_start = start_date + relativedelta(months=initial_train_months)
    
    current_val_start = first_val_start
    
    while True:
        # Calculate dates for current split
        val_end = current_val_start + relativedelta(months=validation_months)
        test_start = val_end
        test_end = test_start + relativedelta(months=test_months)
        
        # Check if we have enough data for test period
        if test_end > end_date:
            print(f"Stopping at split {split_num-1}: not enough data for test period ending {test_end.strftime('%Y-%m-%d')}")
            break
            
        # For expanding window: train from start to validation start
        train_start = start_date
        train_end = current_val_start
        
        # Create masks for each split
        train_mask = (df['Date'] >= train_start) & (df['Date'] < train_end)
        val_mask = (df['Date'] >= current_val_start) & (df['Date'] < val_end)
        test_mask = (df['Date'] >= test_start) & (df['Date'] < test_end)
        
        # Get data for each split
        train_data = df[train_mask].copy()
        val_data = df[val_mask].copy()
        test_data = df[test_mask].copy()
        
        # Only include split if all sets have reasonable size
        if len(train_data) >= 200 and len(val_data) >= 30 and len(test_data) >= 30:
            split_info = {
                'split_num': split_num,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': current_val_start,
                'val_end': val_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'train_size': len(train_data),
                'val_size': len(val_data),
                'test_size': len(test_data)
            }
            
            splits.append(split_info)
            
            print(f"Split {split_num}: Train {train_start.strftime('%Y-%m')} to {train_end.strftime('%Y-%m')} ({len(train_data)} samples)")
            print(f"           Val {current_val_start.strftime('%Y-%m')} to {val_end.strftime('%Y-%m')} ({len(val_data)} samples)")
            print(f"           Test {test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')} ({len(test_data)} samples)")
            
            split_num += 1
        
        # Move to next validation period
        current_val_start += relativedelta(months=step_months)
        
        # Safety check to prevent infinite loop
        if split_num > 50:  # Reasonable max number of splits
            print("Reached maximum number of splits (50)")
            break
    
    print(f"\nCreated {len(splits)} walk-forward splits")
    return splits

def walk_forward_validation(df, selected_features=None, **xgb_params):
    """
    Perform walk-forward validation on the dataset.
    
    Parameters:
    df (DataFrame): Full dataset
    selected_features (list): Features to use (if None, use all except Date/Target)
    **xgb_params: XGBoost parameters
    
    Returns:
    dict: Results from all splits including metrics and predictions
    """
    print("=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    # Create walk-forward splits
    splits = create_walk_forward_splits(df, 
                                      initial_train_months=24,  # 2 years initial training
                                      validation_months=3,      # 3 months validation
                                      test_months=3,           # 3 months test
                                      step_months=6)           # 6 months step (50% overlap)
    
    if len(splits) == 0:
        raise ValueError("No valid splits created. Check your data size and parameters.")
    
    # Store results for all splits
    all_results = {
        'splits': [],
        'summary_metrics': {
            'train_accuracy': [],
            'val_accuracy': [],
            'test_accuracy': [],
            'train_f1': [],
            'val_f1': [],
            'test_f1': []
        },
        'models': [],
        'feature_importance': []
    }
    
    for split_info in splits:
        split_num = split_info['split_num']
        print(f"\n{'='*20} SPLIT {split_num} {'='*20}")
        
        # Prepare data for current split
        train_data = split_info['train_data']
        val_data = split_info['val_data']
        test_data = split_info['test_data']
        
        # Prepare features
        X_train, y_train = prepare_features(train_data, selected_features=selected_features)
        X_val, y_val = prepare_features(val_data, selected_features=selected_features)
        X_test, y_test = prepare_features(test_data, selected_features=selected_features)
        
        # Compute class weights for current training set
        class_weights = compute_class_weights(y_train)
        
        # Train model on current training set
        print(f"\nTraining XGBoost for split {split_num}...")
        model = train_xgboost(X_train, y_train, class_weights=class_weights, **xgb_params)
        
        # Evaluate on all sets
        train_metrics = evaluate_model(model, X_train, y_train, f"Split {split_num} - Train")
        val_metrics = evaluate_model(model, X_val, y_val, f"Split {split_num} - Val")
        test_metrics = evaluate_model(model, X_test, y_test, f"Split {split_num} - Test")
        
        # Store results
        split_results = {
            'split_num': split_num,
            'dates': {
                'train_start': split_info['train_start'],
                'train_end': split_info['train_end'],
                'val_start': split_info['val_start'],
                'val_end': split_info['val_end'],
                'test_start': split_info['test_start'],
                'test_end': split_info['test_end']
            },
            'metrics': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'model': model,
            'feature_importance': model.feature_importances_
        }
        
        all_results['splits'].append(split_results)
        all_results['models'].append(model)
        all_results['feature_importance'].append(model.feature_importances_)
        
        # Store summary metrics
        all_results['summary_metrics']['train_accuracy'].append(train_metrics['accuracy'])
        all_results['summary_metrics']['val_accuracy'].append(val_metrics['accuracy'])
        all_results['summary_metrics']['test_accuracy'].append(test_metrics['accuracy'])
        all_results['summary_metrics']['train_f1'].append(train_metrics['f1_score'])
        all_results['summary_metrics']['val_f1'].append(val_metrics['f1_score'])
        all_results['summary_metrics']['test_f1'].append(test_metrics['f1_score'])
    
    return all_results, X_train.columns.tolist()  # Return feature names for analysis

def analyze_walk_forward_results(results, feature_names):
    """
    Analyze and visualize walk-forward validation results.
    
    Parameters:
    results (dict): Results from walk_forward_validation
    feature_names (list): List of feature names
    """
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION ANALYSIS")
    print("="*80)
    
    metrics = results['summary_metrics']
    
    # Summary statistics
    print(f"\nSUMMARY ACROSS {len(results['splits'])} SPLITS:")
    print("-" * 50)
    
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        print(f"{metric_name:15s}: {mean_val:.4f} ± {std_val:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Performance over time (accuracy)
    splits = [s['split_num'] for s in results['splits']]
    axes[0, 0].plot(splits, metrics['train_accuracy'], 'o-', label='Train', linewidth=2)
    axes[0, 0].plot(splits, metrics['val_accuracy'], 's-', label='Validation', linewidth=2)
    axes[0, 0].plot(splits, metrics['test_accuracy'], '^-', label='Test', linewidth=2)
    axes[0, 0].set_xlabel('Split Number')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Across Time Splits')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Performance over time (F1-score)
    axes[0, 1].plot(splits, metrics['train_f1'], 'o-', label='Train', linewidth=2)
    axes[0, 1].plot(splits, metrics['val_f1'], 's-', label='Validation', linewidth=2)
    axes[0, 1].plot(splits, metrics['test_f1'], '^-', label='Test', linewidth=2)
    axes[0, 1].set_xlabel('Split Number')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('Model F1-Score Across Time Splits')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature importance stability
    feature_importance_matrix = np.array(results['feature_importance'])
    mean_importance = np.mean(feature_importance_matrix, axis=0)
    std_importance = np.std(feature_importance_matrix, axis=0)
    
    # Sort by mean importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_importance': mean_importance,
        'std_importance': std_importance
    }).sort_values('mean_importance', ascending=False)
    
    top_features = importance_df.head(15)
    
    axes[1, 0].barh(range(len(top_features)), top_features['mean_importance'], 
                   xerr=top_features['std_importance'], alpha=0.7)
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features['feature'])
    axes[1, 0].set_xlabel('Feature Importance')
    axes[1, 0].set_title('Top 15 Features (Mean ± Std across splits)')
    axes[1, 0].invert_yaxis()
    
    # 4. Overfitting analysis (train vs validation gap)
    train_val_gap = np.array(metrics['train_accuracy']) - np.array(metrics['val_accuracy'])
    val_test_gap = np.array(metrics['val_accuracy']) - np.array(metrics['test_accuracy'])
    
    axes[1, 1].plot(splits, train_val_gap, 'o-', label='Train-Val Gap', linewidth=2)
    axes[1, 1].plot(splits, val_test_gap, 's-', label='Val-Test Gap', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Split Number')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].set_title('Overfitting Analysis')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('walk_forward_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed split-by-split analysis
    print(f"\nDETAILED SPLIT ANALYSIS:")
    print("-" * 80)
    for split_result in results['splits']:
        split_num = split_result['split_num']
        dates = split_result['dates']
        metrics = split_result['metrics']
        
        print(f"\nSplit {split_num}:")
        print(f"  Period: {dates['test_start'].strftime('%Y-%m-%d')} to {dates['test_end'].strftime('%Y-%m-%d')}")
        print(f"  Test Accuracy: {metrics['test']['accuracy']:.4f}")
        print(f"  Test F1-Score: {metrics['test']['f1_score']:.4f}")
        print(f"  Val-Test Gap:  {metrics['val']['accuracy'] - metrics['test']['accuracy']:+.4f}")
    
    return importance_df

def save_walk_forward_results(results, feature_names, feature_importance_df, datasets_used=['OMXS30']):
    """
    Save walk-forward validation results to files.
    
    Parameters:
    results (dict): Results from walk_forward_validation
    feature_names (list): List of feature names
    feature_importance_df (DataFrame): Feature importance analysis
    datasets_used (list): List of datasets used in training
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the best model (highest average validation accuracy)
    val_accuracies = results['summary_metrics']['val_accuracy']
    best_split_idx = np.argmax(val_accuracies)
    best_model = results['models'][best_split_idx]
    
    # Include datasets in filename if multiple datasets are used
    if len(datasets_used) == 1:
        model_filename = f'walkforward_xgboost_{datasets_used[0]}_{timestamp}.pkl'
    else:
        dataset_str = "_".join(datasets_used[:3])  # Limit to first 3 to avoid too long filenames
        if len(datasets_used) > 3:
            dataset_str += f"_plus{len(datasets_used)-3}more"
        model_filename = f'walkforward_xgboost_combined_{dataset_str}_{timestamp}.pkl'
    
    joblib.dump(best_model, f"models/{model_filename}")
    
    # Helper function to convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Create comprehensive log entry
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'validation_type': 'walk_forward',
        'model_type': 'XGBoost',
        'model_filename': model_filename,
        'datasets_used': datasets_used,
        'combined_datasets': len(datasets_used) > 1,
        'best_split': int(best_split_idx + 1),
        'total_splits': len(results['splits']),
        'parameters': {
            'n_estimators': int(best_model.n_estimators),
            'max_depth': int(best_model.max_depth),
            'learning_rate': float(best_model.learning_rate),
            'subsample': float(best_model.subsample),
            'colsample_bytree': float(best_model.colsample_bytree),
            'min_child_weight': float(best_model.min_child_weight),
            'reg_alpha': float(best_model.reg_alpha),
            'reg_lambda': float(best_model.reg_lambda),
            'random_state': int(best_model.random_state)
        },
        'features': {
            'feature_count': len(feature_names),
            'top_10_features': feature_importance_df.head(10)['feature'].tolist()
        },
        'summary_metrics': {
            'mean_train_accuracy': float(np.mean(results['summary_metrics']['train_accuracy'])),
            'mean_val_accuracy': float(np.mean(results['summary_metrics']['val_accuracy'])),
            'mean_test_accuracy': float(np.mean(results['summary_metrics']['test_accuracy'])),
            'std_test_accuracy': float(np.std(results['summary_metrics']['test_accuracy'])),
            'mean_train_f1': float(np.mean(results['summary_metrics']['train_f1'])),
            'mean_val_f1': float(np.mean(results['summary_metrics']['val_f1'])),
            'mean_test_f1': float(np.mean(results['summary_metrics']['test_f1'])),
            'std_test_f1': float(np.std(results['summary_metrics']['test_f1']))
        },
        'split_details': []
    }
    
    # Add details for each split
    for split_result in results['splits']:
        split_detail = {
            'split_num': int(split_result['split_num']),
            'test_period': f"{split_result['dates']['test_start'].strftime('%Y-%m-%d')} to {split_result['dates']['test_end'].strftime('%Y-%m-%d')}",
            'test_accuracy': float(split_result['metrics']['test']['accuracy']),
            'test_f1_score': float(split_result['metrics']['test']['f1_score']),
            'val_test_gap': float(split_result['metrics']['val']['accuracy'] - split_result['metrics']['test']['accuracy'])
        }
        log_entry['split_details'].append(split_detail)
    
    # Convert any remaining numpy types
    log_entry = convert_numpy_types(log_entry)
    
    # Save to log file
    log_file = 'walk_forward_log.json'
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
    except (json.JSONDecodeError, FileNotFoundError):
        logs = []
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    print(f"\n✅ Walk-forward results saved:")
    print(f"📊 Best model saved as: {model_filename}")
    print(f"📈 Results logged to: {log_file}")
    print(f"🏆 Best split: {best_split_idx + 1} (Val Accuracy: {val_accuracies[best_split_idx]:.4f})")
    
    return model_filename, log_entry

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
        'n_estimators': 300,
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

def main(datasets=['OMXS30'], use_combined_data=True):
    """
    Main function to perform walk-forward validation with XGBoost.
    
    Parameters:
    datasets (list): List of datasets to use. Options: ['OMXS30', 'SP500'] or individual stock names
    use_combined_data (bool): If True and multiple datasets, combine them. If False, train separately.
    """
    print("=" * 80)
    print("WALK-FORWARD VALIDATION FOR STOCK PRICE PREDICTION")
    print("=" * 80)
    
    try:
        # 1. Load the dataset(s)
        if len(datasets) == 1:
            print(f"Loading {datasets[0]} dataset...")
            df = pd.read_csv(f'data/{datasets[0]}.csv')
            print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        elif use_combined_data:
            print(f"Loading and combining {len(datasets)} datasets: {', '.join(datasets)}...")
            dfs = []
            for dataset in datasets:
                temp_df = pd.read_csv(f'data/{dataset}.csv')
                # Add dataset identifier for tracking
                temp_df['Dataset'] = dataset
                dfs.append(temp_df)
                print(f"  {dataset}: {temp_df.shape[0]} rows, {temp_df.shape[1]} columns")
            
            df = pd.concat(dfs, ignore_index=True)
            print(f"\nCombined dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        else:
            # Train separate models for each dataset
            print(f"Training separate models for {len(datasets)} datasets...")
            all_results = {}
            for dataset in datasets:
                print(f"\n{'='*50}")
                print(f"PROCESSING DATASET: {dataset}")
                print(f"{'='*50}")
                
                df = pd.read_csv(f'data/{dataset}.csv')
                print(f"Loaded {dataset}: {df.shape[0]} rows, {df.shape[1]} columns")
                
                # Recursively call main for each dataset
                dataset_results = main(datasets=[dataset], use_combined_data=True)
                all_results[dataset] = dataset_results
            
            return all_results
        
        # 2. Perform walk-forward validation
        print("\n" + "="*60)
        print("STARTING WALK-FORWARD VALIDATION")
        print("="*60)
        
        # Use optimal XGBoost parameters (you can tune these)
        xgb_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        selected_features = [
            "Close",
            "Volume",
            "BB_middle",
            "RSI_14",
            "SMA_20",
            "BB_width",
            
        ]
        
        results, feature_names = walk_forward_validation(df, selected_features=None, **xgb_params)
        
        # 3. Analyze results
        print("\n" + "="*60)
        print("ANALYZING RESULTS")
        print("="*60)
        
        feature_importance_df = analyze_walk_forward_results(results, feature_names)
        
        # 4. Save results
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        model_filename, log_entry = save_walk_forward_results(results, feature_names, feature_importance_df, datasets)
        
        # 5. Compare with traditional validation (if available)
        print("\n" + "="*60)
        print("WALK-FORWARD VALIDATION COMPLETE")
        print("="*60)
        
        summary_metrics = results['summary_metrics']
        
        print(f"\n🎯 FINAL PERFORMANCE SUMMARY:")
        print(f"   Mean Test Accuracy: {np.mean(summary_metrics['test_accuracy']):.4f} ± {np.std(summary_metrics['test_accuracy']):.4f}")
        print(f"   Mean Test F1-Score: {np.mean(summary_metrics['test_f1']):.4f} ± {np.std(summary_metrics['test_f1']):.4f}")
        print(f"   Total Splits Tested: {len(results['splits'])}")
        print(f"   Best Single Split: {np.max(summary_metrics['test_accuracy']):.4f}")
        print(f"   Worst Single Split: {np.min(summary_metrics['test_accuracy']):.4f}")
        
        print(f"\n📈 MODEL STABILITY:")
        test_acc_std = np.std(summary_metrics['test_accuracy'])
        if test_acc_std < 0.02:
            stability = "EXCELLENT"
        elif test_acc_std < 0.04:
            stability = "GOOD"
        elif test_acc_std < 0.06:
            stability = "MODERATE"
        else:
            stability = "POOR"
        print(f"   Accuracy Std Dev: {test_acc_std:.4f} ({stability} stability)")
        
        print(f"\n🔧 OVERFITTING ANALYSIS:")
        train_val_gaps = np.array(summary_metrics['train_accuracy']) - np.array(summary_metrics['val_accuracy'])
        val_test_gaps = np.array(summary_metrics['val_accuracy']) - np.array(summary_metrics['test_accuracy'])
        
        print(f"   Mean Train-Val Gap: {np.mean(train_val_gaps):.4f}")
        print(f"   Mean Val-Test Gap: {np.mean(val_test_gaps):.4f}")
        
        if np.mean(train_val_gaps) > 0.05:
            print("   ⚠️  High train-validation gap suggests overfitting")
        else:
            print("   ✅ Good generalization (low train-validation gap)")
            
        print(f"\n📊 TOP 5 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(feature_importance_df.head(5).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")
        
        print(f"\n📈 Datasets used: {', '.join(datasets)}")
        
        return {
            'results': results,
            'feature_importance': feature_importance_df,
            'model_filename': model_filename,
            'summary': log_entry['summary_metrics'],
            'datasets_used': datasets
        }
        
    except Exception as e:
        print(f"\n❌ Error during walk-forward validation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Show available datasets
    available_datasets = get_available_datasets()
    print("=" * 80)
    print("AVAILABLE DATASETS")
    print("=" * 80)
    print("Available datasets in 'data/' directory:")
    for i, dataset in enumerate(available_datasets, 1):
        print(f"  {i:2d}. {dataset}")
    print()
    
    # Configuration: Choose which datasets to use
    
    # Example 1: Single dataset
    # results = main(datasets=['OMXS30'])
    
    # Example 2: Multiple major indices combined
    # results = main(datasets=['OMXS30', 'SP500'], use_combined_data=True)
    
    # Example 3: Multiple Swedish stocks combined
    results = main(datasets=['atlascopco', 'electrolux', 'ericsson', 'getinge', 'handelsbanken', 'hmb', 'investor', 'nordea', 'sandvik', 'seb', 'skf', 'swedbank', 'telia', 'OMXS30'], use_combined_data=True)
    
    # Example 4: All major indices and some stocks combined
    # results = main(datasets=['OMXS30', 'SP500', 'ericsson', 'volvo'], use_combined_data=True)
    
    # Example 5: Train separate models for each dataset
    # results = main(datasets=['OMXS30', 'SP500', 'ericsson'], use_combined_data=False)
    
    # Example 6: All Swedish bank stocks combined
    # results = main(datasets=['handelsbanken', 'nordea', 'seb', 'swedbank'], use_combined_data=True)
    
    # Example 7: Technology stocks combined
    # results = main(datasets=['ericsson', 'telia'], use_combined_data=True)
    
    # Default: Train on OMXS30 and SP500 combined
    # results = main(datasets=['OMXS30', 'SP500'], use_combined_data=True)
