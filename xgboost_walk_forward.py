import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from dateutil.relativedelta import relativedelta
from progress.bar import Bar
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """Walk-forward validation for time series classification with XGBoost."""
    
    def __init__(self, initial_months=24, validation_months=3, test_months=3, step_months=6):
        self.initial_months = initial_months
        self.validation_months = validation_months 
        self.test_months = test_months
        self.step_months = step_months
        self.results = None

    def load_datasets(self, dataset_names):
        """Load and combine multiple datasets."""
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
            
        dataframes = []
        for name in dataset_names:
            df = pd.read_csv(f'data/{name}.csv')
            df['dataset_source'] = name
            dataframes.append(df)
            
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def prepare_features(self, df, feature_columns=None, exclude_columns=None):
        """Prepare feature matrix and target vector."""
        if exclude_columns is None:
            exclude_columns = ['Date', 'Target', 'dataset_source']
        
        if feature_columns is not None:
            X = df[feature_columns]
        else:
            X = df.drop(columns=exclude_columns)
        
        y = df['Target']
        
        # Remove rows with missing values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        return X, y

    def compute_class_weights(self, y):
        """Compute balanced class weights."""
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        n_classes = len(class_counts)
        
        weights = {}
        for class_label in class_counts.index:
            weights[class_label] = total_samples / (n_classes * class_counts[class_label])
        
        return weights

    def create_time_splits(self, df):
        """Create chronological walk-forward splits."""
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        start_date = df['Date'].min()
        end_date = df['Date'].max()
        
        splits = []
        first_val_start = start_date + relativedelta(months=self.initial_months)
        current_val_start = first_val_start
        split_num = 1
        
        while True:
            val_end = current_val_start + relativedelta(months=self.validation_months)
            test_start = val_end
            test_end = test_start + relativedelta(months=self.test_months)
            
            if test_end > end_date:
                break
                
            train_start = start_date
            train_end = current_val_start
            
            # Create data splits
            train_mask = (df['Date'] >= train_start) & (df['Date'] < train_end)
            val_mask = (df['Date'] >= current_val_start) & (df['Date'] < val_end)
            test_mask = (df['Date'] >= test_start) & (df['Date'] < test_end)
            
            train_data = df[train_mask].copy()
            val_data = df[val_mask].copy()
            test_data = df[test_mask].copy()
            
            # Only include splits with sufficient data
            if len(train_data) >= 200 and len(val_data) >= 30 and len(test_data) >= 30:
                splits.append({
                    'split_id': split_num,
                    'periods': {
                        'train': (train_start, train_end),
                        'val': (current_val_start, val_end),
                        'test': (test_start, test_end)
                    },
                    'data': {
                        'train': train_data,
                        'val': val_data,
                        'test': test_data
                    }
                })
                split_num += 1
            
            current_val_start += relativedelta(months=self.step_months)
            
            if split_num > 50:  # Safety limit
                break
        
        return splits

    def train_model(self, X_train, y_train, params=None):
        """Train XGBoost classifier with class balancing."""
        model_params = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 100,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'num_class': 3
        }
        
        if params:
            model_params.update(params)

        model = XGBClassifier(**model_params)

        # Handle class imbalance with sample weights
        class_weights = self.compute_class_weights(y_train)
        sample_weights = [class_weights.get(label, 1.0) for label in y_train]
        
        model.fit(X_train, y_train, sample_weight=sample_weights)
        return model

    def evaluate_model(self, model, X, y):
        """Evaluate model performance."""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1_score': f1_score(y, predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(y, predictions),
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        return metrics

    def run_validation(self, df, feature_columns=None, model_params=None):
        """Execute walk-forward validation."""
        splits = self.create_time_splits(df)
        
        if not splits:
            raise ValueError("No valid time splits created")
        
        all_results = {
            'splits': [],
            'models': [],
            'feature_importance': [],
            'summary_metrics': {
                'train_accuracy': [], 'val_accuracy': [], 'test_accuracy': [],
                'train_f1': [], 'val_f1': [], 'test_f1': []
            }
        }
        
        # Progress bar for splits
        progress = Bar('Processing splits', max=len(splits))
        
        for split_info in splits:
            split_id = split_info['split_id']
            data = split_info['data']
            
            # Prepare features for current split
            X_train, y_train = self.prepare_features(data['train'], feature_columns)
            X_val, y_val = self.prepare_features(data['val'], feature_columns)
            X_test, y_test = self.prepare_features(data['test'], feature_columns)
            
            # Train and evaluate model
            model = self.train_model(X_train, y_train, model_params)
            
            train_metrics = self.evaluate_model(model, X_train, y_train)
            val_metrics = self.evaluate_model(model, X_val, y_val)
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Store results
            split_result = {
                'split_id': split_id,
                'periods': split_info['periods'],
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                },
                'model': model,
                'feature_importance': model.feature_importances_
            }
            
            all_results['splits'].append(split_result)
            all_results['models'].append(model)
            all_results['feature_importance'].append(model.feature_importances_)
            
            # Update summary metrics
            all_results['summary_metrics']['train_accuracy'].append(train_metrics['accuracy'])
            all_results['summary_metrics']['val_accuracy'].append(val_metrics['accuracy'])
            all_results['summary_metrics']['test_accuracy'].append(test_metrics['accuracy'])
            all_results['summary_metrics']['train_f1'].append(train_metrics['f1_score'])
            all_results['summary_metrics']['val_f1'].append(val_metrics['f1_score'])
            all_results['summary_metrics']['test_f1'].append(test_metrics['f1_score'])
            
            progress.next()
        
        progress.finish()
        
        self.results = all_results
        return all_results, X_train.columns.tolist()

    def analyze_results(self, feature_names):
        """Analyze and visualize validation results."""
        if self.results is None:
            raise ValueError("No results available. Run validation first.")
        
        metrics = self.results['summary_metrics']
        
        print(f"\nWalk-Forward Validation Results ({len(self.results['splits'])} splits)")
        print("-" * 60)
        
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:15s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Create visualization
        self._create_plots(feature_names)
        
        # Feature importance analysis
        feature_importance_matrix = np.array(self.results['feature_importance'])
        mean_importance = np.mean(feature_importance_matrix, axis=0)
        std_importance = np.std(feature_importance_matrix, axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_importance,
            'std_importance': std_importance
        }).sort_values('mean_importance', ascending=False)
        
        print(f"\nTop 10 Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']}: {row['mean_importance']:.4f}")
        
        return importance_df

    def _create_plots(self, feature_names):
        """Create analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = self.results['summary_metrics']
        splits = [s['split_id'] for s in self.results['splits']]
        
        # Accuracy over time
        axes[0, 0].plot(splits, metrics['train_accuracy'], 'o-', label='Train', linewidth=2)
        axes[0, 0].plot(splits, metrics['val_accuracy'], 's-', label='Validation', linewidth=2)
        axes[0, 0].plot(splits, metrics['test_accuracy'], '^-', label='Test', linewidth=2)
        axes[0, 0].set_xlabel('Split Number')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Accuracy Across Time Splits')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1-score over time
        axes[0, 1].plot(splits, metrics['train_f1'], 'o-', label='Train', linewidth=2)
        axes[0, 1].plot(splits, metrics['val_f1'], 's-', label='Validation', linewidth=2)
        axes[0, 1].plot(splits, metrics['test_f1'], '^-', label='Test', linewidth=2)
        axes[0, 1].set_xlabel('Split Number')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_title('F1-Score Across Time Splits')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature importance
        feature_importance_matrix = np.array(self.results['feature_importance'])
        mean_importance = np.mean(feature_importance_matrix, axis=0)
        std_importance = np.std(feature_importance_matrix, axis=0)
        
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
        axes[1, 0].set_title('Top 15 Features')
        axes[1, 0].invert_yaxis()
        
        # Overfitting analysis
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
        plt.savefig('visualizer/images/walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_best_model(self, dataset_names):
        """Save the best performing model."""
        if self.results is None:
            raise ValueError("No results available. Run validation first.")
        
        # Find best model based on validation accuracy
        val_accuracies = self.results['summary_metrics']['val_accuracy']
        best_split_idx = np.argmax(val_accuracies)
        best_model = self.results['models'][best_split_idx]
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(dataset_names) == 1:
            model_filename = f'walkforward_xgboost_{dataset_names[0]}_{timestamp}.pkl'
        else:
            dataset_str = "_".join(dataset_names[:3])
            if len(dataset_names) > 3:
                dataset_str += f"_plus{len(dataset_names)-3}more"
            model_filename = f'walkforward_xgboost_combined_{dataset_str}_{timestamp}.pkl'
        
        # Save model
        joblib.dump(best_model, f"models/{model_filename}")
        
        print(f"\nModel saved:")
        print(f"  Filename: {model_filename}")
        print(f"  Best split: {best_split_idx + 1}")
        print(f"  Best validation accuracy: {val_accuracies[best_split_idx]:.4f}")
        
        return model_filename, best_model


def main(dataset_names=None, feature_columns=None, model_params=None):
    """Run walk-forward validation on specified datasets."""
    if dataset_names is None:
        dataset_names = ['OMXS30']
    
    print("Walk-Forward Validation for Stock Prediction")
    print("=" * 50)
    
    validator = WalkForwardValidator()
    
    print(f"Using datasets: {', '.join(dataset_names)}")
    
    # Load and combine datasets
    df = validator.load_datasets(dataset_names)
    print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\nStarting walk-forward validation...")
    results, feature_names = validator.run_validation(df, feature_columns, model_params)
    
    print("\nAnalyzing results...")
    importance_df = validator.analyze_results(feature_names)
    
    print("\nSaving best model...")
    model_filename, best_model = validator.save_best_model(dataset_names)
    
    # Final summary
    metrics = results['summary_metrics']
    print(f"\nFinal Summary:")
    print(f"  Test Accuracy: {np.mean(metrics['test_accuracy']):.4f} ± {np.std(metrics['test_accuracy']):.4f}")
    print(f"  Test F1-Score: {np.mean(metrics['test_f1']):.4f} ± {np.std(metrics['test_f1']):.4f}")
    print(f"  Total Splits: {len(results['splits'])}")
    
    return {
        'results': results,
        'feature_importance': importance_df,
        'model_filename': model_filename,
        'best_model': best_model,
        'validator': validator
    }


if __name__ == "__main__":
    results = main(["OMXS30", "SP500"])