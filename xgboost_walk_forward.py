import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
from typing import Literal, List, Union, Optional
from dateutil.relativedelta import relativedelta
from progress.bar import Bar
import warnings
warnings.filterwarnings('ignore')

Stocks = Literal[
    "abb",
    "addtech",
    "alfa",
    "assa",
    "astrazeneca",
    'atlascopco',
    "boliden",
    "epiroc",
    "eqt",
    "ericsson",
    "essity",
    "evolution",
    "handelsbanken",
    "hexagon",
    "hmb",
    "industrivarden",
    "investor",
    "lifco",
    "nibe",
    "nordea",
    "saab",
    "sandvik",
    "sca",
    "seb",
    "skanska",
    "skf",
    "swedbank",
    "tele2",
    "telia",
    "volvo",
    
    "OMXS30",
]

class WalkForwardValidator:
    """Walk-forward validation for time series classification with XGBoost."""

    def __init__(self, initial_months=24, validation_months=3, test_months=3, step_months=6):
        """Initialize validator with time split parameters.

        Args:
            initial_months: Initial training period in months
            validation_months: Validation period in months
            test_months: Test period in months
            step_months: Step size for walk-forward in months
        """
        self.initial_months = initial_months
        self.validation_months = validation_months
        self.test_months = test_months
        self.step_months = step_months
        self.results = None
        self.feature_selector = None

    def load_datasets(self, dataset_names):
        """Load and combine multiple datasets."""
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]

        sorted_dataset_names = sorted(dataset_names)
        dataframes = []

        for i, name in enumerate(sorted_dataset_names):
            df = pd.read_csv(f'data/{name}.csv')
            df['dataset_source'] = name
            df['Date'] = pd.to_datetime(df['Date']) + pd.Timedelta(microseconds=i)
            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True)

    def prepare_features(self, df, feature_columns=None, exclude_columns=None):
        """Prepare feature matrix and target vector.

        Args:
            df: DataFrame with features and target
            feature_columns: Specific columns to use as features
            exclude_columns: Columns to exclude from features

        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        if exclude_columns is None:
            exclude_columns = ['Date', 'Target', 'dataset_source']

        X = df[feature_columns] if feature_columns else df.drop(columns=exclude_columns)
        y = df['Target']

        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        return X[valid_mask], y[valid_mask]

    def compute_class_weights(self, y, method='balanced_moderate'):
        """Compute balanced class weights for handling class imbalance.

        Args:
            y: Target vector
            method: Weighting method ('balanced', 'balanced_moderate', 'sqrt_balanced')

        Returns:
            dict: Class weights mapping
        """
        class_counts = y.value_counts().sort_index()
        total_samples = len(y)
        n_classes = len(class_counts)

        if method == 'balanced':
            weights = {
                class_label: total_samples / (n_classes * class_counts[class_label])
                for class_label in class_counts.index
            }
        elif method == 'balanced_moderate':
            weights = {}
            for class_label in class_counts.index:
                raw_weight = total_samples / (n_classes * class_counts[class_label])
                moderate_weight = np.sqrt(raw_weight)
                weights[class_label] = moderate_weight
        elif method == 'sqrt_balanced':
            weights = {}
            for class_label in class_counts.index:
                freq = class_counts[class_label] / total_samples
                weights[class_label] = 1.0 / np.sqrt(freq)
        else:
            weights = {class_label: 1.0 for class_label in class_counts.index}

        print(f"Class distribution: {dict(class_counts)}")
        print(f"Class weights ({method}): {dict(weights)}")

        return weights

    def select_features(self, X_train, y_train, X_val=None, X_test=None,
                       method='f_classif', k=20, percentile=50):
        """Apply univariate feature selection to identify the best features.

        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Validation feature matrix (optional)
            X_test: Test feature matrix (optional)
            method: Selection method ('f_classif', 'mutual_info', 'chi2')
            k: Number of best features to select (used with SelectKBest)
            percentile: Percentile of features to keep (used with SelectPercentile)

        Returns:
            tuple: (X_train_selected, X_val_selected, X_test_selected, selected_features)
        """
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        elif method == 'chi2':
            if (X_train < 0).any().any():
                print("Warning: Chi2 requires non-negative features. Converting negative values.")
                X_train = X_train.copy()
                X_train = X_train - X_train.min() + 1e-8
                if X_val is not None:
                    X_val = X_val - X_val.min() + 1e-8
                if X_test is not None:
                    X_test = X_test - X_test.min() + 1e-8
            score_func = chi2
        else:
            raise ValueError(f"Unknown method: {method}")

        if k is not None and k > 0:
            selector = SelectKBest(score_func=score_func, k=min(k, X_train.shape[1]))
        else:
            selector = SelectPercentile(score_func=score_func, percentile=percentile)

        X_train_selected = selector.fit_transform(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()

        X_val_selected = None
        X_test_selected = None
        if X_val is not None:
            X_val_selected = selector.transform(X_val)
        if X_test is not None:
            X_test_selected = selector.transform(X_test)

        feature_scores = selector.scores_
        self.feature_selection_results = {
            'selected_features': selected_features,
            'feature_scores': dict(zip(X_train.columns, feature_scores)),
            'method': method,
            'n_selected': len(selected_features)
        }

        self.feature_selector = selector

        return X_train_selected, X_val_selected, X_test_selected, selected_features

    def create_time_splits(self, df):
        """Create chronological walk-forward splits.

        Args:
            df: DataFrame with Date column

        Returns:
            list: Time split configurations
        """
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        start_date = df['Date'].min()
        end_date = df['Date'].max()
        splits = []

        current_val_start = start_date + relativedelta(months=self.initial_months)
        split_num = 1

        while split_num <= 50:  # Safety limit
            val_end = current_val_start + relativedelta(months=self.validation_months)
            test_start = val_end
            test_end = test_start + relativedelta(months=self.test_months)

            if test_end > end_date:
                break

            train_mask = (df['Date'] >= start_date) & (df['Date'] < current_val_start)
            val_mask = (df['Date'] >= current_val_start) & (df['Date'] < val_end)
            test_mask = (df['Date'] >= test_start) & (df['Date'] < test_end)

            train_data = df[train_mask].copy()
            val_data = df[val_mask].copy()
            test_data = df[test_mask].copy()

            if len(train_data) >= 200 and len(val_data) >= 30 and len(test_data) >= 30:
                splits.append({
                    'split_id': split_num,
                    'periods': {
                        'train': (start_date, current_val_start),
                        'val': (current_val_start, val_end),
                        'test': (test_start, test_end)
                    },
                    'data': {'train': train_data, 'val': val_data, 'test': test_data}
                })
                split_num += 1

            current_val_start += relativedelta(months=self.step_months)

        return splits

    def train_model(self, X_train, y_train, params=None, class_weight_method='balanced_moderate'):
        """Train XGBoost classifier with improved class balancing."""
        model_params = {
            'n_estimators': 150,
            'max_depth': 4,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'num_class': 3,
            
            'scale_pos_weight': None,
            'max_delta_step': 1,
        }

        if params:
            model_params.update(params)

        class_weights = self.compute_class_weights(y_train, method=class_weight_method)
        sample_weights = [class_weights.get(label, 1.0) for label in y_train]

        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / np.mean(sample_weights)

        model = XGBClassifier(**model_params)
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

    def run_validation(self, df, feature_columns=None, model_params=None,
                      feature_selection=None, class_weight_method='balanced_moderate'):
        """Execute walk-forward validation with optional feature selection.

        Args:
            df: Input dataframe
            feature_columns: Specific columns to use as features
            model_params: XGBoost model parameters
            feature_selection: Dict with feature selection parameters:
                - 'method': 'f_classif', 'mutual_info', or 'chi2'
                - 'k': number of features to select (or None)
                - 'percentile': percentile of features to keep (if k is None)
        """
        splits = self.create_time_splits(df)

        if not splits:
            raise ValueError("No valid time splits created")

        all_results = {
            'splits': [],
            'models': [],
            'feature_importance': [],
            'feature_selection_results': [],
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

            X_train, y_train = self.prepare_features(data['train'], feature_columns)
            X_val, y_val = self.prepare_features(data['val'], feature_columns)
            X_test, y_test = self.prepare_features(data['test'], feature_columns)

            selected_features = None
            if feature_selection:
                method = feature_selection.get('method', 'f_classif')
                k = feature_selection.get('k', 20)
                percentile = feature_selection.get('percentile', 50)

                X_train, X_val, X_test, selected_features = self.select_features(
                    X_train, y_train, X_val, X_test, method, k, percentile
                )

                if isinstance(X_train, np.ndarray):
                    X_train = pd.DataFrame(X_train, columns=selected_features)
                    X_val = pd.DataFrame(X_val, columns=selected_features)
                    X_test = pd.DataFrame(X_test, columns=selected_features)

            model = self.train_model(X_train, y_train, model_params, class_weight_method)
            
            train_metrics = self.evaluate_model(model, X_train, y_train)
            val_metrics = self.evaluate_model(model, X_val, y_val)
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            split_result = {
                'split_id': split_id,
                'periods': split_info['periods'],
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                },
                'model': model,
                'feature_importance': model.feature_importances_,
                'selected_features': selected_features
            }

            all_results['splits'].append(split_result)
            all_results['models'].append(model)
            all_results['feature_importance'].append(model.feature_importances_)
            all_results['feature_selection_results'].append(
                self.feature_selection_results if feature_selection else None
            )
            
            all_results['summary_metrics']['train_accuracy'].append(train_metrics['accuracy'])
            all_results['summary_metrics']['val_accuracy'].append(val_metrics['accuracy'])
            all_results['summary_metrics']['test_accuracy'].append(test_metrics['accuracy'])
            all_results['summary_metrics']['train_f1'].append(train_metrics['f1_score'])
            all_results['summary_metrics']['val_f1'].append(val_metrics['f1_score'])
            all_results['summary_metrics']['test_f1'].append(test_metrics['f1_score'])
            
            progress.next()
        
        progress.finish()
        
        self.results = all_results
        final_features = selected_features if selected_features else X_train.columns.tolist()
        return all_results, final_features

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
        
        self._create_plots(feature_names)
        
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

        if hasattr(self, 'feature_selection_results') and self.feature_selection_results:
            fs_results = self.feature_selection_results
            print(f"\nFeature Selection Summary:")
            print(f"  Method: {fs_results['method']}")
            print(f"  Selected: {fs_results['n_selected']} features")
            print(f"  Top 5 Feature Selection Scores:")
            sorted_scores = sorted(fs_results['feature_scores'].items(),
                                 key=lambda x: x[1], reverse=True)
            for i, (feature, score) in enumerate(sorted_scores[:5]):
                print(f"    {i+1:2d}. {feature}: {score:.4f}")

        return importance_df

    def _create_plots(self, feature_names):
        """Create analysis plots."""
        _, axes = plt.subplots(2, 2, figsize=(15, 12))
        
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
        """Save the best performing model based on validation accuracy.

        Args:
            dataset_names: List of dataset names for filename

        Returns:
            tuple: (model_filename, best_model)
        """
        if self.results is None:
            raise ValueError("No results available. Run validation first.")

        val_accuracies = self.results['summary_metrics']['val_accuracy']
        best_split_idx = np.argmax(val_accuracies)
        best_model = self.results['models'][best_split_idx]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'xgboost_{"_".join(dataset_names)}_{timestamp}.pkl'
        joblib.dump(best_model, f"models/{model_filename}")

        print(f"\nModel saved: {model_filename}")
        print(f"  Best split: {best_split_idx + 1}")
        print(f"  Best validation accuracy: {val_accuracies[best_split_idx]:.4f}")

        return model_filename, best_model


def main(
    dataset_names: Optional[Union[Stocks, List[Stocks]]] = None,
    feature_columns=None, model_params=None, feature_selection=None,
    class_weight_method='balanced_moderate'
):
    """Run walk-forward validation on specified datasets with optional feature selection.

    Args:
        dataset_names: List of dataset names to use
        feature_columns: Specific columns to use as features
        model_params: XGBoost model parameters
        feature_selection: Dict with feature selection parameters:
            - 'method': 'f_classif', 'mutual_info', or 'chi2'
            - 'k': number of features to select (or None)
            - 'percentile': percentile of features to keep (if k is None)
    """
    if dataset_names is None:
        dataset_names = ['OMXS30']

    print("Walk-Forward Validation for Stock Prediction with Feature Selection")
    print("=" * 65)

    if feature_selection:
        method = feature_selection.get('method', 'f_classif')
        k = feature_selection.get('k')
        percentile = feature_selection.get('percentile', 50)
        print(f"Feature Selection: {method}")
        if k:
            print(f"Selecting top {k} features")
        else:
            print(f"Selecting top {percentile}% features")
        print("=" * 65)

    validator = WalkForwardValidator()
    print(f"Using datasets: {', '.join(dataset_names)}")

    df = validator.load_datasets(dataset_names)
    print(f"Combined dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

    results, feature_names = validator.run_validation(df, feature_columns, model_params, feature_selection, class_weight_method)
    importance_df = validator.analyze_results(feature_names)
    model_filename, best_model = validator.save_best_model(dataset_names)
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
    feature_selection_config = {
        'method': 'f_classif',  # "f_classif", 'mutual_info', 'chi2'
        'k': 15,               # select top 15 features
        # 'percentile': 50     #  select top 50% features
    }

    results = main(
        dataset_names=[
            'abb', 'addtech', 'alfa', 'assa', 'astrazeneca', 'atlascopco',
            'boliden', 'epiroc', 'eqt', 'ericsson', 'essity', 'evolution',
            'handelsbanken', 'hexagon', 'hmb', 'industrivarden', 'investor',
            'lifco', 'nibe', 'nordea', 'saab', 'sandvik', 'sca', 'seb',
            'skanska', 'skf', 'swedbank', 'tele2', 'telia', 'volvo'
        ],
        feature_selection=feature_selection_config,
        class_weight_method='balanced_moderate'
    )