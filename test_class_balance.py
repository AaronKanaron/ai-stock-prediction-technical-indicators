#!/usr/bin/env python3
"""
Test improved class balancing for XGBoost models.
"""

from xgboost_walk_forward import main

def test_different_weighting_methods():
    """Test different class weighting methods."""

    print("=" * 80)
    print("TESTING DIFFERENT CLASS WEIGHTING METHODS")
    print("=" * 80)

    feature_selection_config = {
        'method': 'f_classif',
        'percentile': 50
    }

    methods = [
        'balanced_moderate',
        'balanced',
        'sqrt_balanced'
    ]

    results = {}

    for method in methods:
        print(f"\n{'='*20} Testing {method} {'='*20}")

        try:
            result = main(
                dataset_names=['OMXS30'],
                feature_selection=feature_selection_config,
                class_weight_method=method
            )

            test_acc = result['results']['summary_metrics']['test_accuracy']
            test_f1 = result['results']['summary_metrics']['test_f1']

            results[method] = {
                'test_accuracy_mean': sum(test_acc) / len(test_acc),
                'test_f1_mean': sum(test_f1) / len(test_f1)
            }

        except Exception as e:
            print(f"Error with {method}: {e}")
            results[method] = None

    # Print comparison
    print("\n" + "=" * 80)
    print("CLASS WEIGHTING METHOD COMPARISON")
    print("=" * 80)

    for method, result in results.items():
        if result:
            print(f"\n{method.upper().replace('_', ' ')}:")
            print(f"  Mean Test Accuracy: {result['test_accuracy_mean']:.4f}")
            print(f"  Mean Test F1-Score: {result['test_f1_mean']:.4f}")
        else:
            print(f"\n{method.upper().replace('_', ' ')}: FAILED")

if __name__ == "__main__":
    test_different_weighting_methods()