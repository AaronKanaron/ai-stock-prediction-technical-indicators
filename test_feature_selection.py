#!/usr/bin/env python3
"""
Test script for univariate feature selection functionality.
Demonstrates different feature selection methods and parameters.
"""

from xgboost_walk_forward import main

def test_f_classif():
    """Test F-test (ANOVA) feature selection."""
    print("=" * 60)
    print("Testing F-test (ANOVA) Feature Selection")
    print("=" * 60)

    feature_selection_config = {
        'method': 'f_classif',
        'k': 15  # Select top 15 features
    }

    results = main(
        dataset_names=['OMXS30'],
        feature_selection=feature_selection_config
    )
    return results

def test_mutual_info():
    """Test Mutual Information feature selection."""
    print("\n" + "=" * 60)
    print("Testing Mutual Information Feature Selection")
    print("=" * 60)

    feature_selection_config = {
        'method': 'mutual_info',
        'k': 15  # Select top 15 features
    }

    results = main(
        dataset_names=['OMXS30'],
        feature_selection=feature_selection_config
    )
    return results

def test_percentile_selection():
    """Test percentile-based feature selection."""
    print("\n" + "=" * 60)
    print("Testing Percentile-based Feature Selection")
    print("=" * 60)

    feature_selection_config = {
        'method': 'f_classif',
        'percentile': 40  # Select top 40% of features
    }

    results = main(
        dataset_names=['OMXS30'],
        feature_selection=feature_selection_config
    )
    return results

def compare_methods():
    """Compare different feature selection methods."""
    print("\n" + "=" * 80)
    print("COMPARISON OF FEATURE SELECTION METHODS")
    print("=" * 80)

    methods = ['f_classif', 'mutual_info']
    results = {}

    for method in methods:
        print(f"\n--- Testing {method} ---")
        config = {
            'method': method,
            'k': 12
        }

        result = main(
            dataset_names=['OMXS30'],
            feature_selection=config
        )

        test_acc = result['results']['summary_metrics']['test_accuracy']
        test_f1 = result['results']['summary_metrics']['test_f1']

        results[method] = {
            'test_accuracy_mean': sum(test_acc) / len(test_acc),
            'test_f1_mean': sum(test_f1) / len(test_f1),
            'selected_features': result['feature_importance'].index[:12].tolist()
        }

    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    for method, result in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Mean Test Accuracy: {result['test_accuracy_mean']:.4f}")
        print(f"  Mean Test F1-Score: {result['test_f1_mean']:.4f}")
        print(f"  Top 5 Features: {result['selected_features'][:5]}")

if __name__ == "__main__":
    print("Feature Selection Testing Suite")
    print("This script tests different univariate feature selection methods.")
    print("Each test uses OMXS30 dataset for faster execution.\n")

    # Run individual tests
    try:
        # Test 1: F-test
        f_results = test_f_classif()

        # Test 2: Mutual Information
        mi_results = test_mutual_info()

        # Test 3: Percentile selection
        perc_results = test_percentile_selection()

        # Test 4: Method comparison
        compare_methods()

        print("\n" + "=" * 80)
        print("✅ ALL FEATURE SELECTION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()