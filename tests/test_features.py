"""
Unit tests for feature engineering module.
Tests card and merchant rolling feature calculations.
"""

import pandas as pd
import pytest
from src.features.transformations import (
    add_card_features,
    add_merchant_features,
    build_historical_features,
)


def create_test_data() -> pd.DataFrame:
    """
    Create sample test data for feature engineering tests.
    
    Returns
    -------
    pd.DataFrame
        Test DataFrame with 4 transactions:
        - card_1: 3 transactions at 09:30, 10:00, 12:00
        - card_2: 1 transaction at 10:30
        - merchant_1: 3 transactions at 09:30, 10:00, 10:30
        - merchant_2: 1 transaction at 12:00
    """
    data = {
        "cc_num": ["card_1", "card_1", "card_1", "card_2"],
        "merchant": ["merchant_1", "merchant_1", "merchant_2", "merchant_1"],
        "trans_date_trans_time": [
            "2026-01-01 09:30:00",
            "2026-01-01 10:00:00",
            "2026-01-01 12:00:00",
            "2026-01-01 10:30:00",
        ],
        "amt": [100.0, 150.0, 50.0, 200.0],
    }
    return pd.DataFrame(data)


def test_card_features():
    """Test card-level historical features."""
    df = create_test_data()
    result = add_card_features(df)
    
    # Verify all card features are created
    expected_features = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_avg_amt_24h",
        "card_max_amt_24h",
    ]
    
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Verify original columns are preserved
    assert "cc_num" in result.columns, "cc_num column missing"
    assert "merchant" in result.columns, "merchant column missing"
    assert "amt" in result.columns, "amt column missing"
    assert "trans_date_trans_time" in result.columns, "trans_date_trans_time column missing"
    
    # Test card_1 transactions
    card_1_txns = result[result["cc_num"] == "card_1"].sort_values("trans_date_trans_time")
    
    # First transaction (09:30) - no previous transactions
    first_txn = card_1_txns.iloc[0]
    assert first_txn["card_txn_count_1h"] == 0, "First transaction should have 0 count"
    assert first_txn["card_txn_count_24h"] == 0, "First transaction should have 0 count"
    assert first_txn["card_avg_amt_24h"] == 0, "First transaction should have 0 average"
    assert first_txn["card_max_amt_24h"] == 0, "First transaction should have 0 max"
    
    # Second transaction (10:00) - only 09:30 in previous hour
    second_txn = card_1_txns.iloc[1]
    assert second_txn["card_txn_count_1h"] == 1, "Should count previous 09:30 transaction"
    assert second_txn["card_txn_count_24h"] == 1, "Should count previous 09:30 transaction"
    assert second_txn["card_avg_amt_24h"] == 100.0, "Average should be 100.0"
    assert second_txn["card_max_amt_24h"] == 100.0, "Max should be 100.0"
    
    # Third transaction (12:00) - 09:30 and 10:00 in previous 24h
    third_txn = card_1_txns.iloc[2]
    assert third_txn["card_txn_count_24h"] == 2, "Should count both previous transactions"
    assert third_txn["card_avg_amt_24h"] == 125.0, "Average should be (100+150)/2"
    assert third_txn["card_max_amt_24h"] == 150.0, "Max should be 150.0"
    
    # Test card_2 transaction (10:30) - no previous transactions
    card_2_txn = result[result["cc_num"] == "card_2"].iloc[0]
    assert card_2_txn["card_txn_count_1h"] == 0, "Card_2 has no previous transactions"
    assert card_2_txn["card_txn_count_24h"] == 0, "Card_2 has no previous transactions"
    assert card_2_txn["card_avg_amt_24h"] == 0, "Card_2 has no previous transactions"
    assert card_2_txn["card_max_amt_24h"] == 0, "Card_2 has no previous transactions"
    
    print("✅ test_card_features passed!")


def test_merchant_features():
    """Test merchant-level historical features."""
    df = create_test_data()
    result = add_merchant_features(df)
    
    # Verify merchant features are created
    expected_features = [
        "merchant_txn_count_1h",
        "merchant_txn_count_24h",
        "merchant_avg_amt_24h",
    ]
    
    for feature in expected_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Verify original columns are preserved
    assert "cc_num" in result.columns, "cc_num column missing"
    assert "merchant" in result.columns, "merchant column missing"
    assert "amt" in result.columns, "amt column missing"
    assert "trans_date_trans_time" in result.columns, "trans_date_trans_time column missing"
    
    # Test merchant_1 transactions
    merch_1_txns = result[result["merchant"] == "merchant_1"].sort_values("trans_date_trans_time")
    
    # First transaction (09:30) - no previous transactions
    first_txn = merch_1_txns.iloc[0]
    assert first_txn["merchant_txn_count_1h"] == 0, "First transaction should have 0 count"
    assert first_txn["merchant_txn_count_24h"] == 0, "First transaction should have 0 count"
    assert first_txn["merchant_avg_amt_24h"] == 0, "First transaction should have 0 average"
    
    # Second transaction (10:00) - only 09:30 in previous hour
    second_txn = merch_1_txns.iloc[1]
    assert second_txn["merchant_txn_count_1h"] == 1, "Should count previous 09:30 transaction"
    assert second_txn["merchant_txn_count_24h"] == 1, "Should count previous 09:30 transaction"
    assert second_txn["merchant_avg_amt_24h"] == 100.0, "Average should be 100.0"
    
    # Third transaction (10:30) - 09:30 and 10:00 in previous hour
    third_txn = merch_1_txns.iloc[2]
    assert third_txn["merchant_txn_count_1h"] == 2, "Should count both 09:30 and 10:00"
    assert third_txn["merchant_txn_count_24h"] == 2, "Should count both 09:30 and 10:00"
    assert third_txn["merchant_avg_amt_24h"] == 125.0, "Average should be (100+150)/2"
    
    # Test merchant_2 transaction (12:00) - no previous transactions
    merch_2_txn = result[result["merchant"] == "merchant_2"].iloc[0]
    assert merch_2_txn["merchant_txn_count_1h"] == 0, "Merchant_2 has no previous transactions"
    assert merch_2_txn["merchant_txn_count_24h"] == 0, "Merchant_2 has no previous transactions"
    assert merch_2_txn["merchant_avg_amt_24h"] == 0, "Merchant_2 has no previous transactions"
    
    print("✅ test_merchant_features passed!")


def test_historical_features():
    """Test building all historical features together."""
    df = create_test_data()
    result = build_historical_features(df)
    
    # Check that both card and merchant features exist
    all_features = [
        "card_txn_count_1h",
        "card_txn_count_24h",
        "card_avg_amt_24h",
        "card_max_amt_24h",
        "merchant_txn_count_1h",
        "merchant_txn_count_24h",
        "merchant_avg_amt_24h",
    ]
    
    for feature in all_features:
        assert feature in result.columns, f"Missing feature: {feature}"
    
    # Verify original columns are preserved
    assert "cc_num" in result.columns, "cc_num column missing"
    assert "merchant" in result.columns, "merchant column missing"
    assert "amt" in result.columns, "amt column missing"
    assert "trans_date_trans_time" in result.columns, "trans_date_trans_time column missing"
    
    # Verify data integrity
    assert len(result) == len(df), "Data length changed"
    assert set(result["cc_num"]) == set(df["cc_num"]), "cc_num values changed"
    assert set(result["merchant"]) == set(df["merchant"]), "merchant values changed"
    
    # Verify both card and merchant features are correctly combined
    # Check transaction at 10:00 for merchant_1 (should have both features)
    txn_1000 = result[
        (result["merchant"] == "merchant_1") & 
        (result["trans_date_trans_time"] == pd.Timestamp("2026-01-01 10:00:00"))
    ].iloc[0]
    
    # Card features for card_1 at 10:00
    assert txn_1000["card_txn_count_1h"] == 1, "Card feature should be 1"
    assert txn_1000["card_avg_amt_24h"] == 100.0, "Card average should be 100.0"
    
    # Merchant features for merchant_1 at 10:00
    assert txn_1000["merchant_txn_count_1h"] == 1, "Merchant feature should be 1"
    assert txn_1000["merchant_avg_amt_24h"] == 100.0, "Merchant average should be 100.0"
    
    # Check transaction at 10:30 for merchant_1 (card_2)
    txn_1030 = result[
        (result["merchant"] == "merchant_1") & 
        (result["trans_date_trans_time"] == pd.Timestamp("2026-01-01 10:30:00"))
    ].iloc[0]
    
    # Card_2 has no previous transactions
    assert txn_1030["card_txn_count_1h"] == 0, "Card_2 should have 0 previous transactions"
    assert txn_1030["card_avg_amt_24h"] == 0, "Card_2 should have 0 average"
    
    # Merchant_1 has 2 previous transactions (09:30 and 10:00)
    assert txn_1030["merchant_txn_count_1h"] == 2, "Merchant_1 should have 2 previous transactions"
    assert txn_1030["merchant_avg_amt_24h"] == 125.0, "Merchant average should be 125.0"
    
    print("✅ test_historical_features passed!")
    
def test_no_point_in_time_leakage():
    """Test that historical features never use current or future transactions."""
    df = create_test_data()
    result = add_card_features(df)

    # Get card_1 transactions sorted by time
    card_1_txns = result[
        result["cc_num"] == "card_1"
    ].sort_values("trans_date_trans_time")

    # Transaction at 10:00
    txn_1000 = card_1_txns[
        card_1_txns["trans_date_trans_time"]
        == pd.Timestamp("2026-01-01 10:00:00")
    ].iloc[0]

    # Only the 09:30 transaction should be visible.
    assert txn_1000["card_txn_count_24h"] == 1
    assert txn_1000["card_avg_amt_24h"] == 100.0
    assert txn_1000["card_max_amt_24h"] == 100.0

    # The current transaction (150.0) must not be included.
    # If it were included, the average would be 125.0.
    assert txn_1000["card_avg_amt_24h"] != 125.0

    print("✅ test_no_point_in_time_leakage passed!")

def test_project_setup():
    """Test that the project is properly set up."""
    import os
    import sys
    
    # Verify we're in the right directory structure
    assert os.path.exists("src/features/transformations.py"), "Source directory not found"
    assert os.path.exists("tests/test_features.py"), "Test directory not found"
    
    # Verify pandas version
    assert pd.__version__ is not None
    
    # Verify all required functions exist
    assert callable(add_card_features), "add_card_features not callable"
    assert callable(add_merchant_features), "add_merchant_features not callable"
    assert callable(build_historical_features), "build_historical_features not callable"
    
    print("✅ test_project_setup passed!")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("Running Feature Engineering Tests...")
    print("=" * 60)
    print()
    
    test_project_setup()
    test_card_features()
    test_merchant_features()
    test_historical_features()
    
    print()
    print("=" * 60)
    print("🎉 All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()