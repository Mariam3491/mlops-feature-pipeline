"""
Feature engineering module for transaction data.
Creates point-in-time rolling features for cards and merchants.
"""

import pandas as pd


def _add_rolling_features(
    df: pd.DataFrame,
    entity_column: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Add point-in-time rolling transaction features for an entity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with transaction data.
    entity_column : str
        Column name to group by (e.g., 'cc_num' or 'merchant').
    prefix : str
        Prefix for feature names ('card' or 'merchant').
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added rolling features.
    """
    result = df.copy()

    # Ensure datetime format
    result["trans_date_trans_time"] = pd.to_datetime(
        result["trans_date_trans_time"]
    )

    # Sort for proper rolling window calculation
    result = result.sort_values(
        [entity_column, "trans_date_trans_time"]
    ).copy()

    def calculate_group(group: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling features for a single group."""
        group = group.sort_values("trans_date_trans_time").copy()

        # Set index for time-based rolling
        rolling = group.set_index("trans_date_trans_time")["amt"]

        # Create features with closed="left" (exclude current transaction)
        group[f"{prefix}_txn_count_1h"] = (
            rolling.rolling("1h", closed="left").count().fillna(0).to_numpy()
        )
        group[f"{prefix}_txn_count_24h"] = (
            rolling.rolling("24h", closed="left").count().fillna(0).to_numpy()
        )

        if prefix == "card":
            group[f"{prefix}_avg_amt_24h"] = (
                rolling.rolling("24h", closed="left").mean().fillna(0).to_numpy()
            )
            group[f"{prefix}_max_amt_24h"] = (
                rolling.rolling("24h", closed="left").max().fillna(0).to_numpy()
            )
        elif prefix == "merchant":
            group[f"{prefix}_avg_amt_24h"] = (
                rolling.rolling("24h", closed="left").mean().fillna(0).to_numpy()
            )

        return group

    # Process groups without warnings
    grouped = result.groupby(entity_column, group_keys=False)
    processed_groups = []
    
    for _, group in grouped:
        processed_groups.append(calculate_group(group))
    
    result = pd.concat(processed_groups, ignore_index=True)
    
    return result


def add_card_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add point-in-time historical features for each card.
    
    Features added:
    - card_txn_count_1h: Number of card transactions in previous hour
    - card_txn_count_24h: Number of card transactions in previous 24 hours
    - card_avg_amt_24h: Average transaction amount for card in previous 24 hours
    - card_max_amt_24h: Maximum transaction amount for card in previous 24 hours
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with transaction data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with card-level features added.
    """
    return _add_rolling_features(
        df,
        entity_column="cc_num",
        prefix="card",
    )


def add_merchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add point-in-time historical features for each merchant.
    
    Features added:
    - merchant_txn_count_1h: Number of merchant transactions in previous hour
    - merchant_txn_count_24h: Number of merchant transactions in previous 24 hours
    - merchant_avg_amt_24h: Average transaction amount for merchant in previous 24 hours
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with transaction data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with merchant-level features added.
    """
    return _add_rolling_features(
        df,
        entity_column="merchant",
        prefix="merchant",
    )


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all historical card and merchant features.
    
    This function sequentially adds card-level and merchant-level
    rolling features to the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with transaction data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all historical features added.
    
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'cc_num': ['card_1', 'card_1'],
    ...     'merchant': ['merch_1', 'merch_1'],
    ...     'trans_date_trans_time': ['2026-01-01 10:00:00', '2026-01-01 11:00:00'],
    ...     'amt': [100.0, 150.0]
    ... })
    >>> result = build_historical_features(df)
    >>> result.columns.tolist()
    ['cc_num', 'merchant', 'trans_date_trans_time', 'amt', 
     'card_txn_count_1h', 'card_txn_count_24h', 'card_avg_amt_24h', 
     'card_max_amt_24h', 'merchant_txn_count_1h', 'merchant_txn_count_24h', 
     'merchant_avg_amt_24h']
    """
    result = add_card_features(df)
    result = add_merchant_features(result)
    return result