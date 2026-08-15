# Feature Specification

## 1. Overview

This document defines the entities, feature groups, and windowed
historical features used by the fraud detection feature pipeline.

The feature pipeline separates:

- Static features produced during Data Science preprocessing.
- Historical, time-aware features produced by the MLOps feature pipeline.

Historical features must be calculated using only information available
before the transaction event time.

---

## 2. Source Data

Primary source:

`dataset/fraudTrain.csv`

Important source columns:

- `cc_num` - card identifier
- `merchant` - merchant identifier
- `trans_date_trans_time` - transaction event timestamp
- `amt` - transaction amount
- `is_fraud` - target label

---

## 3. Entities

### 3.1 Card Entity

**Entity name:** `card`

**Entity key:** `cc_num`

**Event timestamp:** `trans_date_trans_time`

The card entity represents the card/account associated with a transaction.

Historical card-level features describe the recent transaction behavior
of the same card before the current transaction.

---

### 3.2 Merchant Entity

**Entity name:** `merchant`

**Entity key:** `merchant`

**Event timestamp:** `trans_date_trans_time`

The merchant entity represents the merchant associated with a transaction.

Historical merchant-level features describe recent transaction activity
associated with the same merchant before the current transaction.

---

## 4. Feature Groups

### 4.1 Card Historical Features

Feature group:

`card_historical_features`

Features:

- `card_txn_count_1h`
- `card_txn_count_24h`
- `card_avg_amt_24h`
- `card_max_amt_24h`

---

### 4.2 Merchant Historical Features

Feature group:

`merchant_historical_features`

Features:

- `merchant_txn_count_1h`
- `merchant_txn_count_24h`
- `merchant_avg_amt_24h`

---

## 5. Windowed Feature Specifications

| Feature | Entity | Window | Aggregation | Source | Time Rule |
|---|---|---|---|---|---|
| `card_txn_count_1h` | Card | 1 hour | Count | transactions | `event_time < current_time` |
| `card_txn_count_24h` | Card | 24 hours | Count | transactions | `event_time < current_time` |
| `card_avg_amt_24h` | Card | 24 hours | Mean | `amt` | `event_time < current_time` |
| `card_max_amt_24h` | Card | 24 hours | Max | `amt` | `event_time < current_time` |
| `merchant_txn_count_1h` | Merchant | 1 hour | Count | transactions | `event_time < current_time` |
| `merchant_txn_count_24h` | Merchant | 24 hours | Count | transactions | `event_time < current_time` |
| `merchant_avg_amt_24h` | Merchant | 24 hours | Mean | `amt` | `event_time < current_time` |

---

## 6. Point-in-Time Correctness

All historical features must obey the following rule:

> A feature calculated for transaction `T` may only use transactions that
> occurred before `T`.

The current transaction must not be included in its own historical
features.

Future transactions must never contribute to a historical feature.

For example, for a transaction occurring at `10:00`:

- A transaction at `09:30` may be included.
- A transaction at `09:45` may be included.
- A transaction at `10:00` must not be included.
- A transaction at `10:30` must not be included.

This rule prevents point-in-time data leakage.

---

## 7. Existing Data Science Features

The following features are already produced by the Data Science
preprocessing pipeline and are not reimplemented as part of the
historical feature pipeline:

- `hour`
- `day_of_week`
- `month`
- `age`
- `merchant_distance`
- `job_freq`
- `merchant_freq`

These features remain part of the model-ready dataset.

The MLOps feature pipeline focuses on time-aware historical features.

---

## 8. Global Frequency Features

The existing `job_freq` and `merchant_freq` features are calculated
using dataset-level frequency counts.

They are treated as existing Data Science features and are not considered
windowed historical features.

Future iterations may replace them with point-in-time historical
frequency features if required.

---

## 9. Feature Naming Convention

Historical features follow:

`<entity>_<aggregation>_<window>`

Examples:

- `card_txn_count_1h`
- `card_txn_count_24h`
- `merchant_txn_count_24h`
- `card_avg_amt_24h`

---

## 10. Expected Output

The feature transformation pipeline will produce one row per transaction
with the original transaction identifier and the calculated historical
features.

The resulting features will be used as inputs to downstream model
training and inference.