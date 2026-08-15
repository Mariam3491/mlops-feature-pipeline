# Feature Documentation - Credit Card Fraud Detection

## Overview
This document provides comprehensive documentation for all features used in the credit card fraud detection model.

## Feature Categories

### 1. Transaction Features
| Feature | Type | Description | Transformation |
|---------|------|-------------|----------------|
| `amt` | Float | Transaction amount in USD | StandardScaler |
| `unix_time` | Float | Unix timestamp of transaction | StandardScaler |

### 2. Temporal Features
| Feature | Type | Description | Transformation |
|---------|------|-------------|----------------|
| `hour` | Float | Hour of transaction (0-23) | StandardScaler |
| `day_of_week` | Float | Day of week (0=Monday, 6=Sunday) | StandardScaler |
| `month` | Float | Month of transaction (1-12) | StandardScaler |
| `age` | Float | Cardholder's age at transaction time | StandardScaler |

### 3. Location Features
| Feature | Type | Description | Transformation |
|---------|------|-------------|----------------|
| `lat` | Float | Cardholder's latitude | StandardScaler |
| `long` | Float | Cardholder's longitude | StandardScaler |
| `merch_lat` | Float | Merchant's latitude | StandardScaler |
| `merch_long` | Float | Merchant's longitude | StandardScaler |
| `merchant_distance` | Float | Haversine distance between cardholder and merchant (km) | StandardScaler |

### 4. Categorical Encoded Features
| Feature | Type | Description | Encoding |
|---------|------|-------------|----------|
| `category` | Int | Transaction category | Label Encoding (14 classes) |
| `state` | Int | US State | Label Encoding (51 classes) |
| `gender` | Int | Cardholder gender (0=Female, 1=Male) | Binary Encoding |
| `job_freq` | Int | Frequency of job title in dataset | Frequency Encoding |
| `merchant_freq` | Int | Frequency of merchant in dataset | Frequency Encoding |

### 5. Target Variable
| Feature | Type | Description |
|---------|------|-------------|
| `is_fraud` | Int | Binary label (0=Normal, 1=Fraud) |

## Feature Engineering Details












### Haversine Distance Calculation
```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    # Returns distance in kilometers