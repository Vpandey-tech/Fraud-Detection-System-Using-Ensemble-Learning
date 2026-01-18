# 📊 Dataset Information

## Source
**Credit Card Fraud Detection Dataset**  
Available on Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Description
This dataset contains credit card transactions made by European cardholders in September 2013.

### Statistics
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.172% of all transactions)
- **Legitimate Transactions:** 284,315 (99.828%)
- **Time Period:** 2 days
- **Features:** 30 numerical features + 1 class label

### Features
| Feature | Description | Type |
|---------|-------------|------|
| `Time` | Seconds elapsed between this transaction and the first transaction | Numerical |
| `V1-V28` | Principal Component Analysis (PCA) transformed features | Numerical (Anonymized) |
| `Amount` | Transaction amount | Numerical |
| `Class` | Target variable (0 = Legitimate, 1 = Fraud) | Binary |

**Note:** Features V1-V28 are the result of PCA transformation to protect user identities and sensitive features. The original features are confidential.

## Class Imbalance
This dataset is **highly imbalanced**:
- Fraud: 0.172%
- Legitimate: 99.828%

This is why we use:
- **CTGAN** for generating synthetic fraud samples
- **Class weights** in model training
- **AUPRC** (Area Under Precision-Recall Curve) instead of accuracy as the primary metric

## Setup Instructions

### Option 1: Download from Kaggle (Recommended)
1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click "Download" (requires free Kaggle account)
3. Extract `creditcard.csv`
4. Place it in this `data/` directory

### Option 2: Use Kaggle API
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials (get from kaggle.com/account)
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\Users\<username>\.kaggle\ (Windows)

# Download dataset
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/
```

### Option 3: Use Pre-trained Models (No Data Needed)
If you just want to run the application without retraining:
- **No dataset download required!**
- Pre-trained models are already included in `src/model/`
- Just install dependencies and run the app

## File Structure
```
data/
├── README.md              # This file
├── creditcard.csv         # Download and place here (not included in repo)
└── .gitignore             # Prevents accidental upload of large CSV
```

## Usage in Training Scripts

### Basic Training
```bash
python src/training/train.py --data data/creditcard.csv
```

### Advanced GAN-based Training
```bash
python notebooks/gan_training.py
# Note: Expects creditcard.csv in the same directory or modify path in script
```

### Threshold Optimization
```bash
python notebooks/find_optimal_threshold.py
```

## Data Preprocessing
Our system applies the following preprocessing:
1. **Drop Time column** (not useful for fraud pattern detection)
2. **Scale Amount** using `RobustScaler` (handles outliers better than StandardScaler)
3. **Keep V1-V28** as-is (already PCA-transformed)
4. **Generate synthetic fraud** using CTGAN (for training only)

## Privacy & Ethics
- This is a **public research dataset** provided for academic purposes
- All sensitive information has been removed via PCA transformation
- No personal identifiable information (PII) is present
- Use responsibly and ethically

## Citation
If you use this dataset, please cite:
```
Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi.
Calibrating Probability with Undersampling for Unbalanced Classification.
In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015
```

## Troubleshooting

### "File not found: creditcard.csv"
- Make sure you downloaded the dataset
- Check that the file is in the `data/` directory
- Verify the filename is exactly `creditcard.csv` (case-sensitive on Linux/Mac)

### "Dataset too large / Out of memory"
- The CSV is ~150 MB
- Requires ~2 GB RAM for processing
- Consider using a subset for testing:
  ```python
  df = pd.read_csv('data/creditcard.csv', nrows=50000)  # First 50k rows
  ```

### "Permission denied"
- Ensure you have read permissions on the file
- On Windows, check if file is open in Excel or another program

## Alternative Datasets
If you cannot access Kaggle, similar fraud detection datasets:
- IEEE-CIS Fraud Detection (Kaggle)
- Synthetic Financial Datasets (Paysim)
- Credit Card Fraud Detection (UCI ML Repository)

---

**Last Updated:** January 15, 2026  
**Maintained by:** Vivek Pandey
