# Customer Churn Analysis and Prediction

A beginner-friendly data science portfolio project that explores customer churn patterns and compares two baseline classification models. The CSV contains **synthetic teaching data only**; it does not describe real customers or a real company.

## Project goals

- Explore how churn differs across customer groups.
- Prepare numeric and categorical data with a reproducible preprocessing pipeline.
- Train and compare logistic regression and random forest classifiers.
- Evaluate predictions with accuracy, precision, recall, F1 score, and a confusion matrix.

## Dataset

`data/customer_churn.csv` contains 600 generated sample records. Features include age, tenure, monthly charges, contract type, internet service, paperless billing, and technical support. `churn` is the target column (`Yes` or `No`). The data is generated deterministically with a fixed random seed so results can be reproduced.

The synthetic churn labels are generated from simulated relationships between selected features and churn. They are useful for practicing a workflow, but results should not be interpreted as business findings or evidence about real customers.

## Requirements

Python 3.10 or newer is recommended.

## Run locally

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).

   Windows PowerShell:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install packages and run the analysis:
   ```bash
   python -m pip install -r requirements.txt
   python src/analyze_churn.py
   ```

The script prints a short data-quality summary and model metrics. It creates `outputs/churn_by_contract.png` and `outputs/model_metrics.txt`; the `outputs/` folder is excluded from Git by `.gitignore`.

## Tools and concepts

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, data cleaning, exploratory data analysis, one-hot encoding, feature scaling, train/test split, classification, and model evaluation.

## Possible extensions

- Add cross-validation and tune model parameters.
- Compare precision and recall at different decision thresholds.
- Add a feature importance discussion, while noting the limits of synthetic data.
- Replace the sample CSV only with a properly licensed public dataset and update this README to cite its source.

## License

This project is provided for learning and portfolio demonstration.
