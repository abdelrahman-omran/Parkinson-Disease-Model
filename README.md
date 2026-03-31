# Parkinson's Disease Model

Machine learning models for Parkinson's disease detection and analysis.

## Overview

This project tackles Parkinson's disease from two angles:

- **Regression** — predicts UPDRS (disease severity) scores from patient data
- **Classification** — diagnoses whether a patient has Parkinson's disease or not

Both models are trained, saved as `.pkl` files, and evaluated through a shared test script.

## Project Structure

```
├── classification/
│   └── classification_model.pkl   # Trained classification model
├── regression/
│   └── regression_model.pkl       # Trained regression model
├── test_script.py                 # Loads models and evaluates on test data
├── conda_env                      # Conda environment file
└── README.md
```

## Getting Started

### Set up the environment

```bash
conda create --name parkinson-env --file conda_env
conda activate parkinson-env
```

### Run the test script

Place your test CSVs (`regression_test.csv`, `classification_test.csv`) in the root directory, then run:

```bash
python test_script.py
```

This will print evaluation metrics for both models.

## Evaluation Metrics

| Task           | Metric                        |
|----------------|-------------------------------|
| Regression     | Mean Squared Error, R² Score  |
| Classification | Accuracy                      |

## Dependencies

- `scikit-learn`
- `pandas`
- `numpy`
- `scipy`
- `matplotlib`

See `conda_env` for the full pinned environment.

## License

MIT
