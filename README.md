# Machine Learning Projects

Three end-to-end supervised learning projects. Each notebook loads a public dataset directly from a URL, documents every cleaning decision, and validates results against a stated baseline — no local data files are required.

| Notebook | Task | Approach | Result |
| --- | --- | --- | --- |
| [`Ames Housing — Linear Regression.ipynb`](Ames%20Housing%20%E2%80%94%20Linear%20Regression.ipynb) | Regression: house price prediction (Ames Housing, 2,930 rows) | Linear regression implemented from scratch in PyTorch, verified against scikit-learn at every stage; evidence-driven feature selection (correlation, VIF, ANOVA screening, polynomial term) | RMSE $32,348 / R² 0.87 (validation) |
| [`Titanic Survival — Logistic Regression.ipynb`](Titanic%20Survival%20%E2%80%94%20Logistic%20Regression.ipynb) | Binary classification: passenger survival (Titanic, 891 rows) | Logistic regression with statistically screened feature engineering, per-class evaluation, threshold selection; PyTorch embedding extension reported as a negative result | 84.9% validation accuracy, ROC-AUC 0.868 |
| [`Predicting Heart Disease Model Comparison.ipynb`](Predicting%20Heart%20Disease%20Model%20Comparison.ipynb) | Binary classification: heart disease presence (UCI Cleveland, 303 patients) | Logistic Regression, Decision Tree, Random Forest, and XGBoost compared under 5-fold cross-validation (accuracy, precision, recall, ROC-AUC), with feature importance and clinical-use caveats | Logistic regression recommended on cross-validated ROC-AUC |

## Setup

```
python -m venv venv
# Mac: source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is used for the from-scratch implementations only; the CPU build is sufficient and much smaller than the default download:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Run any notebook top to bottom with Jupyter:

```
jupyter notebook
```

## Notes

- All results are reproducible: random seeds are fixed (`random_state=42` / `random_state=17`, `torch.manual_seed(42)`), and every notebook has been re-executed end to end before commit.
- Validation figures come from held-out splits or cross-validation as stated in each notebook; limitations (single-split caveats, sample size, selection bias) are documented in the final section of each notebook.

## Data sources

- [Ames Housing dataset](https://raw.githubusercontent.com/STATCowboy/pbidataflowstalk/master/AmesHousing.csv)
- [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- [UCI Heart Disease (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) via [TensorFlow-hosted mirror](http://storage.googleapis.com/download.tensorflow.org/data/heart.csv)
