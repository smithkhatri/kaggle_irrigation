# Crop Irrigation Need Prediction

A machine learning pipeline that predicts whether a field needs **Low**, **Medium**, or **High** irrigation based on soil, weather, and crop features. Built for a Kaggle-style tabular classification competition.

## Overview

Farmers and irrigation systems need to decide how much water a field requires. This project frames that decision as a 3-class classification problem and solves it with a gradient-boosted tree model (LightGBM), reaching **97.98% validation accuracy**.

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Low    | 0.98      | 0.99   | 0.99     | 3,755   |
| Medium | 0.98      | 0.97   | 0.97     | 2,355   |
| High   | 0.90      | 0.92   | 0.91     | 190     |

The dataset is heavily imbalanced (only ~3% of samples are "High" need), so class weighting was tuned to keep recall on the minority class high without sacrificing overall accuracy.

## Dataset

- **Train:** 630,000 rows / **Test:** 270,000 rows
- Each row describes a field at a point in time: soil properties, weather conditions, crop info, and irrigation setup.

| Feature | Description |
|---|---|
| `Soil_Type` | Categorical soil classification (Clay, Loamy, Sandy, Silt) |
| `Soil_pH`, `Soil_Moisture`, `Organic_Carbon`, `Electrical_Conductivity` | Soil chemistry/composition readings |
| `Temperature_C`, `Humidity`, `Rainfall_mm`, `Sunlight_Hours`, `Wind_Speed_kmh` | Weather conditions |
| `Crop_Type`, `Crop_Growth_Stage`, `Season` | Crop and cultivation context |
| `Irrigation_Type`, `Water_Source`, `Field_Area_hectare`, `Mulching_Used`, `Previous_Irrigation_mm` | Irrigation setup and history |
| `Region` | Geographic region |
| `Irrigation_Need` | **Target** — Low / Medium / High |

## Approach

1. **Preprocessing**
   - Target label (`Irrigation_Need`) mapped to ordinal integers (Low=0, Medium=1, High=2).
   - Categorical features (`Region`, `Water_Source`, `Irrigation_Type`, `Season`, `Crop_Growth_Stage`, `Crop_Type`, `Soil_Type`) one-hot encoded.
   - `Mulching_Used` mapped from Yes/No to 1/0.
2. **Validation split** — last 1% of training rows held out as a validation set.
3. **Model** — `lightgbm.LGBMClassifier` with:
   - 500 estimators, depth 8, 63 leaves, learning rate 0.05
   - Row/column subsampling (0.8) to reduce overfitting
   - Class weights `{Low: 1, Medium: 1.5, High: 17}` to counter the strong class imbalance
   - Early stopping on validation multi-logloss
4. **Evaluation** — accuracy and per-class precision/recall/F1 on the held-out validation split.

## Results

- **Validation accuracy: 97.98%**
- Strong performance across all three classes, including the rare "High" irrigation need class (F1 = 0.91), thanks to class weighting.
- Final predictions are written to `submission.csv` in the format expected by the competition.

## Tech Stack

- Python, pandas, NumPy
- LightGBM (gradient-boosted trees)
- scikit-learn (metrics)

## Setup

```bash
pip install pandas numpy lightgbm scikit-learn jupyter
```

Place `train.csv` and `test.csv` in the project root, then run [main.ipynb](main.ipynb) top to bottom. The notebook will print validation accuracy and a classification report, and write `submission.csv`.

## Repo Structure

```
main.ipynb       # data loading, preprocessing, training, evaluation, submission
README.md
```

## Possible Improvements

- Hyperparameter tuning (Optuna/grid search) over tree depth, leaves, and learning rate.
- Try alternative models (XGBoost, CatBoost, or a small neural net) and ensemble with LightGBM.
- Feature engineering: interaction terms between soil moisture/rainfall and crop growth stage.
- K-fold cross-validation instead of a single held-out split for a more robust accuracy estimate.
