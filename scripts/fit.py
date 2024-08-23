from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils import read_yaml, save_pkl


def fit_model() -> None:
    """
    Fits a machine learning model using the data from the
    "data/initial_data.csv" file and the hyperparameters
    defined in the "params.yaml" file.

    The model is a CatBoostClassifier with the following preprocessing
    steps:
    - Binary categorical features are one-hot encoded.
    - Other categorical features are encoded using CatBoostEncoder.
    - Numeric features are standardized using StandardScaler.
    - Other features are dropped.

    The fitted model is then saved to a file named "fitted_model.pkl".
    """

    # 1. Reading hyperparams
    params = read_yaml(path=Path("params.yaml"))

    # 2. Reading data
    data = pd.read_csv(
        Path("data/initial_data.csv"), parse_dates=["begin_date", "end_date"]
    )

    # 3. Main part with model fitting
    cat_features = data.select_dtypes(include="object")
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[
        potential_binary_features[potential_binary_features].index
    ].columns.tolist()
    other_cat_features = cat_features[
        potential_binary_features[~potential_binary_features].index
    ].columns.tolist()
    num_features = data.select_dtypes(["float"]).columns.tolist()
    del cat_features, potential_binary_features

    preprocessor = ColumnTransformer(
        [
            (
                "binary",
                OneHotEncoder(drop=params["one_hot_drop"]),
                binary_cat_features,
            ),
            (
                "cat",
                CatBoostEncoder(return_df=False),
                other_cat_features,
            ),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = CatBoostClassifier(
        auto_class_weights=params["auto_class_weights"],
        verbose=False,
    )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(data, data[params["target_col"]])

    # 4. Saving pipeline model
    save_pkl(model=pipeline, path=Path("models/fitted_model.pkl"))


if __name__ == "__main__":
    fit_model()
