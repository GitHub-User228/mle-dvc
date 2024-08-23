from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from utils import read_yaml, read_pkl, save_dict_as_json


def evaluate_model():

    # 1. Reading hyperparams
    params = read_yaml(path=Path("params.yaml"))

    # 2. Reading pipeline and data
    pipeline = read_pkl(path=Path("models/fitted_model.pkl"))
    data = pd.read_csv(
        Path("data/initial_data.csv"), parse_dates=["begin_date", "end_date"]
    )

    # 3. Cross-validation
    cv_strategy = StratifiedKFold(n_splits=params["n_splits"])
    cv_res = cross_validate(
        pipeline,
        data,
        data[params["target_col"]],
        cv=cv_strategy,
        n_jobs=params["n_jobs"],
        scoring=params["metrics"],
    )
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)

    # 4. Saving cv results
    save_dict_as_json(data=cv_res, path=Path("cv_results/cv_res.json"))


if __name__ == "__main__":
    evaluate_model()
