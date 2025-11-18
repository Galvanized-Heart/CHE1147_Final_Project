import os
import pandas as pd
import json
import copy
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor

from config import HPO_ROUNDS, NUM_CROSSVAL_FOLDS, HPO_RESULTS_DIR, HPO_SEARCH_SPACES, ADVANCED_FEATURE_COLS, ExperimentConfig, BASIC_FEATURE_COLS, TEMP_PH_FEATURE_COLS, BASIC_TARGET_COLS, get_feature_cols, get_target_cols
from modeling.train import LinearWrapper, XGBWrapper, MLPWrapper, make_model_pipeline


MODELS = {
    "linear": LinearWrapper(),
    "xgb": XGBWrapper(),
    "nn": MLPWrapper(),
}


def bayes_hpo(model, param_search_space: dict, df: pd.DataFrame, exp_config: ExperimentConfig) -> dict:
    if param_search_space is None or len(param_search_space) == 0:
        return {}
    
    basic_feature_cols, advanced_feature_cols = get_feature_cols(exp_config)

    model.normalize = exp_config.normalize
    model.norm_feature_cols = basic_feature_cols

    feature_cols = basic_feature_cols + advanced_feature_cols 
    target_cols = get_target_cols(exp_config)

    X = df[feature_cols]
    y = df[target_cols]

    model_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_search_space,
        n_iter=HPO_ROUNDS,
        cv=NUM_CROSSVAL_FOLDS,
        n_jobs=-1,
        random_state=0,
        verbose=0,
    )

    model_search.fit(X, y)

    return model_search.best_params_


def bayes_hpo_pipeline(config_name: str, model, param_search_space: dict, df: pd.DataFrame, exp_config: ExperimentConfig) -> dict:
    os.makedirs(HPO_RESULTS_DIR, exist_ok=True)
    output_path = HPO_RESULTS_DIR / f"hpo_{config_name}.json"
    if output_path.exists():
        logger.info(f"Skipping HPO for {config_name}: results already exist at {output_path}")
        with open(output_path, "r") as f:
            best_params = json.load(f)
        return best_params
    
    logger.info(f"Running HPO for config: {config_name}")

    best_params = bayes_hpo(model, param_search_space, df=df, exp_config=exp_config)

    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Saved HPO results to: {output_path}")

    return best_params


def run_full_bayes_hpo(config_name: str, df: pd.DataFrame, exp_config: ExperimentConfig) -> dict:
    best_params_dict = {}
    for (model_name, model) in MODELS.items():
        param_search_space = HPO_SEARCH_SPACES[model_name]
        best_params = bayes_hpo_pipeline(
            config_name=f"{config_name}_{model_name}",
            model=model,
            param_search_space=param_search_space,
            df=df,
            exp_config=exp_config,
        )

        best_params_dict[model_name] = best_params
    
    return best_params_dict


