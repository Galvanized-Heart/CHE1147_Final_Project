from typing import List
from pathlib import Path
from dataclasses import dataclass, field
import copy
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Consider pearson or ther correl metrics
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

from config import ExperimentConfig, NUM_CROSSVAL_FOLDS, BASIC_FEATURE_COLS, TEMP_PH_FEATURE_COLS, ADVANCED_FEATURE_COLS, BASIC_TARGET_COLS, get_feature_cols, get_target_cols


def make_model_pipeline(model, normalize: bool, norm_feature_cols: list):
    to_scale = []
    if normalize:
        to_scale = norm_feature_cols
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), to_scale)
        ],
        remainder='passthrough' 
    )

    pipeline_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    target_scaler = None
    if normalize:
        target_scaler = StandardScaler()
    
    final_model = TransformedTargetRegressor(
        regressor=pipeline_model,
        transformer=target_scaler
    )

    return final_model


class BaseWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, normalize: bool = None, norm_feature_cols: list = None):
        super().__init__()
        self.normalize = normalize
        self.norm_feature_cols = norm_feature_cols
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass


class LinearWrapper(BaseWrapper):
    def __init__(self, normalize: bool = None, norm_feature_cols: list = None):
        super().__init__(normalize=normalize, norm_feature_cols=norm_feature_cols)
                
    def fit(self, X, y):
        self.model_ = LinearRegression()
        self.model_ = make_model_pipeline(self.model_, self.normalize, self.norm_feature_cols)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class XGBWrapper(BaseWrapper):
    def __init__(self, n_estimators: int = None, max_depth: int = None, learning_rate: float = None, normalize: bool = None, norm_feature_cols: list = None):
        super().__init__(normalize=normalize, norm_feature_cols=norm_feature_cols)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
     
    def fit(self, X, y):
        self.model_ = MultiOutputRegressor(XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=0,
            verbosity=0,
        ))
        self.model_ = make_model_pipeline(self.model_, self.normalize, self.norm_feature_cols)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


class MLPWrapper(BaseWrapper):
    def __init__(self, hidden_layer_1: int = None, hidden_layer_2: int = None, activation: str = None, solver: str = None, 
                 learning_rate_init: float = None, max_iter: int = None, early_stopping: bool = None, n_iter_no_change: int = None, 
                 normalize: bool = None, norm_feature_cols: list = None):

        super().__init__(normalize=normalize, norm_feature_cols=norm_feature_cols)
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.activation = activation
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
    
    def fit(self, X, y):
        self.model_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_layer_1, self.hidden_layer_2),
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            random_state=0,
        )
        self.model_ = make_model_pipeline(self.model_, self.normalize, self.norm_feature_cols)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


@dataclass
class ParityData:
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class ModelMetrics:
    # Each list will store results from the k-folds.
    # Each entry in the list will be a list of metric values, one for each target.
    # e.g., val_mse = [[mse_fold1_target1, mse_fold1_target2], [mse_fold2_target1, mse_fold2_target2], ...]
    train_mse: List[List[float]] = field(default_factory=list)
    val_mse: List[List[float]] = field(default_factory=list)
    train_mae: List[List[float]] = field(default_factory=list)
    val_mae: List[List[float]] = field(default_factory=list)
    train_r2: List[List[float]] = field(default_factory=list)
    val_r2: List[List[float]] = field(default_factory=list)


def train_and_eval(model_metrics: ModelMetrics, parity_data: ParityData, model: BaseWrapper, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate metrics for each target separately by using multioutput='raw_values'
    train_mse = mean_squared_error(y_train, y_train_pred, multioutput='raw_values')
    val_mse = mean_squared_error(y_val, y_val_pred, multioutput='raw_values')
    train_mae = mean_absolute_error(y_train, y_train_pred, multioutput='raw_values')
    val_mae = mean_absolute_error(y_val, y_val_pred, multioutput='raw_values')
    train_r2 = r2_score(y_train, y_train_pred, multioutput='raw_values')
    val_r2 = r2_score(y_val, y_val_pred, multioutput='raw_values')

    # Save metrics by appending the list of per-target metrics for this fold
    model_metrics.train_mse.append(train_mse.tolist())
    model_metrics.val_mse.append(val_mse.tolist())
    model_metrics.train_mae.append(train_mae.tolist())
    model_metrics.val_mae.append(val_mae.tolist())
    model_metrics.train_r2.append(train_r2.tolist())
    model_metrics.val_r2.append(val_r2.tolist())

    # Save parity data, handling the initial empty array case
    if parity_data.y_pred.size == 0:
        parity_data.y_pred = y_val_pred
        parity_data.y_true = y_val
    else:
        parity_data.y_pred = np.concatenate([parity_data.y_pred, y_val_pred], axis=0)
        parity_data.y_true = np.concatenate([parity_data.y_true, y_val], axis=0)


def single_model_experiment(
    df: pd.DataFrame,
    exp_config: ExperimentConfig,
    model: BaseWrapper
    ):
    kf = KFold(n_splits=NUM_CROSSVAL_FOLDS, shuffle=True, random_state=0)

    model_metrics = ModelMetrics()
    parity_data = ParityData(
        y_true=np.array([]),
        y_pred=np.array([])
    )

    basic_feature_cols, advanced_feature_cols = get_feature_cols(exp_config)
    target_cols = get_target_cols(exp_config)

    model.normalize = exp_config.normalize
    model.norm_feature_cols = basic_feature_cols

    total_feat_cols = basic_feature_cols + advanced_feature_cols
    
    X = df[total_feat_cols]
    y = df[target_cols]

    for (kfold_index, (train_index, test_index)) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        train_and_eval(model_metrics=model_metrics, parity_data=parity_data, model=model, X_train=X_train, y_train=y_train, X_val=X_test, y_val=y_test)
    
    model.fit(X, y)
    
    return {
        "model": model,
        "metrics": model_metrics,
        "parity_data": parity_data
    }


def single_experiment(
    df: pd.DataFrame,
    exp_config: ExperimentConfig,
    xgb_params: dict,
    nn_params: dict,
    ):
    """
    Want:
    1. Returns dict with each metric for scatter plots
    2. Predicted vs actual data for parity plots
    3. Fully trained models for feature importance plots
    """
    basic_feature_cols, _ = get_feature_cols(exp_config)
    linear_model = LinearWrapper(normalize=exp_config.normalize, norm_feature_cols=basic_feature_cols)
    xgb_model = XGBWrapper(
        n_estimators=xgb_params['n_estimators'], 
        max_depth=xgb_params['max_depth'], 
        learning_rate=xgb_params['learning_rate'], 
        normalize=exp_config.normalize, 
        norm_feature_cols=basic_feature_cols
    )
    nn_model = MLPWrapper(
        hidden_layer_1=nn_params['hidden_layer_1'],
        hidden_layer_2=nn_params['hidden_layer_2'],
        activation=nn_params['activation'],
        solver=nn_params['solver'],
        learning_rate_init=nn_params['learning_rate_init'],
        max_iter=nn_params['max_iter'],
        early_stopping=nn_params['early_stopping'],
        n_iter_no_change=nn_params['n_iter_no_change'],
        normalize=exp_config.normalize,
        norm_feature_cols=basic_feature_cols
    )

    linear_results = single_model_experiment(df=df, exp_config=exp_config, model=linear_model)
    xgb_results = single_model_experiment(df=df, exp_config=exp_config, model=xgb_model)
    nn_results = single_model_experiment(df=df, exp_config=exp_config, model=nn_model)

    return {
        "linear": linear_results,
        "xgb": xgb_results,
        "nn": nn_results,
    }


