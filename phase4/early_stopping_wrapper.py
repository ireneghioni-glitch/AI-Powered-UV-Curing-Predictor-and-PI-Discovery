''' 
==================== EARLY STOPPING WRAPPER CLASS ====================

To implement hyperparameter tuning with **GridSearch** and 
make it compatible with the Early Stopping strategy.  

The wrapper class below:  
1. receives the `model`, `eval_set` (Validation set for monitoring) 
   and `early_stopping_rounds`.
2. does override of fit() method to include early stopping.
3. gives back the trained model with optimal number of trees.
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import warnings


class EarlyStoppingXGBWrapper(BaseEstimator, RegressorMixin):
    '''
    A scikit-learn compatible wrapper for XGBoost that enables early stopping
    during hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
    
    This wrapper passes the eval_set (validation data) to the XGBoost fit method,
    allowing early stopping to work inside cross-validation folds.
    '''
    def __init__(
        self,
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=10,
        random_state=None,
        verbosity=0,
        eval_set=None,  # <-- Validation set given as parameter
        **kwargs
    ):
        '''
        Initialize the wrapper with XGBoost hyperparameters.
        
        All parameters are stored as instance attributes and passed to the
        underlying XGBRegressor during fit(). This allows GridSearchCV to
        modify them via the get_params() and set_params() methods.
        
        Args:
            n_estimators (int): Maximum number of trees (early stopping will reduce this)
            learning_rate (float): Step size shrinkage
            max_depth (int): Maximum tree depth
            subsample (float): Fraction of samples used per tree
            colsample_bytree (float): Fraction of features used per tree
            reg_alpha (float): L1 regularization penalty
            reg_lambda (float): L2 regularization penalty
            early_stopping_rounds (int): Stop if validation loss doesn't improve for this many rounds
            random_state (int): Random seed for reproducibility
            verbosity (int): Level of XGBoost output (0 = silent)
            eval_set (list): Validation set for early stopping, e.g., [(X_val, y_val)]
            **kwargs: Additional XGBoost parameters
        '''
        # Store all parameters as attributes for scikit-learn compatibility
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.verbosity = verbosity
        self.eval_set = eval_set
        self.kwargs = kwargs
        self.best_iteration_ = None

    def fit(self, X, y, **fit_params):
        '''
        Train the XGBoost model with early stopping support.
        
        This method is called by GridSearchCV during cross-validation.
        It creates the underlying XGBRegressor, passes the validation set
        (from eval_set), and trains with early stopping.
        
        Args:
            X (array-like): Training features
            y (array-like): Training targets
            **fit_params: Additional fit parameters, typically containing 'eval_set'
            
        Returns:
            self: The fitted wrapper instance
        '''
        # Create the underlying XGBoost model with all stored parameters
        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbosity=self.verbosity,
            early_stopping_rounds=self.early_stopping_rounds,
            **self.kwargs
    )

        # Extract eval_set from fit_params (passed by GridSearchCV)
        # or use the one stored in self.eval_set
        eval_set = fit_params.get('eval_set', self.eval_set)

        # If validation set is available, train with early stopping
        if eval_set is not None:
            self.model_.fit(
                X, y,
                eval_set=eval_set,
                verbose=False
            )
            # Store the optimal number of trees for later inspection
            self.best_iteration = self.model_.best_iteration
        else:
            # fallback for no eval_set given (no early stopping)
            warnings.warn("No eval_set provided. Training without early stopping.")
            self.model_.fit(X, y)
            self.best_iteration_ = self.n_estimators

        return self

    def predict(self, X):
        '''
        Make predictions using the fitted XGBoost model.
        
        Args:
            X (array-like): Features to predict on
            
        Returns:
            array-like: Predicted values
        '''
        return self.model_.predict(X)

    def get_params(self, deep=True):
        '''
        Get all parameters of the wrapper.
        
        This method is required by scikit-learn for GridSearchCV.
        It returns a dictionary of all hyperparameters that can be tuned.
        
        Args:
            deep (bool): If True, returns parameters of nested objects (unused here)
            
        Returns:
            dict: All parameters of the wrapper
        '''
        params = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'verbosity': self.verbosity,
            'eval_set': self.eval_set
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        '''
        Set parameters of the wrapper.
        
        This method is required by scikit-learn for GridSearchCV.
        It allows GridSearchCV to modify individual hyperparameters during tuning.
        
        Args:
            **params: Parameter names and values to set
            
        Returns:
            self: The wrapper instance with updated parameters
        '''
        for key, value in params.items():
            setattr(self, key, value)
        return self