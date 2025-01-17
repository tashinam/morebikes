import glob
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score, KFold

def load_all_processed_data():
    all_files = glob.glob(os.path.join('Processed', '*.csv'))
    return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

def load_all_phase2_data():
    all_files = glob.glob(os.path.join('Processed/phase2', 'phase_2_station_*.csv'))
    return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

def load_all_phase3_data():
    all_files = glob.glob(os.path.join('Processed/phase3', 'phase_3_station_*.csv'))
    return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

class ManualFeatureSelector(TransformerMixin):
    """
    Transformer for manual selection of features using sklearn style transform method.
    """

    def __init__(self, features):
        self.features = features
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]

class Baseline(sklearn.base.BaseEstimator):
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y):
        pass
    def predict(self, X):
        return X[self.column]
    
def cross_val_grouped_means(regr):
    def fold_means(groupdf):
        scores = cross_val_score(regr, groupdf.drop(columns=['bikes']), groupdf.bikes, cv=KFold(n_splits=5, shuffle=True), scoring='neg_mean_absolute_error')

        return scores.mean()
    return fold_means

def cross_val_group_mean(regr, groups):
    return np.fromiter(map(cross_val_grouped_means(regr), groups), dtype=np.float).mean()

def per_station_models_cross_val_mean(regr, df):
    per_station_groups = [station_df.drop(columns=['station']) for station_id, station_df in df.groupby('station')]

    return cross_val_group_mean(regr, per_station_groups)

phase2_modelled_stations = np.arange(1, 201)
phase2_model_types = ['full_temp', 'full', 'short_full_temp', 'short_full', 'short_temp', 'short']
phase2_model_names = [f'model_station_{station_id}_rlm_{model_type}' for station_id in phase2_modelled_stations for model_type in phase2_model_types]

