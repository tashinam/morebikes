import glob
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score, KFold

def load_all_processed_data():
    all_files = all_files = glob.glob(os.path.join('Processed', '*.csv'))
    return pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

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

