import numpy as np
import pandas as pd
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, Ridge
from scipy.stats import zscore, mstats
from constrained_linear_regression import ConstrainedLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rolling_regression_sklearn_advanced(data, rolling_window, n_step_ahead=1, 
                                        l1_ratio=0.1, 
                                        dropna=False, remove_outliers=False, 
                                        winsorize=False, winsorize_limits=(0.05, 0.95),
                                        fit_intercept=False, min_coef=None, max_coef=None,
                                        expanding=False):
    """
    Perform rolling regression from sklearn with additional data processing options.
    
    Parameters:
        data (pd.DataFrame): DataFrame where one of the columns should be "target". 
                             Should have a DateTimeIndex.
        rolling_window (int): Number of samples to use for each regression.
        n_step_ahead (int, optional): Number of steps ahead to predict. Default is 1.
        l1_ratio (float, optional): The L1 regularization ratio. Default is 0.1.
        dropna (bool, optional): Whether to drop NaN values. Default is False.
        remove_outliers (bool, optional): Whether to remove outliers based on Z-score. Default is False.
        winsorize (bool, optional): Whether to winsorize data. Default is False.
        winsorize_limits (tuple, optional): Percentiles for winsorizing. Default is (0.05, 0.95).
    
    Returns:
        pd.DataFrame: Coefficients for each window.
        pd.Series: Predictions.
    """
    # Drop NaN values if requested
    datac = data.copy()
    if dropna:
        datac = datac.dropna()
    n_samples, n_features_plus_one = datac.shape
    n_features = n_features_plus_one - 1
    coefs = pd.DataFrame(index=datac.index, columns=datac.drop('target', axis=1).columns)
    predictions = pd.Series(index=datac.index, name='predictions')
    
    for start in range(0, n_samples - rolling_window - n_step_ahead + 1, n_step_ahead):
        if expanding:
            window = datac.iloc[0:start + rolling_window].copy()  # Use copy to avoid SettingWithCopyWarning
        else:
            window = datac.iloc[start:start + rolling_window].copy()  # Use copy to avoid SettingWithCopyWarning
        
        # Remove outliers if requested
        if remove_outliers:
            z_scores = np.abs(zscore(window))
            window = window[(z_scores < 3).all(axis=1)]
        
        # Winsorize data if requested
        if winsorize:
            window = window.apply(lambda col: mstats.winsorize(col, limits=winsorize_limits), axis=0)
        
        X, y = window.drop('target', axis=1), window['target']

        model = ConstrainedLinearRegression(ridge=l1_ratio, normalize=True, fit_intercept=fit_intercept)
        model.fit(X, y, min_coef=min_coef, max_coef=max_coef)

        end_idx = start + rolling_window

        coefs.iloc[end_idx] = model.coef_
        future_X = datac.iloc[end_idx:end_idx + n_step_ahead, :-1]
        future_preds = model.predict(future_X)
        predictions.iloc[end_idx:end_idx + n_step_ahead] = future_preds
    df_results = pd.concat([coefs.ffill(limit=30), predictions], axis=1).reindex(data.index).ffill(limit=10)
    return df_results, model


def feature_engineering(df, transformations):
    '''
    Extends the dataframe with engineered features based on the specified transformations.
    
    Parameters:
        df (pd.DataFrame): The original dataframe.
        transformations (dict): A dictionary with keys as column names and values as lists of transformations.
    
    Returns:
        pd.DataFrame: The dataframe with added engineered features.
    '''
    engineered_df = df.copy()
    
    for column, transformation_list in transformations.items():
        if column in engineered_df.columns:
            for transformation in transformation_list:
                operation, window = transformation  # Unpack operation and parameters
                if operation in ['mean', 'std', 'kurt', 'pct_change', 'diff']:
                    if operation in ['mean', 'std', 'kurt']:
                        transformed_series = getattr(df[column].rolling(window=window), operation)()
                    elif operation == 'pct_change':
                        transformed_series = df[column].pct_change(periods=window)
                    elif operation == 'diff':
                        transformed_series = df[column].diff(periods=window)
                    engineered_df[f"{column}_{operation}_{window}period"] = transformed_series
    return engineered_df

def lag_variables(df, lag_dict):
    dfc = df.copy()
    for col, windows in lag_dict.items():
        if col in dfc.columns:
            for window in windows:
                dfc[f'{col}_lag{window}'] = dfc[col].shift(window)

    return dfc


# def natural_lag_monthly(df):
#     # monthly_cols = [col for col in df.columns if (col.split('_')[1] == 'M') and (len(col.split('_')) > 1)]
#     monthly_cols = []
#     for col in df.columns:
#         if 'Monthly' in col:
#             monthly_cols.append(col)
#     monthly_cols_not_lagged = [col for col in monthly_cols if 'lag' not in col]
#     dfc = df.copy()
#     for col in monthly_cols_not_lagged:
#         dfc[f'{col}_natlag1'] = dfc[col].shift(1)

#     return dfc


def natural_lag(df):

    cols_not_lagged = [col for col in df.columns if 'lag' not in col]
    dfc = df.copy()
    for col in cols_not_lagged:
        dfc[f'{col}_natlag1'] = dfc[col].shift(1)

    return dfc


def generate_stats_table(regression_streamlit_state):
    y_pred = regression_streamlit_state['df_coefs_dict']['predictions'].dropna()
    y_true = regression_streamlit_state['df_transformed_lag'][regression_streamlit_state['selected_target']].loc[y_pred.index].dropna()
    y_pred = y_pred.loc[y_true.index]
    stats = [('r2', col, r2_score(y_true.loc[y_pred.index].values, y_pred[col])) for col in y_pred.columns]  + \
        [('mse', col, mean_squared_error(y_true.loc[y_pred.index].values, y_pred[col])) for col in y_pred.columns] + \
        [('mae', col, mean_absolute_error(y_true.loc[y_pred.index].values, y_pred[col])) for col in y_pred.columns] + \
        [('corr', col, np.corrcoef(y_true.loc[y_pred.index].values, y_pred[col].values)[0][1]) for col in y_pred.columns]
        
    df_stats = pd.DataFrame(stats, columns=['metrics', 'model', 'value'])
    return df_stats

def mean_absolute_directional_loss(y_true, y_pred):
    adl = np.sign(y_true * y_pred) * np.abs(y_true) * (-1)
    madl = np.mean(adl)
    return madl