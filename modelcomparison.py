import pandas as pd
import numpy as np
import streamlit as st
import quantutils.regression_utils as regutils
import quantutils.general_utils as genutils
from quantutils.general_utils import get_df_regime_label
import quantutils.streamlit_utils as stutils
from functools import partial
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objs as go
from copy import deepcopy
from plotly.subplots import make_subplots
from pathlib import Path
import os

### This script takes fitted models and results, and then make comparison
'''
### Things to show
- prediction over time
- scatter plot
- weighting scheme (hard to implement)
- correlation between models (both error and residuals)
- ranking of models

'''


def model_comparison(models_list, target_variable_index_name, prediction_agg_function='mean', shift=1):
    target_index = deepcopy(models_list[0]['df_transformed_lag'][target_variable_index_name])
    df_preds_mega = []
    df_residuals_mega = []
    sharpes = []
    df_stats_mega = []
    df_x_variables = []
    df_pnls_mega = []
    for model in models_list:
        df_predictions = deepcopy(model['df_coefs_dict']['predictions']).dropna()
        df_residuals = deepcopy(model['df_coefs_dict']['residuals'].dropna())
        df_stats = regutils.generate_stats_table(model).set_index('metrics')
        stats_agg = df_stats.select_dtypes('number').groupby('metrics').mean()
        pred_agg = df_predictions.mean(axis=1)
        residuals_agg = df_residuals.mean(axis=1)
        pred_agg.name = model['name']
        residuals_agg.name = model['name']
        pnl = pred_agg.shift(shift) * target_index.pct_change()
        pnl.name = model['name']
        df_pnls_mega.append(pnl.cumsum())
        sharpe = (pnl.mean()/pnl.std())
        stats_agg.loc[len(stats_agg)] = (sharpe)
        stats_agg.index = list(stats_agg.index[:-1]) + ['sharpe']
        stats_agg.name = model['name']
        df_preds_mega.append(pred_agg)
        df_residuals_mega.append(residuals_agg)
        df_stats_mega.append(stats_agg)
        sharpes.append(sharpe)
        df_x_variables.append(pd.Series(model['selected_x_variables']))
    
    df_preds_mega = pd.concat(df_preds_mega, axis=1)
    df_pnls_mega = pd.concat(df_pnls_mega, axis=1)
    df_residuals_mega = pd.concat(df_residuals_mega, axis=1)
    df_stats_mega = pd.concat(df_stats_mega, axis=1)
    df_stats_mega.columns = [model['name'] for model in models_list]
    df_x_variables = pd.concat(df_x_variables, axis=1)
    df_x_variables.columns = [model['name'] for model in models_list]

    df_adl = np.sign(df_preds_mega.mul(model['df_transformed_lag'][model['selected_target']], axis='index'))
    df_adl = df_adl.mul(model['df_transformed_lag'][model['selected_target']].abs(), axis='index') * (-1)

    df_preds_mega['target'] = model['df_transformed_lag'][model['selected_target']]


    return df_preds_mega.dropna(), df_residuals_mega.dropna(), df_stats_mega, df_x_variables, df_pnls_mega, df_adl

with st.form(key='Select Target Variable'):
    all_targets = os.listdir('../RegressionTools/Models')
    selected_target = st.selectbox('Target Variable', all_targets)
    target_variable_index_name = selected_target.split('_')[0]
    st.session_state['selected_target'] = selected_target
    st.session_state['target_variable_index_name'] = target_variable_index_name
    submit_button_selected_target = st.form_submit_button(label='Confirm Target Variable')

if submit_button_selected_target:
    models_list = []
    directory = f'../RegressionTools/Models/{st.session_state["selected_target"]}/'
    model_files = os.listdir(directory)
    for file_name in model_files:
        model = genutils.load_object(file_name, directory)
        models_list.append(model)

    st.session_state['models_list'] = models_list

if 'models_list' in st.session_state:
    df_preds_mega, df_residuals_mega, df_stats_mega, df_x_variables, df_pnls_mega, df_adl = model_comparison(st.session_state['models_list'], st.session_state['target_variable_index_name'])
    st.write('#### Selected X Variables')
    st.write(df_x_variables)
    st.write('#### Stats Comparison')
    st.dataframe(df_stats_mega)
    st.write('#### Naive PnL')
    st.line_chart(df_pnls_mega.dropna())

    st.write('#### Time Series of Predictions vs Target')
    st.line_chart(df_preds_mega)

    residual_corr = deepcopy((df_residuals_mega)).corr()
    predictions_corr = deepcopy((df_preds_mega.iloc[:, :-1])).corr()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Prediction Correlation', 'Residuals Correlation'), horizontal_spacing=0.2)
    fig.add_trace(
        go.Heatmap(
            z=residual_corr.values,
            x=residual_corr.columns.tolist()[::-1],
            y=residual_corr.index.tolist(),
            colorscale='RdBu',
            showscale=True,
            zmid=0  # Center the color scale at zero
        ),
        row=1, col=2
    )

    # Add the second heatmap to the second column
    fig.add_trace(
        go.Heatmap(
            z=predictions_corr.values,
            x=predictions_corr.columns.tolist()[::-1],
            y=predictions_corr.index.tolist(),
            colorscale='RdBu',
            showscale=True,
            zmid=0  # Center the color scale at zero
        ),
        row=1, col=1
    )

    # Update the layout
    fig.update_layout(
        title_text='Correlation of Prediction and Residuals',
        showlegend=False,
    )

    st.plotly_chart(fig)

    df_ranks = df_residuals_mega.abs().rolling(12).mean().rank(axis=1, ascending=True)
    fig = stutils.plotly_ranking(df_ranks, 'Model Ranking Over Time')
    st.plotly_chart(fig)

    df_residual = deepcopy((df_residuals_mega).abs()**2).rolling(30).mean().dropna()

    ##Rolling MSE of models
    st.write('#### Rolling MSE')
    st.line_chart(df_residual)

    st.line_chart(df_adl.rolling(12).mean())
