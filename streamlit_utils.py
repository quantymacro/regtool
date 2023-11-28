import pandas as pd
import numpy as np
import streamlit as st
from functools import partial
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objs as go
from copy import deepcopy
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from quantutils.regression_utils import generate_stats_table


def get_df_shap_and_error_contribution(regression_streamlit_state, df_features):
    df_coefs_dict = deepcopy(regression_streamlit_state['df_coefs_dict'])
    shap_list = []
    for x_var in regression_streamlit_state['selected_x_variables']:
        df_coef = df_coefs_dict[x_var].copy()
        df_shap = df_coef.mul(df_features[x_var], axis='index')
        shap_mean = df_shap.mean(axis=1)
        shap_mean.name = f'{x_var}_shap'
        shap_list.append(shap_mean)
    df_shap = pd.concat(shap_list, axis=1)
    target =  deepcopy(regression_streamlit_state['df_transformed_lag'][regression_streamlit_state['selected_target']])
    df_error_contribution = pd.DataFrame()
    for x_var in regression_streamlit_state['selected_x_variables']:
        df_shap_without_x_var = df_shap.drop(f'{x_var}_shap', axis=1)
        pred_without_x_var = df_shap_without_x_var.sum(axis=1)
        error_without_x_var = pred_without_x_var - target
        df_error_contribution[f'w/o_{x_var}_error'] = error_without_x_var

    return df_shap, df_error_contribution

def plotly_ranking(df_ranks, title):

    fig = go.Figure()
    # Add a line to the figure for each series
    for column in df_ranks.columns:
        fig.add_trace(go.Scatter(x=df_ranks.index, y=df_ranks[column], mode='lines+markers', name=column))

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Ranking',
        yaxis=dict(
            autorange='reversed',  # Reverse y-axis so that rank 1 appears at the top
            tickmode='array',
            tickvals=np.arange(1, len(df_ranks.columns) + 1),
            ticktext=[f"Rank {i}" for i in range(1, len(df_ranks.columns) + 1)]
        ),
        template='plotly_white'
    )

    return fig



def regression_analytics(regression_streamlit_state, df_regime_labelled=None):

    st.dataframe(generate_stats_table(regression_streamlit_state))
    df_x_variables_corr = regression_streamlit_state['df_transformed_lag'][regression_streamlit_state['selected_x_variables']].corr()
    fig = go.Figure()
    fig.add_trace(
            go.Heatmap(
                z=df_x_variables_corr.values,
                x=df_x_variables_corr.columns.tolist()[::-1],
                y=df_x_variables_corr.index.tolist(),
                colorscale='RdBu',
                showscale=True,
                zmid=0  # Center the color scale at zero
            ),
        )
    
    st.plotly_chart(fig)
    for x_var in regression_streamlit_state['selected_x_variables']:
        st.write(f"#### Coefficients_{x_var}")
        st.line_chart(regression_streamlit_state['df_coefs_dict'][x_var])

    st.write('#### Time Series of Predictions vs Target')
    df_pred_and_target = deepcopy(regression_streamlit_state['df_coefs_dict']['predictions'])
    df_pred_and_target = pd.concat([df_pred_and_target, regression_streamlit_state['df_transformed_lag'][st.session_state['selected_target']]], axis=1).ffill().dropna()
    st.line_chart(df_pred_and_target)

    fig = px.scatter()

        # Add scatter plots for each column against 'y'
    for col in regression_streamlit_state['df_coefs_dict']['predictions'].columns:  # Exclude the 'y' column
        fig.add_scatter(x=regression_streamlit_state['df_coefs_dict']['predictions'][col], y=regression_streamlit_state['df_transformed_lag'][regression_streamlit_state['selected_target']], mode='markers', name=f'{col}')
    fig.update_layout(
        title='Scatter Plot of Each Window Prediction against Target',
        xaxis_title='Prediction',
        yaxis_title='Target'
    )

    st.plotly_chart(fig)
    st.write('#### Time Series of Residuals')
    st.line_chart(regression_streamlit_state['df_coefs_dict']['residuals'])
    if df_regime_labelled is not None:
        df_residuals_by_regime = pd.concat([df_regime_labelled, regression_streamlit_state['df_coefs_dict']['residuals'].mean(axis=1)], axis=1)
        #Plot mean residuals by regime
        residuals_by_regime = [pd.concat([df_regime_labelled, regression_streamlit_state['df_coefs_dict']['residuals'].abs().mean(axis=1)], axis=1).select_dtypes(include='number').groupby(col).mean()[0] for col in df_regime_labelled.columns]
        residuals_by_regime = pd.concat(residuals_by_regime)
        residuals_by_regime.name = 'MeanResidualsbyRegime'
        residuals_by_regime = pd.DataFrame(residuals_by_regime)
        residuals_by_regime.index.name = 'Regime'

        fig = px.bar(residuals_by_regime)
        fig.update_layout(
            title='Residuals by Regime',
            xaxis_title='Regime',
            yaxis_title='Residuals',
            showlegend=False,
        )
        st.plotly_chart(fig)

        fig = px.bar(residuals_by_regime.sort_values('MeanResidualsbyRegime'))
        fig.update_layout(
            title='Residuals by Regime Sorted',
            xaxis_title='Regime',
            yaxis_title='Residuals',
            showlegend=False,
        )
        st.plotly_chart(fig)


    ##Relative Ranking of Model
    df_residual = deepcopy((regression_streamlit_state['df_coefs_dict']['residuals']).abs()**2).rolling(30).mean().dropna()
    df_ranks = df_residual.rank(axis=1, ascending=True)

    st.plotly_chart(plotly_ranking(df_ranks=df_ranks, title='Ranking of Models Over Time'))
    ##Rolling MSE of models
    st.write('#### Rolling MSE')
    st.line_chart(df_residual)


    ##Heatmap of Correlation Matrix
    residual_corr = deepcopy((regression_streamlit_state['df_coefs_dict']['residuals'])).corr()
    predictions_corr = deepcopy((regression_streamlit_state['df_coefs_dict']['predictions'])).corr()
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


    df_shap, df_error_contribution = get_df_shap_and_error_contribution(regression_streamlit_state, df_features=st.session_state['df_transformed_lag'])
    df_shap['pred'] = df_shap.sum(axis=1)
    # df_shap['pred'] = regression_streamlit_state['df_coefs_dict']['predictions'].mean(axis=1)
    st.write('#### Feature Contribution')
    st.line_chart(df_shap.dropna())
    st.write('#### Error Contribution')
    st.line_chart(df_error_contribution.abs().rolling(10).mean())

    st.plotly_chart(plotly_ranking(df_error_contribution.abs().rolling(10).mean().rank(axis=1, ascending=True), title='Ranking of Models Based on Error Contribution'))