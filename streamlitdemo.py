import pandas as pd
import numpy as np
import streamlit as st
import quantutils.regression_utils as qu
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
import datetime

if 'transformations' not in st.session_state:
    st.session_state['transformations'] = {}

if 'lag' not in st.session_state:
    st.session_state['lag'] = {}

# Function to add a transformation to a specific variable
def add_transformation(variable, transformation_type, window):
    if variable not in st.session_state['transformations']:
        st.session_state['transformations'][variable] = []
    
    st.session_state['transformations'][variable].append([transformation_type, window])

def add_lag(variable, lag):
    if variable not in st.session_state['lag']:
        st.session_state['lag'][variable] = []
    st.session_state['lag'][variable].append(lag)

# def collect_transformation(variable, transformation_type, lag, window):
def remove_transformation(variable, transformation_type, window):
    try:
        # Find and remove the transformation from the list
        transformations_list = st.session_state['transformations'].get(variable, [])
        transformation_to_remove = [transformation_type, window] 
        if transformation_to_remove in transformations_list:
            transformations_list.remove(transformation_to_remove)
            st.session_state['transformations'][variable] = transformations_list
            st.success(f"Removed transformation {transformation_to_remove} from {variable}")
        else:
            st.error(f"Transformation {transformation_to_remove} not found in {variable}")
    except ValueError as e:
        st.error(f"Error removing transformation: {e}")

st.write("""
## Regression Tool
         
         """)


np.random.normal(100)
SIZE=300
# df = pd.DataFrame(np.random.normal(size=SIZE))
# df.columns = ['X1']
# df['X2'] = np.random.normal(size=SIZE)
# df['X3'] = np.random.normal(size=SIZE)
# df = df.rolling(10).mean()
# df['target'] = df['X1'] + 0.5*df['X2'].shift(1) + 1.2*df['X2'].shift(2) + np.random.normal(scale=0.2,size=SIZE)
# df = df.dropna()
# df.index = pd.date_range(start='2010-01-01', periods=len(df))

# df_daily = df.copy()
# df_monthly = df.copy() + np.random.randint(1, size=df.shape)
# df_monthly.columns = ['A1', 'A2', 'A3', 'A4']

df = pd.read_excel('../Data/sample_data.xlsx', skiprows=1)
df.columns = ['Date', 'CPIUSA', 'CPIUNSA', 'ManheimUsedVehicle', 'PPIUsedVehicles', 'PPIAutomotive', 'PPIPassenger_Cars', '5yTreasury']
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df.columns = [f'{col}Monthly' for col in df.columns]




# N_MACRO = 3
# MACRO_COLUMNS = ['Fed', 'GDP', 'VIX']
# df_regime = pd.DataFrame(np.random.normal(size=(len(df), N_MACRO)), index=df.index)
# df_regime.columns = MACRO_COLUMNS[:N_MACRO]



# def helper_regime(df_regime, q_list):
#     df_regimec = df_regime.copy()
#     for i, col in enumerate(df_regime):
#         q = q_list[i]
#         df_regimec[col] = np.ceil(df_regimec[col].rank(pct=True)*q)
#     return df_regimec

# label_dict = {'Fed': {1.0 : 'Hiking', 2.0: 'Cutting'},
#               'GDP': {1.0: 'Accelerating', 2.0: 'Slowing', 3.0: 'Falling'},
#               'VIX': {1.0: 'High', 2.0: 'Medium', 3.0: 'Low'},
#               }

# df_regime_discrete = helper_regime(df_regime, [2, 3, 3])        
# df_regime_labelled = get_df_regime_label(df_regime_discrete, label_dict)


feature_list = list(df.columns)
feature_list = list(set(feature_list))
st.session_state['expanding'] = False
#--------------------------------------------------------------------------------------
variables = st.multiselect('Relevant Variables', feature_list)
# st.session_state['target'] = y_variable
transformations_dict = {}
st.write("### Create variables Transformation")

with st.form(key='transformation_form'):
    # Use columns to create a 2x2 grid layout
    col1, col2 = st.columns(2)
    with col1:
        variable = st.selectbox('Variable',  variables)
        transformation_type = st.selectbox('Select transformation type', ['mean', 'std', 'diff', 'pct_change'])
    with col2:
        window = st.number_input('Window size', min_value=1, value=3)
    # Submit button for the form
    submit_button = st.form_submit_button(label='Confirm Transformation')


if submit_button:
    add_transformation(variable, transformation_type, window)
    # Display the updated transformations

# Function to remove a transformation from a specific variable

# Section for removing transformations
st.write("### Remove a Transformation")
with st.form(key='remove_transformation_form'):
    col1, col2, col3= st.columns(3)
    with col1:
        variable_to_remove = st.text_input('Variable to remove from', key='remove_variable')
    with col2:
        transformation_type_to_remove = st.selectbox('Transformation type to remove', ['mean', 'std', 'diff', 'pct_change'], key='remove_trans_type')
    with col3:
        window_to_remove = 1  # Default for 'lag'
        window_to_remove = st.number_input('Window size to remove', min_value=1, value=3, key='remove_window')

    remove_button = st.form_submit_button(label='Remove Transformation')

if remove_button:
    remove_transformation(variable_to_remove, transformation_type_to_remove, window_to_remove)

st.write(f"{st.session_state['transformations']}")

if st.button('Create Transformations'):
    unique_cols = list(set(variables))
    unique_transformed_cols = list(df.columns)
    df_transformed = qu.feature_engineering(df[unique_cols], st.session_state['transformations'])
    st.session_state['df_transformed'] = df_transformed
    st.session_state['unique_transformed_cols'] = unique_transformed_cols
    st.write(unique_transformed_cols)

if 'unique_transformed_cols' in st.session_state:
    st.write('### Create Lagged Variables')
    with st.form('Create Lagged Variables'):
        variable_to_lag = st.selectbox('X-variable', [col for col in st.session_state['unique_transformed_cols']])
        lag_window = st.number_input('Lag window', min_value=0, value=1)
        submit_button_lag = st.form_submit_button(label='Confirm Lag')
    if submit_button_lag:
        add_lag(variable_to_lag, lag_window)

st.write(f'{st.session_state["lag"]}')

if st.button('Implement Lag'):

    ## Daily lag is straightforward
    ## Monthly lag is not
    ## What about natural lag?
    # df_daily_transformed_lag = qu.lag_variables(st.session_state['df_daily_transformed'], st.session_state['lag'])
    # df_monthly_transformed_lag = qu.lag_variables(st.session_state['df_monthly_transformed'], st.session_state['lag'])
    # st.session_state['df_daily_transformed_lag'] = df_daily_transformed_lag
    # st.session_state['df_monthly_transformed_lag'] = df_monthly_transformed_lag
    # df_transformed_lag = pd.concat([df_daily_transformed_lag, df_monthly_transformed_lag], axis=1).ffill()
    df_transformed= st.session_state['df_transformed']

    df_transformed_lag = qu.natural_lag(df_transformed)
    st.session_state['df_transformed_lag'] = df_transformed_lag
    st.write(st.session_state['df_transformed_lag'].tail())

with st.form('X-variables in regression'):
    st.write('### Confirm X-Variables')
    if 'df_transformed_lag' in st.session_state:
        selected_target = st.selectbox('Y-variable', st.session_state['df_transformed_lag'].columns)

        selected_x_variables = st.multiselect('X-variables', st.session_state['df_transformed_lag'].columns)
        st.session_state['selected_x_variables'] = selected_x_variables
        st.session_state['selected_target'] = selected_target
        ## If all selected_x_variables is monthly, change df_transformed_lag to df_monthly_transformed_lag
        

        submit_x_variable_button = st.form_submit_button(label='Confirm X-Variables in Regression')
        # ## If all selected_x_variables is monthly, change df_transformed_lag to df_monthly_transformed_lag
        # check_if_all_monthly = [col for col in df_daily.columns if col in selected_x_variables]

        # if check_if_all_monthly:
        #     st.session_state['df_transformed_lag'] = deepcopy(st.session_state['df_monthly_transformed_lag'])



# Check if selected_x_variables are available
if 'selected_x_variables' in st.session_state:
    # Initialize dictionaries if not already present
    # if 'min_coefs_dict' not in st.session_state:
    st.session_state['min_coefs_dict'] = {xvar: -np.inf for xvar in st.session_state['selected_x_variables']}
    # if 'max_coefs_dict' not in st.session_state:
    st.session_state['max_coefs_dict'] = {xvar: np.inf for xvar in st.session_state['selected_x_variables']}
    col1, col2 = st.columns(2)
    col1.header("Minimum Coefficients")
    col2.header("Maximum Coefficients")
    # Create a Nx2 panel for min and max coefficient constraints
    for xvar in st.session_state['selected_x_variables']:
        with col1:
            st.session_state['min_coefs_dict'][xvar] = st.number_input(f"Min coef for {xvar}", value=float(-999999), format="%f", key=f"min_{xvar}")
        with col2:
            st.session_state['max_coefs_dict'][xvar] = st.number_input(f"Max coef for {xvar}", value=float(9999999), format="%f", key=f"max_{xvar}")
    
    # Button to confirm the constraints
    if st.button('Confirm Coefficient Constraints'):
        st.write("Min Coefficients Constraints:", st.session_state['min_coefs_dict'])
        st.write("Max Coefficients Constraints:", st.session_state['max_coefs_dict'])


with st.form('Rolling Windows'):
    numbers_str = st.text_input('Enter a list of numbers, separated by commas')
    # Initialize an empty list to hold integers
    numbers_list = []
    # Convert the string to a list of integers
    if numbers_str:
        try:
            # Split the string into a list where each number is a string
            numbers_str_list = numbers_str.split(',')
            # Convert each string number to an integer and add to the list
            numbers_list = [int(number.strip()) for number in numbers_str_list]
            st.success('List of windows: ' + str(numbers_list))
        except ValueError:
            # If conversion fails, inform the user
            st.error('Please enter a valid list of numbers, separated by commas')
    submit_windows_button = st.form_submit_button(label='Confirm selected windows')
    st.session_state['windows'] = numbers_list

with st.form('Refit Model Frequency'):
    n_step_ahead = st.number_input('Refitting Model Frequency (1 means daily fit)', value=1)
    submit_refit_freq = st.form_submit_button(label='Confirm Re-fit Model Freq')
    expanding = st.checkbox('Expanding Window')
    if expanding:
        st.session_state['expanding'] = True
        print('Expanding Window Mode Enabled')
    st.session_state['n_step_ahead'] = n_step_ahead

with st.form('Start Date'):
    start_date = st.date_input('Start Date', value=datetime.date(2000, 1, 1), format='YYYY/MM/DD', max_value=datetime.date.today())
    submit_refit_freq = st.form_submit_button(label='Confirm Start Date')
    st.session_state['start_date'] = pd.Timestamp(start_date)
    st.session_state['df_transformed_lag'] = st.session_state['df_transformed_lag'][start_date:]

if st.button('Run Regression'):

    if 'df_transformed_lag' in st.session_state and 'selected_x_variables' and 'n_step_ahead' and 'start_date' in st.session_state:
        
        df_regression = st.session_state['df_transformed_lag'][st.session_state['selected_x_variables']].copy()
        df_regression['target'] = st.session_state['df_transformed_lag'][st.session_state['selected_target']]
        # df_regression = df_regression[st.session_state['start_date']:].copy()
        df_coefs_dict = {k: [] for k in st.session_state['selected_x_variables']}
        df_coefs_dict['predictions'] = []
        min_coef = []
        max_coef = []
        for xvar in df_regression.drop('target', axis=1).columns:
            min_coef.append(st.session_state['min_coefs_dict'][xvar])
            max_coef.append(st.session_state['max_coefs_dict'][xvar])

        
        for window in tqdm(st.session_state['windows']):
            df_results, model = qu.rolling_regression_sklearn_advanced(df_regression, rolling_window=window, n_step_ahead=st.session_state['n_step_ahead'],
                                                                        dropna=True, min_coef=min_coef, max_coef=max_coef, expanding=st.session_state['expanding'])
            for x_var in st.session_state['selected_x_variables']:
                df_coefs_dict[x_var].append(df_results[[x_var]].rename(columns={f'{x_var}': f'{window}'}).squeeze())
            df_coefs_dict['predictions'].append(df_results[['predictions']].rename(columns={f'predictions': f'predictions_{window}'}).squeeze())
        for x_var in st.session_state['selected_x_variables']:
            df_coefs_dict[x_var] = pd.concat(df_coefs_dict[x_var], axis=1).dropna()

        df_coefs_dict['predictions'] = pd.concat(df_coefs_dict['predictions'], axis=1).dropna()
        df_coefs_dict['residuals'] = df_coefs_dict['predictions'].sub(df_regression['target'], axis='index').dropna()
        df_coefs_dict['residuals'].columns = [f'residuals_{str(col).split("_")[-1]}' for col in df_coefs_dict['residuals'].columns]
        st.session_state['df_coefs_dict'] = df_coefs_dict
        regression_streamlit_state = deepcopy(dict(st.session_state))
        stutils.regression_analytics(regression_streamlit_state, df_regime_labelled=None)


    else:
        st.write('Cannot run regression')


with st.form(key='save_model_form'):
    name = st.text_input('Model Name')
    st.session_state['name'] = name
    save_model_submit = st.form_submit_button('Save Model')

if save_model_submit:
    directory = f"../RegressionTools/Models/{st.session_state['selected_target']}/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if name:  # Check if the name is not empty
        regression_streamlit_state = deepcopy(dict(st.session_state))
        genutils.save_object(regression_streamlit_state, f'{name}.pkl', directory)
        st.success(f'Model saved as {name}.pkl')
    else:
        st.error('Please enter a valid model name.')

