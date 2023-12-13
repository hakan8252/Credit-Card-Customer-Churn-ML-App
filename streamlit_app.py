import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import plotly.express as px

# Load the LightGBM model
# Load the LightGBM model

# Load the original DataFrame (replace 'your_data.csv' with the actual file path)
original_df = pd.read_csv('train_df.csv')

X = original_df.drop('Attrition_Flag', axis=1)
y = original_df['Attrition_Flag']

original_df.drop('Attrition_Flag', axis=1, inplace=True)

# Default values (you can adjust these)
default_values = {
    'Customer_Age': 46,
    'Dependent_count': 2,
    'Months_on_book': 35,
    'Total_Relationship_Count': 3,
    'Months_Inactive_12_mon': 2,
    'Contacts_Count_12_mon': 2,
    'Credit_Limit': 8627,
    'Total_Revolving_Bal': 1162,
    'Avg_Open_To_Buy': 7465,
    'Total_Amt_Chng_Q4_Q1': 0.76,
    'Total_Trans_Amt': 4414,
    'Total_Trans_Ct': 65,
    'Total_Ct_Chng_Q4_Q1': 0.71,
    'Avg_Utilization_Ratio': 0.16,
    'Customer_Tenure': 1,
    'Gender': "M",
    'Education_Level': "High School",
    'Income_Category': "60𝐾−80K",
    'Card_Category': "Blue",
    'Marital_Status': "Married",

}

# Streamlit App
st.title('Customer Attrition Prediction App')

# Sidebar with buttons
selected_page = st.sidebar.radio('Select Page:', ['Prediction', 'Numeric Variables Visualization', 'Categorical Variables Visualization'])


if selected_page == 'Prediction':
    # Sidebar with default values
    st.sidebar.title('Input Parameters')

    # User inputs
    user_inputs = {}
    for feature, value in default_values.items():
        if feature in ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']:
            # Use default values if not present in the original DataFrame
            default_index = original_df[feature].to_list().index(value) if value in original_df[feature].unique() else 0
            user_inputs[feature] = st.sidebar.selectbox(f'Select {feature}', original_df[feature].unique(), index=default_index)
        else:
            if original_df[feature].dtype == 'float64':
                min_value = original_df[feature].min()
                max_value = original_df[feature].max()
                step = (max_value - min_value) / 100  # Adjust the step based on the range
                # Set the min and max values for numerical features
                user_inputs[feature] = st.sidebar.slider(f'Select {feature}', min_value=min_value, max_value=max_value, value=float(value),  step=step, key=f"{feature}_slider")
            else:
                min_value = int(original_df[feature].min())
                max_value = int(original_df[feature].max())
                # Set the min and max values for numerical features
                user_inputs[feature] = st.sidebar.number_input(f'Select {feature}', min_value=min_value, max_value=max_value, value=int(value))

    # Feature Engineering for dynamic computation of categorical variables
    user_inputs['Family_Size'] = user_inputs['Dependent_count'] + 1
    user_inputs['Customer_Tenure'] = user_inputs['Months_on_book'] // 12

    # Apply one-hot encoding to categorical variables
    categorical_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']

    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Assuming X and y are your feature matrix and target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Assuming you have 'X_train_full' and 'y_train_full' as your full training set
    lightgbm_model = lgb.LGBMClassifier()  # Replace 'CatBoostClassifier' with 'LGBMClassifier' if using LightGBM
    lightgbm_model.fit(X_train, y_train)

    # Display user inputs
    st.subheader('User Inputs:')
    st.write(user_inputs)


    # Add a "Predict" button
    if st.button('Predict'):
        # Convert user inputs to DataFrame
        user_df = pd.DataFrame([user_inputs])

        # Concatenate user inputs with the original DataFrame
        concatenated_df = pd.concat([user_df, original_df], ignore_index=True)

        # Ensure columns in user input align with the training data columns
        user_input_for_prediction = concatenated_df.reindex(columns=original_df.columns, fill_value=0)

        # Apply one-hot encoding to categorical variables
        user_input_for_prediction_encoded = pd.get_dummies(user_input_for_prediction, columns=categorical_columns, drop_first=True)

        # Make prediction and get probability estimates
        prediction_proba = lightgbm_model.predict_proba(user_input_for_prediction_encoded)

        # Display prediction and probabilities
        st.subheader('Prediction:')
        if prediction_proba[0, 1] > 0.5:
            st.error(f'Attrition (1) - Probability: {prediction_proba[0, 1]:.5f}')
        else:
            st.success(f'No Attrition (0) - Probability: {prediction_proba[0, 0]:.5f}')

    # Add buttons to show examples for 0 and 1
    # Add buttons to show examples for 0 and 1
    if st.button('Show Example for No Attrition (0)'):
        example_0 = X_train[y_train == 0].sample(1)
        st.subheader('Example for No Attrition (0):')
        st.write(example_0)

        # Concatenate user inputs with the original DataFrame
        concatenated_df = pd.concat([example_0, original_df], ignore_index=True)
        example_0_input_for_prediction = concatenated_df.reindex(columns=original_df.columns, fill_value=0)
        example_0_input_encoded = pd.get_dummies(example_0_input_for_prediction, columns=categorical_columns, drop_first=True)
        prediction_proba_0 = lightgbm_model.predict_proba(example_0_input_encoded)

        # Display prediction for example_0
        st.subheader('Prediction for No Attrition (0) Example:')
        if prediction_proba_0[0, 1] > 0.5:
            st.error(f'Attrition (1) - Probability: {prediction_proba_0[0, 1]:.5f}')
        else:
            st.success(f'No Attrition (0) - Probability: {prediction_proba_0[0, 0]:.5f}')

    if st.button('Show Example for Attrition (1)'):
        example_1 = X_train[y_train == 1].sample(1)
        st.subheader('Example for Attrition (1):')
        st.write(example_1)

        # Concatenate user inputs with the original DataFrame
        concatenated_df = pd.concat([example_1, original_df], ignore_index=True)
        example_1_input_for_prediction = concatenated_df.reindex(columns=original_df.columns, fill_value=0)
        example_1_input_encoded = pd.get_dummies(example_1_input_for_prediction, columns=categorical_columns, drop_first=True)
        prediction_proba_1 = lightgbm_model.predict_proba(example_1_input_encoded)

        # Display prediction for example_1
        st.subheader('Prediction for Attrition (1) Example:')
        if prediction_proba_1[0, 1] > 0.5:
            st.error(f'Attrition (1) - Probability: {prediction_proba_1[0, 1]:.5f}')
        else:
            st.success(f'No Attrition (0) - Probability: {prediction_proba_1[0, 0]:.5f}')

elif selected_page == 'Numeric Variables Visualization':
    # Page for Numeric Variables Visualization
    st.subheader('Numeric Variables Visualization Options')
    selected_numeric_variable = st.selectbox('Select Numeric Variable:', original_df.select_dtypes(include=['int64', 'float64']).columns)

    if st.button('Generate Histogram'):
        st.subheader(f'Histogram for {selected_numeric_variable}')

        # Create histogram using Plotly
        fig = px.histogram(original_df, x=selected_numeric_variable, title=f'Histogram for {selected_numeric_variable}')

        st.plotly_chart(fig)

elif selected_page == 'Categorical Variables Visualization':
    # Page for Categorical Variables Visualization
    st.subheader('Categorical Variables Visualization Options')
    selected_categorical_variable = st.selectbox('Select Categorical Variable:',
                                                 original_df.select_dtypes(include='object').columns)
    chart_type = st.selectbox('Select Chart Type:', ['Bar Chart', 'Pie Chart', 'Sunburst Chart'])


    st.subheader(f'{chart_type} for {selected_categorical_variable}')

    if chart_type == 'Bar Chart':
        # Create a DataFrame with counts for each category
        count_df = original_df[selected_categorical_variable].value_counts().reset_index()
        count_df.columns = [selected_categorical_variable, 'Count']

        # Create bar chart using Plotly with data labels
        fig = px.bar(count_df, x=selected_categorical_variable, y='Count',
                     title=f'Bar Chart for {selected_categorical_variable}', text='Count')

    elif chart_type == 'Pie Chart':
        # Create pie chart using Plotly
        fig = px.pie(original_df, names=selected_categorical_variable,
                     title=f'Pie Chart for {selected_categorical_variable}')



    elif chart_type == 'Sunburst Chart':

        # # Allow user to choose order for path
        # st.subheader(f'Sunburst Chart for {selected_categorical_variable}')

        # Get the unique categorical columns
        categorical_columns = original_df.select_dtypes(include='object').columns

        # Convert categorical_columns to a list
        categorical_columns_list = list(categorical_columns)

        # Exclude the selected categorical variable from the list
        path_order = st.multiselect('Select Order for Path (top to bottom):',
                                    [col for col in categorical_columns_list if col != selected_categorical_variable])

        # Create sunburst chart using Plotly without color parameter
        fig = px.sunburst(original_df, path=path_order + [selected_categorical_variable],
                          title=f'Sunburst Chart for {selected_categorical_variable} (Path Order: {path_order})')

        # Adjust layout for better visibility of data labels
        # fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)

    st.plotly_chart(fig)

# You can add more features, visualizations, or improvements based on your needs.
