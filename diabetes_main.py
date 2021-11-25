import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn import tree
from sklearn import metrics
import diabetes_predict
import diabetes_home
import diabetes_plots

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
                    page_icon = 'random',
                    layout = 'wide',
                    initial_sidebar_state = 'auto'
                    )

# Loading the dataset.
@st.cache()
def load_data():
    # Load the Diabetes dataset into DataFrame.

    df = pd.read_csv('https://s3-whjr-curriculum-uploads.whjr.online/b510b80d-2fd6-4c08-bfdf-2a24f733551d.csv')
    df.head()

    # Rename the column names in the DataFrame.
    df.rename(columns = {"BloodPressure": "Blood_Pressure",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree_Function",}, inplace = True)

    df.head() 

    return df

diabetes_df = load_data()

def grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age):    
    feature_columns = list(diabetes_df.columns)
    # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
    feature_columns.remove('Pregnancies')
    feature_columns.remove('Skin_Thickness')
    feature_columns.remove('Outcome')
    X = diabetes_df[feature_columns]
    y = diabetes_df['Outcome']
    # Split the train and test dataset. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    param_grid = {'criterion':['gini','entropy'], 'max_depth': np.arange(4,21), 'random_state': [42]}

    # Create a grid
    grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', n_jobs = -1)

    # Training
    grid_tree.fit(X_train, y_train)
    best_tree = grid_tree.best_estimator_
    
    # Predict diabetes using the 'predict()' function.
    prediction = best_tree.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(grid_tree.best_score_ * 100, 3)

    return prediction, score
    

pages_dict = {"Home": diabetes_home, 
              "Predict Diabetes": diabetes_predict, 
              "Visualise Decision Tree": diabetes_plots}
# Add radio buttons in the sidebar for navigation and call the respective pages based on user selection.
st.sidebar.title('Navigation')
user_choice = st.sidebar.radio('Go to', tuple(pages_dict.keys()))
if user_choice == 'Home' :
  diabetes_home.app(diabetes_df)
if user_choice == 'Predict Diabetes' :
  diabetes_predict.app(diabetes_df)
else :
  diabetes_plots.app(diabetes_df) 


def app(diabetes_df):
    st.markdown("<p style='color:red;font-size:25px'>This app uses <b>Decision Tree Classifier</b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
    st.subheader("Select Values:")
    
    # Create six sliders with the respective minimum and maximum values of features. 
    # store them in variables 'glucose', 'bp', 'insulin, 'bmi', 'pedigree' and 'age'
    # Write your code here:
    

    st.subheader("Model Selection")

    # Add a single select drop down menu with label 'Select the Classifier'
    predictor = st.selectbox("Select the Decision Tree Classifier",('Decision Tree Classifier', 'GridSearchCV Best Tree Classifier'))

    if predictor == 'Decision Tree Classifier':
        if st.button("Predict"):            
            prediction, score = d_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write("The accuracy score of this model is", score, "%")


    elif predictor == 'GridSearchCV Best Tree Classifier':
        if st.button("Predict"):
            prediction, score = grid_tree_pred(diabetes_df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Optimised Decision Tree Prediction results:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")

