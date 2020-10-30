import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Breast Cancer Recurrence Prediction App

This app predicts whether the breast cancer is a recurring or non-recurring one.

Note that the model was trained on a small dataset. So if you can't select the correct options, know that the dataset did not have them either.

Data obtained from the [UCI Machine learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer).
""")

st.sidebar.header('User Input Features')
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/sujeendran/ClassificationWebApp/master/BreastCancerRecurrence/breast_cancer_sample.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        age = st.sidebar.selectbox('Age',('20-29','30-39','40-49','50-59','60-69','70-79'))
        menopause = st.sidebar.selectbox('Menopause',('lt40','ge40','premeno'))
        tumor_size = st.sidebar.selectbox('Tumor-size',('0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54'))
        inv_nodes = st.sidebar.selectbox('Inv-Nodes',('0-2','3-5','6-8','9-11','12-14','15-17','24-26'))
        node_caps = st.sidebar.selectbox('Node-caps',('yes','no'))
        deg_malig = st.sidebar.slider('Degree of malignant tumor', 1,3,2)
        breast = st.sidebar.selectbox('Breast',('left','right'))
        breast_quad = st.sidebar.selectbox('Breast-quad',('left_up','left_low','right_up','right_low','central'))
        irradiat = st.sidebar.selectbox('Irradiated',('yes','no'))
        data = {'age': age,
                'menopause': menopause,
                'tumor_size': tumor_size,
                'inv_nodes': inv_nodes,
                'node_caps': node_caps,
                'deg_malig':deg_malig,
                'breast':breast,
                'breast_quad':breast_quad,
                'irradiat': irradiat}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire cancer dataset
# This will be useful for the encoding phase
dataset_raw = pd.read_csv('breast_cancer_cleaned.csv')
dataset = dataset_raw.drop(columns=['class'])
df = pd.concat([input_df,dataset],axis=0)

# Encoding of ordinal features
encode = ['age','menopause','tumor_size','inv_nodes','node_caps','breast','breast_quad','irradiat']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('bcancer_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediction')
cancer_types = np.array(['no-recurrence-events','recurrence-events'])
st.write(cancer_types[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)