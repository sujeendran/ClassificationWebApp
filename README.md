# Multi-Variate Data Classifiers using Python and Streamlit

This repo contains some simple projects I ran to see how the Streamlit module can be used along with sklearn to train a model and provide a user-friendly interface for testing.

The Penguin classifier is basically a fork of [Chanin Nantasenamat's example](https://towardsdatascience.com/how-to-build-a-data-science-web-app-in-python-penguin-classifier-2f101ac389f3).

The Breast Cancer Recurrence classifier is a modification of the same classifier to try a different [dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer).

## Demo

[Live link for Breast Cancer Recurrence classifier](https://share.streamlit.io/sujeendran/classificationwebapp/BreastCancerRecurrence/breast_cancer_webapp.py)

To run a demo install the python modules:

`pip install streamlit pandas numpy scikit-learn`

Now enter one of the directories and run any of the following:

`streamlit run breast_cancer_webapp.py`

OR

`streamlit run pwebapp.py`
