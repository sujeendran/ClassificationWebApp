import pandas as pd
penguins = pd.read_csv('breast_cancer_cleaned.csv')

df = penguins.copy()
target = 'class'
encode = ['age','menopause','tumor_size','inv_nodes','node_caps','breast','breast_quad','irradiat']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'no-recurrence-events':0, 'recurrence-events':1}
def target_encode(val):
    return target_mapper[val]

df[target] = df[target].apply(target_encode)

# Separating X and Y
X = df.drop(target, axis=1)
Y = df[target]

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features=None)
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('bcancer_clf.pkl', 'wb'))