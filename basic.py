#By Tushar Aggarwal

import numpy as np
from sklearn import datasets
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
st.title("Basic Datasets & Models by Tushar Aggarwal")
st.write("""
# Explore different Classifier
Which one is the best?
""")
dataset_names = st.sidebar.selectbox("Select a Dataset",("Iris","Wine"))

classifier_names =st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forests")) 

def get_dataset(dataset_names):
    if dataset_names =="Iris":
        data = datasets.load_iris()
    else:
        data =datasets.load_wine()
    X=data.data
    y=data.target

    return X,y 

X,y = get_dataset(dataset_names)
st.write("Shape of the dataset", X.shape)
st.write("Number of Classes", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name=="KNN":
        K=st.sidebar.slider("K", 1, 15)
        params["K"]=K
    elif clf_name=="SVM":
        C = st.sidebar.slider("C", 0.1, 10.0 )
        params["C"]=C
    else:
        max_depth = st.sidebar.slider("Max_depth",2, 15 )
        n_estimator = st.sidebar.slider("N_estimators", 1, 100)
        params["N_estimators"]=n_estimator
        params["Max_depth"]=max_depth
    return params

params = add_parameter_ui(classifier_names)


def get_classifier(clf_name, params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])    
    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(max_depth=params["Max_depth"], n_estimators=params["Max_depth"]
                                   ,random_state=123)
    return clf

clf = get_classifier(classifier_names, params)

#Classification
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc =accuracy_score(y_test, y_pred)

st.write(f"Classifier={classifier_names}")
st.write(f"Accuracy={acc}")

#Plot
pca = PCA(2)
X_projected=pca.fit_transform(X)

x_1=X_projected[:,0]
x_2=X_projected[:,1]

fig=plt.figure()

plt.scatter(x_1,x_2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)