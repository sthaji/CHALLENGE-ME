# Import necessary libraries
import os

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

def train():
    # import lasso regression
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = LogisticRegression(penalty='l2', solver='liblinear)
    model = RandomForestClassifier()
    #model1 = XGBClassifier()
    #model2 = GradientBoostingClassifier()
    # model = SVC(kernel="poly", degree=2, coef0=1, C=5)
    label_enc = LabelEncoder()
    # print(class_name)

    # Load the dataset
    # Get the parent directory path of the current script file
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path_Train = os.path.join(parent_directory, 'Challenge-me/TrainOnMe.csv')
    dataset_path_Evaluate = os.path.join(parent_directory, 'Challenge-me/EvaluateOnMe.csv')
    df = pd.read_csv(dataset_path_Train, sep=',')
    df_eval = pd.read_csv(dataset_path_Evaluate, sep=',')
    print(df)
    ################ REPLACE MISSTYPES #############################
    print(df.columns)
    print(df['y'].value_counts())
    df['y'] = df['y'].replace('Bobborg', 'Boborg')
    df['y'] = df['y'].replace('Atsutoob', 'Atsutobob')
    df['y'] = df['y'].replace('Jorggsuto', 'Jorgsuto')
    print(df['x7'].value_counts())
    # replace faulty spelling with correct spelling
    df['x7'] = df['x7'].replace('Polkagriss', 'Polkagris')
    df['x7'] = df['x7'].replace('Schottisgriss', 'Schottisgris')
    print(df['x7'].value_counts())
    # Google search for correct spelling
    df['x7'] = label_enc.fit_transform(df['x7'])
    # df_eval['x7'] = label_enc.fit_transform(df_eval['x7'])
    # print first 5 values of x7
    ################ REPLACE MISSTYPES #############################



    ####################### CHECK FOR NULL #########################
    print(df[df.isnull().sum(axis=1) > 1]) # check rows for more than one null value
    df = df[df.isnull().sum(axis=1) <= 1]
    print("second check")
    df = df.dropna()
    print(df.isnull().sum(axis=0))
    ####################### CHECK FOR NULL #############################


    #######################turn into float##############################
    # Assuming 'df' is your DataFrame
    for column in df.columns:
        print(f"Column '{column}' has data type: {df[column].dtype}")
    df['x1'] = df['x1'].astype('float64')
    #######################turn into float##############################

    # print max min values of x1
    print(df['x1'].max())
    print(df['x1'].min())
    print(df['x1'].mean())
    # delete row wheere x1 is max value
    df = df[df.x1 != df.x1.max()]
    print(df['x1'].mean())


    ###################### DROP X12 ###################################
    # Find unique values in column x12
    print(df['x12'].value_counts())
    df = df.drop('x12', axis=1)
    # df_eval = df_eval.drop('x12', axis=1)
    ###################### DROP X12 ###################################


    ############## divide into training and test set################
    df['y'] = label_enc.fit_transform(df['y'])
    #print(df['y'].value_counts()) IF YOU WANT TO USE NUMERIC VALUES OF Y
    Y = df['y']
    X = df.drop('y', axis=1).copy()
    X = X.drop(X.columns[0], axis=1)
    # df_eval = df_eval.drop(df_eval.columns[0], axis=1)
    # print(df['y'].value_counts())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y) # add stratify = Y

    # Step 1: PCA Dimensionality Reduction
    n_pca_components = 12  # Choose the number of PCA components
    pca = PCA(n_components=n_pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Step 2: Subspace Method (Linear Discriminant Analysis - LDA)
    n_lda_components = 2  # Choose the number of LDA components
    lda = LinearDiscriminantAnalysis(n_components=n_lda_components)
    X_train_lda = lda.fit_transform(X_train_pca, Y_train)
    X_test_lda = lda.transform(X_test_pca)

    # Step 3: Training a Classifier (Random Forest in this example)
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_lda, Y_train)
    classifier2 = GradientBoostingClassifier()
    classifier2.fit(X_train_lda, Y_train)

    # Step 5: Model Evaluation
    Y_pred = classifier.predict(X_test_lda)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy random:", accuracy)

    Y_pred  = classifier2.predict(X_test_lda)
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy gradient:", accuracy)


train()