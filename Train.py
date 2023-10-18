from sklearn.metrics import classification_report

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
# from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer('all-distilroberta-v1')
# model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
# model = SentenceTransformer('all-MiniLM-L12-v2')


# Sentence embeddings
def train():
    model = RandomForestClassifier()
    label_enc = LabelEncoder()
    # print(class_name)

    # Load the dataset
    # Get the parent directory path of the current script file
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Specify the path to your dataset file relative to the parent directory
    dataset_path_Train = os.path.join(parent_directory, 'Challenge-me/TrainOnMe.csv')
    dataset_path_Evaluate = os.path.join(parent_directory, 'Challenge-me/EvaluateOnMe.csv')
    print(dataset_path_Train)
    df = pd.read_csv(dataset_path_Train, sep=',')
    df_eval = pd.read_csv(dataset_path_Evaluate, sep=',')
    print(df)
    # print first column
    print(df.columns)
    print(df['y'].value_counts())
    # print values of first column
    print(df_eval['x7'].value_counts())
    df['y'] = df['y'].replace('Bobborg', 'Boborg')
    df['y'] = df['y'].replace('Atsutoob', 'Atsutobob')
    df['y'] = df['y'].replace('Jorggsuto', 'Jorgsuto')
    print(df.columns)
    print(df.shape[0])
    # print first 5 rows
    print(df.head())
    # check rows for more than one null value
    # print(df[df.isnull().sum(axis=1) > 1])
    # delete these rows from df and check again
    df = df[df.isnull().sum(axis=1) <= 1]
    print("second check")
    # print(df[df.isnull().sum(axis=1) > 1])

    # check columns for null values
    print(df_eval.isnull().sum(axis=0))
    print("babe")
    # replace with empty string
    # df = df.fillna('')

    # drop rows with null values
    df = df.dropna()
    print(df.isnull().sum(axis=0))
    # Find unique values in column x12
    print(df['x12'].value_counts())
    # drop column x12
    df = df.drop('x12', axis=1)
    df_eval = df_eval.drop('x12', axis=1)

    #################FAULTY SPELLING################################
    print(df['x7'].value_counts())
    # replace faulty spelling with correct spelling
    # Polkagris instead of Polkagriss
    df['x7'] = df['x7'].replace('Polkagriss', 'Polkagris')
    df['x7'] = df['x7'].replace('Schottisgriss', 'Schottisgris')
    print(df['x7'].value_counts())
    # Google search for correct spelling
    df['x7'] = label_enc.fit_transform(df['x7'])
    df_eval['x7'] = label_enc.fit_transform(df_eval['x7'])
    # print first 5 values of x7
    print(df['x7'].head())
    #################FAULTY SPELLING################################

    # divide into training and test set
    # x is all columns x1 to x12
    # y is column y
    Y = df['y']
    X = df.drop('y', axis=1).copy()
    X = X.drop(X.columns[0], axis=1)
    df_eval = df_eval.drop(df_eval.columns[0], axis=1)

    # print(df['y'].value_counts())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

    # (df_eval.shape)
    # print(df_eval.columns)
    # X_train = X
    # Y_train = Y
    # X_test = df_eval
    model.fit(X_train, Y_train)
   #  model.score(X_test, Y_test)
    accuracy = model.score(X_test, Y_test)
    print("\n")
    print(f"your score is {accuracy*100} % Babeh")
    # i want the false positives and false negatives
    y_pred = model.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    num_samples = len(Y_test)
    num_classes = len(cm)  # Number of classes

    class_names = ["Atsutobob","Boborg", "Jorgsuto"]
    for class_idx in range(num_classes):
        tp = cm[class_idx, class_idx]
        fp = sum(cm[i, class_idx] for i in range(num_classes) if i != class_idx)
        fn = sum(cm[class_idx, i] for i in range(num_classes) if i != class_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"Class {class_names[class_idx]}:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}\n")

    report = classification_report(Y_test, y_pred, target_names=["Atsutobob", "Boborg", "Jorgsuto"])
    print(report)


train()