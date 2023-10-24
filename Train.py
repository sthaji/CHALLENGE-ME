from sklearn.metrics import classification_report, accuracy_score

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
from xgboost import XGBClassifier  # Import XGBoost classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


# model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-mpnet-base-v2')
# model = SentenceTransformer('all-distilroberta-v1')
# model = SentenceTransformer('average_word_embeddings_glove.6B.300d')
# model = SentenceTransformer('all-MiniLM-L12-v2')

# Sentence embeddings
########################## K CROSS ##########################################
# Perform k-fold cross-validation and get accuracy scores for each fold
def cross_val(model, X, Y, cv=5):
    accuracy_scores = cross_val_score(model, X, Y, cv=None, scoring='accuracy')

    # Print the accuracy scores for each fold
    for fold, accuracy in enumerate(accuracy_scores):
        print(f"Fold {fold + 1}: Accuracy = {accuracy * 100:.2f}%")

    # Calculate and print the mean and standard deviation of accuracy scores
    mean_accuracy = accuracy_scores.mean()
    std_accuracy = accuracy_scores.std()
    print(f"Mean Accuracy = {mean_accuracy * 100:.2f}%")
    print(f"Standard Deviation of Accuracy = {std_accuracy * 100:.2f}%")
########################## K CROSS ##########################################



def clean_data(df):
    label_enc = LabelEncoder()
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
    print(df[df.isnull().sum(axis=1) > 1])  # check rows for more than one null value
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

    """ CHECK FOR OUTLIERS
    print("x2")
    print(df['x2'].max())
    print(df['x2'].min())
    print(df['x2'].mean())

    print("x3")
    print(df['x3'].max())
    print(df['x3'].min())
    print(df['x3'].mean())

    print("x4")
    print(df['x4'].max())
    print(df['x4'].min())
    print(df['x4'].mean())

    print("x5")
    print(df['x5'].max())
    print(df['x5'].min())
    print(df['x5'].mean())

    print("x6")
    print(df['x6'].max())
    print(df['x6'].min())
    print(df['x6'].mean())

    print("x7")
    print(df['x7'].max())
    print(df['x7'].min())
    print(df['x7'].mean())

    print("x8")
    print(df['x8'].max())
    print(df['x8'].min())
    print(df['x8'].mean())

    print("x9")
    print(df['x9'].max())
    print(df['x9'].min())
    print(df['x9'].mean())


    print("x10")
    print(df['x10'].max())
    print(df['x10'].min())
    print(df['x10'].mean())

    print("x11")
    print(df['x11'].max())
    print(df['x11'].min())
    print(df['x11'].mean())

    print("x13")
    print(df['x13'].max())
    print(df['x13'].min())
    print(df['x13'].mean())
    """

    ###################### DROP X12 ###################################
    # Find unique values in column x12
    print(df['x12'].value_counts())
    # drop column named Unnamed: 0

    print(df.columns)
    df = df.drop('x12', axis=1)
    # df_eval = df_eval.drop('x12', axis=1)
    ###################### DROP X12 ###################################
    df = df.drop(df.columns[0], axis=1)
    df['y'] = label_enc.fit_transform(df['y'])
    print(df.head())
    return df

def train():
    # import lasso regression
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = LogisticRegression(penalty='l2', solver='liblinear)
    best_params_d = {
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 100
    }


    best_params = {
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_leaf': 4,
        'min_samples_split': 5,
        'n_estimators': 100
    }

    best_params2 = {
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 2,
        'min_samples_split': 5,
        'n_estimators': 200
    }
    best_paramsXboost = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.2,
    'max_depth': 6,
    'n_estimators': 100,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'subsample': 0.9
    }

    best_paramsXboost2 = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 200,
    'reg_alpha': 0.5,
    'reg_lambda': 0,
    'subsample': 0.9
    }


    model_randfore = RandomForestClassifier(random_state=42)
    model_randforeGrid = RandomForestClassifier(**best_params_d)
    model_XGBC_Normal = XGBClassifier()
    model_XGBC = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0,
        reg_lambda=0
    )
    model_XGBC2 = XGBClassifier(**best_paramsXboost)
    model_XGBC3 = XGBClassifier(**best_paramsXboost2) ## for removed of x3 and x13


    model_gradientBoost1 = GradientBoostingClassifier(**best_params)
    model_gradientBoost2 = GradientBoostingClassifier(**best_params2)

    models = [model_randfore,model_randforeGrid, model_XGBC_Normal, model_XGBC, model_XGBC2, model_XGBC3, model_gradientBoost1, model_gradientBoost2]

    # model = SVC(kernel="poly", degree=2, coef0=1, C=5)
    # print(class_name)

    # Load the dataset
    # Get the parent directory path of the current script file
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path_Train = os.path.join(parent_directory, 'Challenge-me/TrainOnMe.csv')
    dataset_path_Evaluate = os.path.join(parent_directory, 'Challenge-me/EvaluateOnMe.csv')
    df = pd.read_csv(dataset_path_Train, sep=',')
    df_eval = pd.read_csv(dataset_path_Evaluate, sep=',')


    df = clean_data(df)



    ############## divide into training and test set################
    #
    print(df['y'].value_counts())
    Y = df['y']
    X = df.drop('y', axis=1)
    #X = X.drop('x3', axis=1)
    # X = X.drop('x13', axis=1)
    #scaler = MinMaxScaler()
    print(X.head())
    #X = scaler.fit_transform(X.values)
    n_pca_components = 12  # Choose the number of PCA components
    pca = PCA(n_components=n_pca_components)
    X = pca.fit_transform(X)
    # df_eval = df_eval.drop(df_eval.columns[0], axis=1)
    # print(df['y'].value_counts())
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9, random_state=42, stratify=Y)
    for model in models:
        model.fit(X_train, Y_train)
        model.fit(X_train, Y_train)
        accuracy = model.score(X_test, Y_test)
        print("\n")
        model_name = str(model).split("(")[0]  # Extract the model name
        print(model_name)
        print(f"your score is {accuracy * 100} % Babeh")

    #  model.score(X_test, Y_test)


      # Y_train is your original class labels
    # (df_eval.shape)
    # print(df_eval.columns)
    # X_train = X
    # Y_train = Y
    # X_test = df_eval

    ########################### GRID SEARCH ############################################

    ####################### DECISION TREES ##########################################
    param_gridD = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt'],
    }

    ####################### BOOSTCLASSIFIER ##########################################
    param_gridB = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0]
    }

    ####################### XGBClassifier ##########################################
    param_gridX = {
        'n_estimators': [100, 200, 300],  # You can adjust these values
        'max_depth': [3, 4, 5, 6],  # You can adjust these values
        'learning_rate': [0.01, 0.1, 0.2],  # You can adjust these values
        'subsample': [0.7, 0.8, 0.9],  # You can adjust these values
        'colsample_bytree': [0.7, 0.8, 0.9],  # You can adjust these values
        'reg_alpha': [0, 0.1, 0.5],  # You can adjust these values
        'reg_lambda': [0, 0.1, 0.5],  # You can adjust these values
    }






    # Create a GridSearchCV object with cross-validation
    """grid_search = GridSearchCV(estimator=model_XGBC_Normal, param_grid=param_gridX, cv=5, scoring='accuracy')

    # Fit the grid search to your data
    grid_search.fit(X_train, Y_train)

    # Get the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Print the hyperparameters of the best model
    print(f"Best hyperparameters: {best_params}")
    print(f"Best model: {best_model}")

    Y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy of best model of {model_XGBC}: {accuracy * 100:.2f}%")
    """
    ########################### GRID SEARCH ############################################

    # i want the false positives and false negatives
    y_pred = model_randfore.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    num_samples = len(Y_test)
    num_classes = len(cm)  # Number of classes


    for model in models:
        model_name = str(model).split("(")[0]  # Extract the model name
        print(model_name)
        cross_val(model, X, Y, cv=5)

"""
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
"""

train()