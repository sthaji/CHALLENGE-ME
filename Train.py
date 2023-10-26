from sklearn.metrics import classification_report, accuracy_score

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import random
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


########################## K CROSS ##########################################
# Perform k-fold cross-validation and get accuracy scores for each fold
def cross_val(model, X, Y, cv=5):
    accuracy_scores = cross_val_score(model, X, Y, cv=None, scoring='accuracy')

    # accuracy scores for each fold
    for fold, accuracy in enumerate(accuracy_scores):
        print(f"Fold {fold + 1}: Accuracy = {accuracy * 100:.2f}%")

    # mean and standard deviation of accuracy scores
    mean_accuracy = accuracy_scores.mean()
    std_accuracy = accuracy_scores.std()
    print(f"Mean Accuracy = {mean_accuracy * 100:.2f}%")
    print(f"Standard Deviation of Accuracy = {std_accuracy * 100:.2f}%")
########################## K CROSS ##########################################



def clean_data(df, df_eval):
    label_encx = LabelEncoder() # for x7
    label_ency = LabelEncoder() # for y
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
    df['x7'] = label_encx.fit_transform(df['x7'])
    df_eval['x7'] = label_encx.transform(df_eval['x7'])
    ################ REPLACE MISSTYPES #################################

    ####################### CHECK FOR NULL ##############################
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
    df_eval['x1'] = df_eval['x1'].astype('float64')
    #######################turn into float##############################

    ##############print max min values of x1############################
    print(df['x1'].max())
    print(df['x1'].min())
    print(df['x1'].mean())
    # delete row where x1 is max value
    df = df[df.x1 != df.x1.max()]
    print(df['x1'].mean())
    ##############print max min values of x1############################


    ####################CHECK FOR OUTLIERS################################
    """
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
    ####################CHECK FOR OUTLIERS#################################


    ###################### DROP COLUMNS ###################################
    print(df['x12'].value_counts())
    # drop column named Unnamed: 0


    print(df.columns)
    df = df.drop('x12', axis=1)
    df_eval = df_eval.drop('x12', axis=1)
    df = df.drop('x3', axis=1)
    df_eval = df_eval.drop('x3', axis=1)
    df = df.drop('x13', axis=1)
    df_eval = df_eval.drop('x13', axis=1)
    df = df.drop('x2', axis=1)
    df_eval = df_eval.drop('x2', axis=1)
    #df = df.drop('x9', axis=1)
    #df = df.drop('x1', axis=1)
    df = df.drop(df.columns[0], axis=1)
    df_eval = df_eval.drop('Unnamed: 0', axis=1)
    # print first 20 y values
    print(df)
    df['y'] = label_ency.fit_transform(df['y'])
    print(df)
    print(df.head())
    print(df_eval.head())
    ###################### DROP COLUMNS ###################################

    return df, df_eval, label_ency

def train():
    # import lasso regression
    # model = LogisticRegression(penalty='l1', solver='liblinear')
    # model = LogisticRegression(penalty='l2', solver='liblinear)
    best_params_d1 = {
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 100
    }

    best_params_d2 = {
        'max_depth': None,
        'max_features': 'sqrt',
        'min_samples_leaf': 1,
        'min_samples_split': 2,
        'n_estimators': 300
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

    best_params3 = {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 10, 'n_estimators': 100,
                      'subsample': 0.8}

    best_params4 ={'learning_rate'
    : 0.1,
    'max_depth'
    : 4,
    'min_samples_split'
    : 10,
    'n_estimators'
    : 100,
    'subsample': 0.8
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


    model_randfore = RandomForestClassifier()
    model_randforeGrid = RandomForestClassifier(**best_params_d1)
    model_randforeGrid2 = RandomForestClassifier(**best_params_d2) ## This one is trained for the removed x3 and x13
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

    model_gradient = GradientBoostingClassifier()
    model_gradientBoost1 = GradientBoostingClassifier(**best_params)
    model_gradientBoost2 = GradientBoostingClassifier(**best_params2)
    model_gradient3 = GradientBoostingClassifier(**best_params3) ## SELECT
    model_gradient4 = GradientBoostingClassifier(**best_params4)

    models = [model_randfore, model_XGBC_Normal, model_XGBC, model_XGBC2, model_XGBC3, model_gradient, model_gradientBoost1, model_gradientBoost2, model_gradient3, model_gradient4]


    # Load the dataset
    # Get the parent directory path of the current script file
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path_Train = os.path.join(parent_directory, 'Challenge-me/TrainOnMe.csv')
    dataset_path_Evaluate = os.path.join(parent_directory, 'Challenge-me/EvaluateOnMe.csv')
    df = pd.read_csv(dataset_path_Train, sep=',')
    df_eval = pd.read_csv(dataset_path_Evaluate, sep=',')


    df, df_eval, label_encoder = clean_data(df,df_eval)




    print(df['y'].value_counts())
    Y = df['y']
    X = df.drop('y', axis=1)
    #scaler = MinMaxScaler()
    print("X HEAD")
    print(X.head())
    #X = scaler.fit_transform(X.values)
    n_pca_components = 9  # Choose the number of PCA components
    pca = PCA(n_components=n_pca_components)
    X = pca.fit_transform(X)
    # df_eval = df_eval.drop(df_eval.columns[0], axis=1)
    # print(df['y'].value_counts())
    ############################# divide into training and test set########################################
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, random_state=42, stratify=Y)
    ############################# divide into training and test set########################################



    #############################Test accuracy for all models########################################
    for model in models:
        model.fit(X_train, Y_train)
        accuracy = model.score(X_test, Y_test)
        print("\n")
        model_name = str(model).split("(")[0]  # Extract the model name
        print(model_name)
        print(f"your score is {accuracy * 100} % Babeh")
    #############################Test accuracy for all models########################################



    #############################k-fold cross validation########################################
    for model in models:
        model_name = str(model).split("(")[0]  # Extract the model name
        print(model_name)
        cross_val(model, X, Y, cv=5)
    #  model.score(X_test, Y_test)
    #############################k-fold cross validation########################################



    ########################### CODE FOR GRID SEARCH #################################################################


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
        'subsample': [0.8, 0.9, 1.0],
        'min_samples_split': [2, 5, 10]
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

    ########################### CODE FOR GRID SEARCH ###########################################################################
    """
    modelx = GradientBoostingClassifier()
    # Create a GridSearchCV object with cross-validation
    grid_search = GridSearchCV(estimator=modelx, param_grid=param_gridB, cv=5, scoring='accuracy')

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
    print(f"Accuracy of best model of {model_randfore}: {accuracy * 100:.2f}%")
    """
    ########################### CODE FOR GRID SEARCH ###########################################################################




    ############################# PICK AND EXTRACT data from chosen MODEL ########################################
    # i want the false positives and false negatives

    model_final = GradientBoostingClassifier(**best_params3) 
    model_randfore_final = GradientBoostingClassifier()
    # print first 5 rows of X
    print(Y.shape)
    model_final.fit(X,Y)
    # check that df_eval has the same columns as X
    print(df_eval.columns)
    df_eval = pca.transform(df_eval)
    print(X.shape)
    print(df_eval.shape)
    labels = model_final.predict(df_eval)
    ### comparing that he file has been translated correclty
    with open('labelsNOTIT.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)
    print(labels)
    #transform labels to original labels
    labels = label_encoder.inverse_transform(labels)
    print(labels)
    # save labels in a txt file
    with open('Sams_predictions3.txt', 'w') as f:
        for item in labels:
            f.write("%s\n" % item)

train()