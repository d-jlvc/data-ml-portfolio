""" 
>. Titanic Dataset - Exploratory Data Analysis
>. Author: Danilo Jelovac
>. Training the top ranked model shown in 
    ./notebooks/titanic_data_ml_eval.ipynb.

>>. Goal:
---------
- Our goal here is to retrain the chosen model, `Logistic Regression`
to correctly predict passenger survival. We're training it on prepared
and preprocessed data, our Titanic-Dataset.

>>. Additional info:
--------------------
- This will be commented in a step-by-step manner as someone just starting
to learn may read this, and I want it to be as clear as possible.

>>. Requirements:
------------
- pandas
- skicit-learn

"""

# -------
# Imports:
# -------


import pandas as pd  # --loading data...
# --
from sklearn.compose import ColumnTransformer  # --neccessary tools...
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# --
from sklearn.linear_model import LogisticRegression  # --our chosen model...
# --
import joblib  # --for saving the model


# -------------------
# Loading the dataset:
# -------------------


# ---------------------------
FOLDER_NAME = 'datasets'
FILE_NAME = 'Titanic-Dataset'
EXTENSION = '.csv'

FILE_PATH = f'./Titanic_Dataset/{FOLDER_NAME}/{FILE_NAME}_ML_Train{EXTENSION}'
# ---------------------------

# --Loading the dataset:
try:
    data = pd.read_csv(FILE_PATH)
    print(f"File '{FILE_NAME}_ML_Train{EXTENSION}' loaded!\n")
except FileNotFoundError:
    print(f"File not found! Please check if path is correct: '{FILE_PATH}'")
    
# print(list(data.columns))  # -- used for checking the columns.


# ------------------------------------------------
# Retraining the chosen model [LogisticRegression]:
# ------------------------------------------------


# -- Getting data that our model will use to learn patterns:
data_for_learing = data[['Title', 'Sex', 'Age', 'FamilyStatus', 'Class', 'Fare']]

# -- Step#1: Splitting the data into input (X) and output (y):
X = data_for_learing
y = data['SurvivedBinary']

# -- Step#2: Splitting `training and testing` parts (80:20):
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -- Step#3: Preprocess data so model can understand it:
preprocess_data = ColumnTransformer(
    transformers=[
        ('Categoricals', OneHotEncoder(handle_unknown='ignore'), ['Title', 'Sex', 'FamilyStatus']),
        ('Numericals', MinMaxScaler(), ['Age', 'Class', 'Fare'])
    ]
)

# -- Step#4: Pipeline - Preprocessed data > Choose a model:
pipeline = Pipeline(steps=[
   ('PreprocessedData', preprocess_data),
   ('Model', LogisticRegression(max_iter=200))
])

# -- Step#5: Training the model:
pipeline.fit(X_train, y_train)

# -- Step#6: Prediction and classification report:
y_predict = pipeline.predict(X_test)

print(f"\n==== Model [LogisticRegression] CLASSIFICATION REPORT ====\n")
print(classification_report(y_test, y_predict), "\n")

# -- Step#7: Saving the model:

# ----------------------------
SAVE_FOLDER_NAME = 'models'
SAVE_FILE_NAME = 'titanic_logisticregression_model'
SAVE_EXTENSION = '.pkl'
# ----------------------------
SAVE_FILE_PATH = f'./Titanic_Dataset/{SAVE_FOLDER_NAME}/{SAVE_FILE_NAME}{SAVE_EXTENSION}'


try:
    joblib.dump(pipeline, SAVE_FILE_PATH)
    print(f"\n>>>. Model '{SAVE_FILE_NAME}' successfuly created!")
    print(f">>>. Model saved in {SAVE_FILE_PATH}.")
except Exception:
    print("Something went wrong... Please, check your code and/or path.")