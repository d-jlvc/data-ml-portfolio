""" 
>. Heart Disease Dataset - Exploratory Data Analysis
>. Author: Danilo Jelovac
>. Training the top ranked model shown in 
    ./notebooks/heart_disease_eda.ipynb.

>>. Goal:
---------
- Our goal here is to retrain the chosen model, `RandomForest` to
correctly learn and predict the cause of Heart Diseases with patients.

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
from sklearn.ensemble import RandomForestClassifier  # --our chosen model...
# --
import joblib  # --for saving the model


# -------------------
# Loading the dataset:
# -------------------

PROJECT_FOLDER = 'Heart_Disease_Dataset'
FOLDER_NAME = 'datasets'
FILE_NAME = 'heart_disease_ml'
EXTENSION = '.csv'

FILE_PATH = f'./{PROJECT_FOLDER}/{FOLDER_NAME}/{FILE_NAME}{EXTENSION}'
# ---------------------------

# --Loading the dataset:
try:
    ml_df = pd.read_csv(FILE_PATH)
    print(f"File '{FILE_NAME}{EXTENSION}' loaded!\n")
except FileNotFoundError:
    print(f"File not found! Please check if path is correct: '{FILE_PATH}'")


# ------------------------------------------------
# Retraining the chosen model [LogisticRegression]:
# ------------------------------------------------


# -- Getting data that our model will use to learn patterns:
input_data = ml_df.drop(columns='HeartStatusBinary')
output_data = ml_df['HeartStatusBinary']

# -- Splitting the data into input (X) and output (y):
X = input_data
y = output_data

# -- Splitting `training and testing` parts (80:20):
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -- Dividing categorical and numerical data:
categorical_data = ['Sex', 'ChestPain', 'FastingBloodSugar', 'RestECG',
                    'ExerciseAngina', 'PeakSlope', 'ThaliumStressTest',]

numerical_data = ['Age', 'RestBloodPressure', 'Cholesterol', 
                  'MaxHeartRate', 'OldPeak', 'NumMajorVessels']

# -- Preprocess data so model can understand it:
preprocess_data = ColumnTransformer(
    transformers=[
        ('Categoricals', OneHotEncoder(handle_unknown='ignore'), categorical_data),
        ('Numericals', MinMaxScaler(), numerical_data)
    ]
)

# -- Pipeline - Preprocessed data > Model reading prepared data:
pipeline = Pipeline(steps=[
   ('PreprocessData', preprocess_data),
   ('Model', RandomForestClassifier())
])

# -- Training the model:
trained_model = pipeline.fit(X_train, y_train)

# -- Prediction and classification report:
y_predict = trained_model.predict(X_test)

print(f"\n--[RandomForestClassifier]:\n")
print(classification_report(y_test, y_predict), "\n")

# -- Saving the model:
SAVE_FOLDER_NAME = 'models'
SAVE_FILE_NAME = 'ml_randomforest'
SAVE_EXTENSION = '.pkl'

SAVE_FILE_PATH = f'./{PROJECT_FOLDER}/{SAVE_FOLDER_NAME}/{SAVE_FILE_NAME}{SAVE_EXTENSION}'


try:
    joblib.dump(trained_model, SAVE_FILE_PATH)
    print(f"\n>. Model '{SAVE_FILE_NAME}' successfuly created!")
    print(f">. Model saved in {SAVE_FILE_PATH}.")
except Exception:
    print("Something went wrong... Please, check your code and/or path.")
