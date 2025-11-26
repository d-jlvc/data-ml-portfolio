""" 
>. Heart Disease Dataset - Exploratory Data Analysis
>. Author: Danilo Jelovac
>. Training the top ranked model shown in 
    ./notebooks/heart_disease_eda.ipynb.

>>. Goal:
---------
- Our goal here is to test retrained model, `RandomForest`
to see if it correctly predicts heart disease outcome. We 
trained it on prepared and preprocessed data, our Heart_Disease_Dataset.

>>. Additional info:
--------------------
- This will be commented in a step-by-step manner as someone just starting
to learn may read this, and I want it to be as clear as possible.

>>. Requirements:
------------
- pandas

"""

# -------------------
# Importing libraries:
# -------------------

import pandas as pd
import joblib


# -----------------
# Testing our model:
# -----------------

PROJECT_FOLDER = 'Heart_Disease_Dataset'

MODEL_FOLDER_NAME = 'models'
SAMPLE_FOLDER_NAME = 'datasets'
MODEL_FILE_NAME = 'ml_randomforest'
SAMPLE_FILE_NAME = 'model_test_data'
MODEL_FILE_EXTENSION = '.pkl'
SAMPLE_FILE_EXTENSION = '.csv'

MODEL_FILE_PATH = f'./{PROJECT_FOLDER}/{MODEL_FOLDER_NAME}/{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}'
SAMPLE_FILE_PATH = f'./{PROJECT_FOLDER}/{SAMPLE_FOLDER_NAME}/{SAMPLE_FILE_NAME}{SAMPLE_FILE_EXTENSION}'

# -- Loading the model:
model = joblib.load(MODEL_FILE_PATH)
sample_data = pd.read_csv(SAMPLE_FILE_PATH)


# -- Running the testing program:
if __name__ == "__main__":
    
    # -- Testing the model:
    print("""
    ==== HEART DISEASE DATASET MODEL TEST ====
    +----------------------------------------+
        """)
    # -- Sample data presentation:
    print(sample_data.head())

    # -- Predictions and probabilities:
    predictions = model.predict(sample_data)
    probabilities = model.predict_proba(sample_data)


    print(f"""
1. Predictions:

[0 - No Disease, 1 - Heart Disease] -> {predictions}
---------------------------------------
2. Probabilities:

[No / Yes]
-------------  
{probabilities}
-------------
        """)



# --------------------------------------------------------------
# TESTING THE TRAINED MODEL ON MANUALLY CREATED PATIENTS:
# --------------------------------------------------------------
#
# This code block creates a small, manually constructed set of
# patients (sample_data) to test how our trained pipeline handles
# new, previously unseen data.
#
# The goal is to confirm the following:
# 1. The pipeline correctly applies preprocessing (encoding, scaling)
#    to inputs shaped like the original dataset.
# 2. The model produces predictions + probability scores
#    for these new patients.
#
# --------------------------------------------------------------
# IMPORTANT:
# All features must match the categories used in the training data:
#
# - 'Sex'                  → ['Male', 'Female']
# - 'ChestPain'            → ['TypicalAngina', 'AtypicalAngina',
#                             'NonAnginalPain', 'Asymptomatic']
# - 'FastingBloodSugar'    → ['Present', 'NotPresent']
# - 'RestECG'              → ['Normal', 'Abnormality']
# - 'ExerciseAngina'       → ['Induced', 'NotInduced']
# - 'PeakSlope'            → ['Upsloping', 'Flat', 'Downsloping']
# - 'ThaliumStressTest'    → ['Normal', 'ReversibleDefect',
#                             'FixedDefect', 'NotDescribed']
#
# Numerical columns remain numerical (Age, OldPeak, Cholesterol, …)
#
# --------------------------------------------------------------
# MODEL OUTPUTS:
#   0 → No Heart Disease
#   1 → Heart Disease Present
#
# We also extract prediction probabilities so we can see how
# confident the model is in its decision.
#
# --------------------------------------------------------------
