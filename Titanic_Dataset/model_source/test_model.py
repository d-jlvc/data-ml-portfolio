""" 
>. Titanic Dataset - Exploratory Data Analysis
>. Author: Danilo Jelovac
>. Testing the top ranked model shown in 
    ./notebooks/titanic_data_ml_eval.ipynb.

>>. Goal:
---------
- Our goal here is to test retrained model, `Logistic Regression`
to see if it correctly predicts passenger survival. We trained it 
on prepared and preprocessed data, our Titanic-Dataset.

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


# ------------------------------------------
MODEL_FOLDER_NAME = 'models'
MODEL_FILE_NAME = 'titanic_logisticregression_model'
MODEL_FILE_EXTENSION = '.pkl'

MODEL_FILE_PATH = f'./Titanic_Dataset/{MODEL_FOLDER_NAME}/{MODEL_FILE_NAME}{MODEL_FILE_EXTENSION}'
# ------------------------------------------

model = joblib.load(MODEL_FILE_PATH)


print("""
+-------------------------------------+
==== TITANIC DATASET MODEL TESTING ====
+-------------------------------------+
      """)

sample_data = pd.DataFrame({
    'Title': ['Mr', 'Mrs', 'Miss', 'Mr', 'Mrs', 'Miss', 'Mr', 'Mrs', 'Miss', 'Mr'],
    'Sex': ['male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male'],
    'Age': [35, 40, 19, 8, 50, 22, 45, 60, 28, 32],
    'FamilyStatus': ['single', 'with_family', 'single', 'with_family', 'single', 'single', 'single', 'with_family', 'with_family', 'single'],
    'Class': [3, 2, 3, 1, 3, 3, 1, 2, 1, 2],
    'Fare': [8.5, 32.0, 7.9, 55.0, 6.2, 7.0, 50.0, 15.0, 80.0, 12.0]
})

predictions = model.predict(sample_data)
probabilities = model.predict_proba(sample_data)


print(f"""
+-------------------------------------+
Predictions -> {predictions}
---------------------------------------
Probabilities:
   [Survived / Killed]
{probabilities}
+-------------------------------------+
      """)


# --------------------------------------------------------------
# TESTING THE TRAINED MODEL ON MANUALLY CREATED PASSENGERS GUIDE:
# --------------------------------------------------------------
#
# We createt a small, controlled sample of passengers (sample_data).
# The goal is to verify that our trained pipeline (preprocessing + model)
# correctly handles new, unseen inputs using the same feature structure
# as during training!
#
# IMPORTANT THING TO KNOW:
# - 'Title's must match the cleaned categories: ['Mr', 'Mrs', 'Miss']
# - 'Sex' must be one of ['male', 'female']
# - 'FamilyStatus' must match the engineered categories used in training
#
# --------------------------------------------------------------
# The model outputs:
#   0 → passenger did NOT survive
#   1 → passenger DID survive
#
#---------------------------------------------------------------
#
# Additionally, we extract prediction probabilities so we can understand
# how confident the model is for each prediction.
#
# ------------------------------------------------------------
