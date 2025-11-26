### Heart Disease Dataset – Data Cleaning, Visualization & Machine Learning  
#### Author: Danilo Jelovac  

This project focuses on a complete end-to-end workflow applied to the **Heart Disease Dataset** found on Kaggle.  
It is structured to reflect real data analysis practices: from raw data, through cleaning and feature engineering and data visualization, to model training and evaluation.

---

#### 1) Project Structure

```
Heart_Disease_Dataset/
│
├── datasets/
│   ├── heart_disease_clean.csv            # After cleaning & prep
│   ├── heart_disease_raw.csv              # Raw dataset
│   ├── heart_disease_ml.csv               # Added 'SurvivedBinary'
│   └── model_test_data.csv                # Sample for model testing
│
├── notebooks/
│   ├── heart_disease_eda.ipynb            # Jupyter Notebook, Cleaning, Viz, MachineLearning
│
├── model_source/
│   ├── train_model.py                     # Training pipeline
│   └── test_model.py                      # Prediction script
│
├── models/
│   └── ml_randomforest.pkl                # Best ranked model
│
├── exports/
│   └── *.html                             # Exported notebooks (optional)
│
├── README.md
│ 
│ 
└── requirements.txt                       # Necessary python libraries
```

---

#### Step 1. Data Cleaning, EDA, ML Evaluation
Performed in  
```
notebooks/heart_disease_eda.ipynb
```

Key operations:
- Handling missing values  
- Converting categorical features (Title extraction, family status)  
- Removing unused or misleading columns  
- Saving cleaned versions into `/datasets/`  
- Encoding binary target (`SurvivedBinary`)  
- Sex, Class, and Fare show the strongest correlation with survival  
- Age distribution shows a weak relationship  
- “Women and children first” pattern visible  
- Family presence slightly increases chance of survival  
- Visualizations using matplotlib and seaborn libraries  

Evaluated models:
- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- Linear SVC  

Results:
- **Random Forest** and **Decision Tree** produced the most stable and interpretable performance.

---

#### Step2. Model Training & Prediction Scripts

**train_model.py**  
- Loads ML-ready dataset  
- Applies preprocessing (OneHotEncoder + MinMaxScaler)  
- Trains Random Forest  
- Saves the model to `/models/`

**test_model.py**  
- Loads the saved model  
- Provides sample inputs  
- Outputs both prediction and probability (`predict_proba`)  

---

#### Final Notes
This project showcases:
- A full ML workflow  
- Clean structure aligned with industry standards  
- Reusable training pipeline  
- Clear notebook documentation  

---

### Contact  
LinkedIn: https://www.linkedin.com/in/danilo-jelovac-b1b7a5396/
GitHub: https://github.com/d-jlvc/data-ml-portfolio
