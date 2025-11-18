### Titanic Dataset – Data Cleaning, Visualization & Machine Learning  
#### Author: Danilo Jelovac  

This project focuses on a complete end-to-end workflow applied to the classic **Titanic Dataset**.  
It is structured to reflect real data analysis practices: from raw data, through cleaning and feature engineering and data visualization, to model training and evaluation.

---

#### 1) Project Structure

```
Titanic_Dataset/
│
├── datasets/
│   ├── Titanic-Dataset.csv                # Raw dataset
│   ├── Titanic-Dataset_Cleaned.csv        # After cleaning & prep
│   ├── Titanic-Dataset_ML_rdy.csv         # Added 'SurvivedBinary'
│   └── Titanic-Dataset_ML_Train.csv       # Final ML training set
│
├── notebooks/
│   ├── titanic_data_cleaning.ipynb        # Part 1: Cleaning
│   ├── titanic_data_visualization.ipynb   # Part 2: Visualization
│   └── titanic_data_ml_eval.ipynb         # Part 3: ML evaluation
│
├── model_source/
│   ├── train_model.py                     # Training pipeline
│   └── test_model.py                      # Prediction script
│
├── models/
│   └── titanic_logisticregression_model.pkl
│
├── exports/
│   └── *.html                              # Exported notebooks (optional)
│
└── README.md
```

---

#### Step1. Data Cleaning
Performed in  
```
notebooks/titanic_data_cleaning.ipynb
```

Key operations:
- Handling missing values  
- Converting categorical features (Title extraction, family status)  
- Removing unused or misleading columns  
- Saving cleaned versions into `/datasets/`

---

#### Step2. Exploratory Data Analysis (EDA)
Performed in  
```
notebooks/titanic_data_visualization.ipynb
```

Highlights:
- Encoding binary target (`SurvivedBinary`)  
- Sex, Class, and Fare show the strongest correlation with survival  
- Age distribution shows a weak relationship  
- “Women and children first” pattern visible  
- Family presence slightly increases chance of survival  
- Visualizations using matplotlib and seaborn libraries  

---

#### Step3. Machine Learning Evaluation
Performed in  
```
notebooks/titanic_data_ml_eval.ipynb
```

Evaluated models:
- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- Linear SVC  

Results:
- **Logistic Regression** produced the most stable and interpretable performance.

The final model was saved as a `.pkl` file for later predictions.

---

#### Step4. Model Training & Prediction Scripts

**train_model.py**  
- Loads ML-ready dataset  
- Applies preprocessing (OneHotEncoder + MinMaxScaler)  
- Trains Logistic Regression  
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
