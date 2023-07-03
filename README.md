
# **Breast Cancer Prediction Model Using (KNN CLASSIFIER)**

Breast cancer is a prevalent and significant public health issue among women worldwide. Accurate early diagnosis plays a vital role in improving prognosis and survival rates. In addition, precise classification of breast tumors as benign or malignant is crucial for avoiding unnecessary treatments.

Using Python and data analysis techniques, we can leverage machine learning algorithms such as K-Nearest Neighbors (KNN) to aid in the early diagnosis and classification of breast cancer. KNN is a powerful algorithm that can analyze medical data and clinical variables to make accurate predictions.

 Flask application was also used to deploy the breast cancer diagnosis and classification model, allowing healthcare professionals to access it conveniently. The Flask framework provides a user-friendly interface, enabling seamless interaction with the machine learning model.

Through this application, healthcare professionals can input relevant patient data, and the KNN algorithm implemented in Python will provide predictions regarding the presence of breast cancer and the classification of tumors as benign or malignant. This empowers medical practitioners to make informed decisions about timely clinical treatments, reducing unnecessary interventions for patients with benign tumors.

By combining the power of Python, data analysis, KNN, and Flask deployment, we can contribute to improving breast cancer diagnosis and classification, ultimately leading to better patient outcomes and more efficient allocation of healthcare resources.


**Benign**: Not likely to get cancer (2) = 0

**Malignant**: Likely to get cancer (4)** = 1


## Authors

- [@octokatherine](https://github.com/stankovix)

Thanks To:
- [@octokatherine](https://www.linkedin.com/in/mrbriit/) for the Bootcamp. It was very reach in content.


## Installation

Kindly take note of the following libaries and models below:

```bash
  
import warnings
warnings.filterwarnings("ignore")

IMPORT THE FOLLOWING LIBRARIES:

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib import pylab as plt
from statsmodels.graphics.gofplots import qqplot
from IPython.core.interactiveshell import InteractiveShell
```
    
## Run Locally On PyCharm or any IDE

Clone the project

Install and run the following:




```bash
install flask


Run the following python code:


  python model.py
```

```bash
  python app.py
```

## **STEPS TAKEN IN THIS PROJECT:**

Below  are the steps I took in this Python data analysis project on breast cancer for benign or malignant classification, along with Flask for deployment:

1. Data Preparation:
   - Import the necessary Python libraries such as pandas, numpy,matplotlib.pyplot and scikit-learn.
   - Load the breast cancer dataset, which contains relevant features and target variables.
   - Perform data preprocessing tasks such as handling missing values, converting some columns/variables/features to required data type, and scaling numeric features.

2. Data Analysis and Model Development:
   - Explore and analyze the dataset using descriptive statistics, data visualization, and correlation analysis.
   - Split the dataset into training and testing sets.
   - Choose an appropriate machine learning algorithm for classification, such as K-Nearest Neighbors (KNN),RandomForestClassifier, and Support Vector Machine (SVC)
   - Train the modelS on the training data.
   - Evaluate the model's performance using appropriate evaluation metrics (e.g., accuracy, precision, recall, F1-score).

3. Flask Application Development:
   - Set up a Flask project and install the required dependencies.
   - Create a route in Flask to handle incoming requests for breast cancer classification.
   - Design an HTML form or user interface where users can input relevant patient data.
   - Implement the necessary backend logic to preprocess the input data and feed it into the trained KNN model for classification.(NOTE: THE KNN HAD THE BEST ACCURACY, SO WE USED IT FOR THE FLASK DEPLOYMENT).
   - Retrieve the model's prediction for benign or malignant classification.
   - Return the prediction to the user through the Flask application's response.


![Logo](https://github.com/stankovix/Breast-Cancer-Predictor-App/blob/main/FP.png?raw=true)

