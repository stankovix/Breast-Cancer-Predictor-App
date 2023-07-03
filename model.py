import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings('ignore')

### Import Dataset
df = pd.read_csv("BreastCancer_dp.csv")
# we change the class values (at the column number 2) from B to 0 and from M to 1
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)

### Splitting Data

X = df[['clump_thickness','size_uniformity','shape_uniformity','marginal_adhesion','epithelial_size','bare_nucleoli','bland_chromatin','normal_nucleoli','mitoses']]
y = df['class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

#### Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )
KNN.fit(x_train, y_train)
predicted_1 = KNN.predict(x_test)
predicted_1

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("Confusion Matrix : \n\n" , confusion_matrix(predicted_1,y_test))

print("Classification Report : \n\n" , classification_report(predicted_1,y_test),"\n")


pickle.dump(KNN, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model)