#BREAST CANCER ANALYSIS USING SVM 

#IMPORTING THE LIBRARIES
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#IMPORTING THE DATASET
data=pd.read_csv('breast_cancer.csv')
data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
data = data.set_index('id')
del data['Unnamed: 32']
Y = data['diagnosis'].values
X = data.drop('diagnosis', axis=1).values

#SPLITTING THE DATASET INTO TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.20, random_state=21)

# FEATURE SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_scaled = sc.transform(X_test)

# FITTING SVM TO THE TRAINING MODEL
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

#CHECKING THE ACCURACY
predictions = classifier.predict(X_test_scaled)
print("Accuracy score %f" % accuracy_score(Y_test, predictions))
