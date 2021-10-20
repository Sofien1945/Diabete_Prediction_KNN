""""Diabete prediction usng KNN
Part of Simplean Machine learning course
Date: 20.10.2021
Done By: Sofien Abidi"""

#Import Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import Dataset
file_path = 'C:/Users/Sofien/Desktop/diabetes.csv'
dataset = pd.read_csv(file_path)

#Import KNN model, algothims score verification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

#Replace Zeros
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

#Split data into train and test
X = dataset.iloc[:,0:8]
y = dataset.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Training the model
classifier = KNeighborsClassifier(n_neighbors=11, metric='euclidean', p=2)
classifier.fit(X_train, y_train)

#Testing the model
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
score = classifier.score(X_test, y_test)
score_f1 = f1_score(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print(score_f1)
print(acc_score)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = f'Accuracy Score: {score:.2f}'
plt.title(all_sample_title, size = 12)
plt.show()
