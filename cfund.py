# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('cfund.csv')
dataset = dataset.drop(['id','name','city','state','country','blurb_length','launched_at','deadline','name_length','usd_pledged'],axis=1)

 
# simply change yes/no to 1/0 for successful and failed
dataset['status'].replace({'failed': 0, 'successful': 1},inplace = True)
#Let us get rid of all null values in df
dataset = dataset.dropna(how='any')
catcols=['currency','main_category','sub_category','start_Q','end_Q']
dataset = pd.get_dummies(dataset, columns=catcols)

X = dataset.iloc[:,0:201].values
X=np.delete(X, 2, axis=1)
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 224, activation = 'relu', input_dim = 200))


# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = keras.optimizers.Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
h = classifier.fit(X_train, y_train, batch_size = 64, epochs = 100, validation_data=(X_test, y_test))

plt.plot(h.history['acc'], label='Accuracy')
plt.plot(h.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(h.history['loss'], label='Loss')
plt.plot(h.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm, '\n')

print(classification_report(y_test, y_pred))

#classifier.save("cfund.h5")
#classifier.save_weights("cfund_weights.h5")