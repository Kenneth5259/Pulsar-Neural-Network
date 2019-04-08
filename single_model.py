import pandas as pd
import os


def setClassifier(filePath):
    if ((os.path.isfile(filePath))):
        from keras.models import load_model
        print('Model Loaded')
        classifier = load_model(filePath)
    else:
        # Model Development
        from keras.models import Sequential
        from keras.layers import Dense

        classifier = Sequential()

        classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu', input_dim=8))
        classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        classifier.fit(X_train, Y_train, batch_size=10, epochs=100)
        classifier.save(filePath)
        print('Model Created')
    return classifier

def test_Accuracy(cm):
    passed = float(cm[0][0]) + float(cm[1][1])
    failed = float(cm[1][0]) + float(cm[0][1])

    perc_passed = passed/(passed+failed) * 100
    perc_failed = failed/(passed+failed) * 100
    perc_false_neg = float(cm[0][1])/(passed+failed) * 100
    perc_false_positive = float(cm[1][0])/(passed+failed) * 100

    return("Passed: " + "{0:.2f}".format(perc_passed) + ", Failed: " + "{0:.2f}".format(perc_failed) + "\nFalse Negatives: " + "{0:.2f}".format(perc_false_neg) + ", False Positive: " + "{0:.2f}".format(perc_false_positive))

modelName = '8-Layersigmoidsoftsignsoftsignreluselurelusoftsignsoftplus.h5'
filePath = './Models/' + modelName

#Data Loading
dataset = pd.read_csv('./HTRU2/HTRU_2.csv')

X = dataset.iloc[:, 0:8]
Y = dataset.iloc[:, 8]


#Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = setClassifier(filePath)

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
#'8-Layer sigmoid softsign softsign relu selu relu softsign softplus.h5'
print("8 Layer Sequential Model")
print("Layer 1 - Sigmoid, Layer 2 - Soft Sign")
print("Layer 3 - Soft Sign, Layer 4 - Rectifier")
print("Layer 5 - SELU, Layer 6 - Rectifier")
print("Layer 7 - Soft Sign, Layer 8 - Soft Plus")
print(test_Accuracy(cm))

#[0][0], [1][1] are pass values, [0][1], [1][0] are fails

