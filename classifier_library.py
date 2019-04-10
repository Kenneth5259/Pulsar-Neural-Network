from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os
import random
import re

lowest_false_neg=0
lowest_name = ''
is_lowest = False
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'selu', 'softplus', 'softsign']
path = os.getcwd() + '/Models/'
# added to add a delimiter between each activation function

#function to test accuracy of each model's confusion matrix
def test_Accuracy(cm):
    global lowest_false_neg
    global is_lowest
    passed = float(cm[0][0]) + float(cm[1][1])
    failed = float(cm[1][0]) + float(cm[0][1]) #false positive + false negative

    perc_passed = passed/(passed+failed) * 100
    perc_failed = failed/(passed+failed) * 100
    perc_false_neg = float(cm[0][1])/(passed+failed) * 100
    perc_false_positive = float(cm[1][0])/(passed+failed) * 100

    if(lowest_false_neg <= 0):
        lowest_false_neg =  perc_passed
        is_lowest = True
    if(perc_false_neg < lowest_false_neg and perc_false_neg > 0):
        lowest_false_neg =  perc_passed
        is_lowest = True
    return ["{0:.2f}".format(perc_passed), "{0:.2f}".format(perc_false_positive), "{0:.2f}".format(perc_false_neg)]


#function to rename model with accuracy standard
def model_rename():
    for filename in os.listdir('./Models'):
        if filename.endswith('.h5'):
            classifier = load_model('./Models/' + filename)
            Y_pred = classifier.predict(X_test)
            Y_pred = (Y_pred > 0.5)
            cm = confusion_matrix(Y_test, Y_pred)
            model_info = test_Accuracy(cm) # returns array [0] = accuracy, [1] = false pos, [2] = false neg
            filename = (os.getcwd() + "/Models/" + filename)
            new_name = filename[:-3] + "-" + model_info[0] + "-" + model_info[1] + "-" + model_info[2] + '.h5'
            print("Old File: " + filename)
            print("New File: " + new_name)

            os.rename(filename, new_name)
        else:
            continue

#returns randomized layer count
def getLayerCount():
    return random.randint(1,11) #generates a number between 1 and 10

#returns randomized activation function
def getActivationFunction():
    return activation_functions[random.randint(0,5)]

#function to create and save classifier models when the number of units per layer is passed   
def createClassifier(units):
    from keras.models import Sequential
    from keras.layers import Dense
    layers = getLayerCount()
    filePath = './Models/' + str(layers) + '-Layer'
    classifier = Sequential()

    classifier.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=8))
    for i in range (0, layers):
        activationType = getActivationFunction()
        filePath += ('-' + activationType)
        classifier.add(Dense(units=units, kernel_initializer='uniform', activation=activationType))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, Y_train, batch_size=10, epochs=100)
    filePath += '.h5'
    classifier.save(filePath)


def filter_by_accuracy():
    count = 0
    for filename in os.listdir('./Models'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                accuracy = float(x[0])
                fp = float(x[1])
                fn = float(x[2])
                if(accuracy > 98.1 and fn > 0):
                    print(filename)
                    count += 1
                    os.rename(path + filename, path + 'Performance/' + filename)
                else:
                    os.rename(path + filename, path + 'Suboptimal/' + filename)
    print("There are " + str(count) + " models that meet minimum accuracy")

#Data Loading
dataset = pd.read_csv('./HTRU2/HTRU_2.csv')

X = dataset.iloc[:, 0:8]
Y = dataset.iloc[:, 8]

#Data Splitting

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
