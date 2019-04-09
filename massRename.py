from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os

lowest_false_neg=0
lowest_name = ''
is_lowest = False

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
    print("False Negative Percentage: " + "{0:.2f}".format(perc_false_neg) + ", False Positive Percentage: " + "{0:.2f}".format(perc_false_positive))
    return {perc_passed, perc_false_positive, perc_false_neg}

def model_rename():
    for filename in os.listdir('./Models'):
        if filename.endswith('.h5'):
            classifier = load_model('./Models/' + filename)
            Y_pred = classifier.predict(X_test)
            Y_pred = (Y_pred > 0.5)
            cm = confusion_matrix(Y_test, Y_pred)
            model_info = test_Accuracy(cm) # returns array [0] = accuracy, [1] = false pos, [2] = false neg
            new_name = filename[:-3]
            os.rename(filename, new_name)
            print(current_name)
            print(filename)

            #os.rename(filename, filename + str(model_info[0]) + str(model_info[1]) + str(model_info[2]) + '.h5')
        else:
            continue
    

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

model_rename()

print('The Model with the least false negatives is ' + highest_name)