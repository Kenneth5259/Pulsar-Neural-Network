from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import pandas as pd



def createClassifier(filePath, units, layers):
    from keras.models import Sequential
    from keras.layers import Dense

    classifier = Sequential()

    classifier.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=8))
    for i in range (0, layers):
        classifier.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, Y_train, batch_size=10, epochs=100)
    classifier.save(filePath)

def test_Accuracy(cm):
    passed = float(cm[0][0]) + float(cm[1][1])
    failed = float(cm[1][0]) + float(cm[0][1])

    perc_passed = passed/(passed+failed) * 100
    perc_failed = failed/(passed+failed) * 100

    return("Passed: " + "{0:.2f}".format(perc_passed) + ", Failed: " + "{0:.2f}".format(perc_failed))

modelName = 'triple-model.h5'
filePath = './Models/' + modelName

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



for i in range(1, 101):
    modelName = str(i) + '-layer.h5'
    filePath = './Models/' + modelName
    classifier = createClassifier(filePath, 5, i)
    