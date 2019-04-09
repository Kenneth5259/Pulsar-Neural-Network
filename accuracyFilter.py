import os
import re

path = os.getcwd() + '/Models/'
count = 0

def filter_by_accuracy():
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
