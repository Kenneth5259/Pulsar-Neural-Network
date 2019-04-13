import matplotlib.pyplot as plt
import numpy as np
import os
import re



def acc_vs_neg():
    accuracies = []
    false_negs = []
    for filename in os.listdir('./Models/Performance'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                accuracies.append(float(x[0]))
                false_negs.append(float(x[2]))

    for filename in os.listdir('./Models/Suboptimal'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                if(float(x[0]) > 95):
                    accuracies.append(float(x[0]))
                    false_negs.append(float(x[2]))

    plt.scatter(false_negs, accuracies, np.pi*3, (0,0,0), alpha=0.5)
    plt.title('Model Accuracy vs False Negatives')
    plt.xlabel('%' + ' False Negative')
    plt.ylabel('%' + ' Total Accuracy')
    plt.savefig('Accuracy vs False Neg.png')

acc_vs_neg()