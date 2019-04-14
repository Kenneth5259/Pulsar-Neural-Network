import matplotlib.pyplot as plt
import classifier_library as cl
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

def acc_vs_layer():
    accuracies = []
    layers = []
    for filename in os.listdir('./Models/Performance'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                accuracies.append(float(x[0]))
                layers.append(float(filename[:1]))

    for filename in os.listdir('./Models/Suboptimal'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                if(float(x[0]) > 95):
                    accuracies.append(float(x[0]))
                    layers.append(float(filename[:1]))

    plt.scatter(layers, accuracies, np.pi*3, (0,0,0), alpha=0.5)
    plt.title('Model Accuracy vs Layers')
    plt.xlabel('#' + ' Layers')
    plt.ylabel('%' + ' Total Accuracy')
    #plt.show()
    plt.savefig('Accuracy vs Layers.png')

def layers_vs_runtime():
    rt = []
    layers = []
    for filename in os.listdir('./Models/Performance'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                layers.append(float(filename[:1]))
                rt.append(cl.runtime('./Models/Performance/' + filename))

    for filename in os.listdir('./Models/Suboptimal'):
            if filename.endswith('.h5'):
                x = re.findall("\d+\.\d+", filename)
                if(float(x[0]) > 95):
                    layers.append(float(filename[:1]))
                    rt.append(cl.runtime('./Models/Suboptimal/' + filename))
    plt.scatter(layers, rt, np.pi*3, (0,0,0), alpha=0.5)
    plt.title('Model Accuracy vs Layers')
    plt.xlabel('#' + ' Layers')
    plt.ylabel('%' + ' Runtimes')
    plt.show()
    #plt.savefig('Accuracy vs Layers.png')
    
layers_vs_runtime()