import os
activation_functions = ['relu', 'sigmoid', 'hard_sigmoid', 'selu', 'softplus', 'softsign']
patches = ["ur", "uh", "elus", "dr", "ds", "dh", "sr", "ss", "sh", "nr", "nh", "ns", "yers"]
dictPatch = {
    "ur": "u-r",
    "uh": "u-h",
    "elus": "elu-s", # special case to avoiud breaking softplus
    "dr": "d-r",
    "ds": "d-s",
    "dh": "d-h",
    "sr": "s-r",
    "ss": "s-s",
    "sh": "s-h",
    "nr": "n-r",
    "nh": "n-h",
    "ns": "n-s",
    "yers": "yer-s"
}

def activation_rename():
    for filename in os.listdir('./Models'):
        if filename.endswith('.h5'):
            filename = (os.getcwd() + "/Models/" + filename)
            tempname = filename
            print(tempname)
            for i in range(0,13):
                z = patches[i]
                y = dictPatch[z]
                tempname = tempname.replace(z, y)
            os.rename(filename, tempname)

activation_rename()