import classifier_library as cl


for i in range(1, 200):
    cl.createClassifier(5)

cl.model_rename()
cl.filter_by_accuracy()

