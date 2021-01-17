# Compare Algorithms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
# load dataset
arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

Y = arq["Y"]
X = arq.drop('Y', axis=1)
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC(C=1, kernel='poly', degree=4, gamma='scale', random_state=seed)))
models.append(('MLPClassifier', MLPClassifier(verbose = False, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(500), activation = 'relu',
                              batch_size=69, learning_rate_init=0.001, random_state=seed)))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
