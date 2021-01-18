# Compare Algorithms
import pandas as pd
import numpy as np
from keras.datasets import imdb
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# load dataset
arq = pd.read_csv('dataset1.csv', sep=';')  
del arq["imgName"]
X = arq.drop('Y', axis=1)
Y = arq["Y"]

#AQUI DEVE VIR UM LAÇO FOR DE 1 A 30
seed = np.random.randint(30)
import random 
np.random.seed(seed)

max_features = 9
x_treino, x_teste, y_treino, y_teste = train_test_split(X, Y, test_size=0.2)
   
#Normalizando os dados
#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
X_norm = normalizador.fit_transform(x_treino)
Y = y_treino

# prepare configuration for cross validation test harness

# prepare models
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=1)))
models.append(('SVM', SVC(C=0.8, kernel='poly', degree=2, gamma='scale')))
'''
models.append(('MLPClassifier', MLPClassifier(verbose = False, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(500), activation = 'relu',
                              batch_size=50, learning_rate_init=0.001, random_state=seed)))
'''
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, shuffle = True)
for name, model in models:
	cv_results = model_selection.cross_val_score(model, X_norm, Y, cv=kfold, scoring=scoring)
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
