#Referencia
# https://machinelearningmastery.com/repeated-k-fold-cross-validation-with-python/#:~:text=Repeated%20k%2Dfold%20cross%2Dvalidation%20provides%20a%20way%20to%20improve,all%20folds%20from%20all%20runs

# compare the number of repeats for repeated k-fold cross-validation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot

#create dataset
dataset = pd.read_csv('dataset.csv', sep=';')  
del dataset["imgName"]
X = dataset.drop('Y', axis=1)
y = dataset["Y"]

normalizador = MinMaxScaler(feature_range = (0, 1))
X = normalizador.fit_transform(X)

# create model
KNN_model = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=1)
SVC_model = SVC(C=0.8, kernel='poly', degree=2, gamma='scale', random_state=1)
MLPClassifier_Model = MLPClassifier(verbose = True, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation = 'relu',
                              batch_size=200, learning_rate_init=0.001)

# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats, model):
	# prepare the cross-validation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=repeats, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

comparacao = []
# configurations to test do KNN
def test(model):
    repeats = range(1,30)
    results = list()
    for r in repeats:
        #evaluate using a given number of repeats
        scores = evaluate_model(X, y, r, model)
        # summarize
        print('>%d mean=%.4f se=%.3f' % (r, mean(scores), sem(scores)))
        # store
        results.append(scores)
    # plot the results
    pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
    pyplot.show()
    return 

print('KNN')
test(KNN_model)

print('SVC')
test(SVC_model)

print('MLPClassifier')
test(MLPClassifier_Model)



