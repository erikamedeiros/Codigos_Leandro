import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

# Classificadores
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# KFold e Metricas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

np.random.seed(0)

arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

y = arq["Y"]
x = arq.drop('Y', axis=1)

#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
x = normalizador.fit_transform(x)



svc_modelo = SVC(C=1, kernel='poly', degree=4, gamma='scale', random_state=0)

mlp_modelo = MLPClassifier(verbose = False, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(500), activation = 'relu',
                              batch_size=69, learning_rate_init=0.001, random_state=0)

knn_modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 1)



npr = np.random.random_integers(1,2000, 30)
len(np.unique(npr))

knn_acuracia_media = []
svc_acuracia_media = []
mlp_acuracia_media = []

for i in npr:
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)  
    
    svc_resultado = cross_val_score(svc_modelo,x,y, cv = kfold, n_jobs=-1)
    svc_media = svc_resultado.mean()
    svc_acuracia_media.append(svc_media)
    
    knn_resultado = cross_val_score(knn_modelo,x,y, cv = kfold, n_jobs=-1)
    knn_media = knn_resultado.mean()
    knn_acuracia_media.append(knn_media)  

    mlp_resultado = cross_val_score(knn_modelo,x,y, cv = kfold, n_jobs=-1)
    mlp_media = mlp_resultado.mean()
    mlp_acuracia_media.append(mlp_media) 

print('Acuracia Final do SVM: ', np.array(svc_acuracia_media).mean())
print('Acuracia Final do KNN: ', np.array(knn_acuracia_media).mean())
print('Acuracia Final do MLPClassifier: ', np.array(knn_acuracia_media).mean())