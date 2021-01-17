import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


arq = pd.read_csv('dataset.csv', sep=';')  
del arq["imgName"]

y = arq["Y"]
x = arq.drop('Y', axis=1)

#normalizando as variaveis
normalizador = MinMaxScaler(feature_range = (0, 1))
x = normalizador.fit_transform(x)

#print(results)

#modelo = SVC(C=1, kernel='poly', degree=4, gamma='scale', random_state=0)
modelo = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 1)

np.random.seed(0)
npr = np.random.rand(30)

for j in npr:
    kfold = KFold(n_splits=10, shuffle=True, random_state=int(j*1000))  
    resultado = cross_val_score(modelo,x,y, cv = kfold, n_jobs=-1)
    media = resultado.mean()
    print(media)
    
    
