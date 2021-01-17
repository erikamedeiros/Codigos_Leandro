import pandas as pd
from sklearn.model_selection import train_test_split
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


for i in range(1,30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)
    resultado = cross_val_score(modelo,x,y, cv = kfold, n_jobs=-1)
    media = resultado.mean()
    print(media)
    
    
