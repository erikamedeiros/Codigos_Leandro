# Checagem da acurácia em 30 amostragens com kfold = 10
# Seed fixo: np.random.seed(0)
# 30 valores distintos pseudo-aleatórios a partir do Seed Fixo. Chequei se eram distintos, para garantir as 30 iterações.
# Laço "for" com random_state recebendo os 30 valores e calculando o kfold
# todos os modelos partem do mesmo Seed no kfold, de modo a serem comparavéis.
# A cada iteração, a média de 10 acurácias retornadas do cross_val_score é feita
# Essas 30 médias de cada classificador, tem suas médias finais exibidas (teste estatisticos por fazer)

# Os classificadores implementados são KNN, SVM e MLPClassifier
