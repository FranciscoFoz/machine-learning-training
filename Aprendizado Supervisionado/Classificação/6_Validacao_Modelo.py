# VALIDACAO_MODELO

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Importando dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

#Definindo x e y
x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]


# Dividindo em treino e teste
SEED = 564164
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,stratify = y)

print("\nTreinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
print(20*'-')

# DUMMY CLASSIFIER - Baseline
dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(treino_x, treino_y)
acuracia = dummy_stratified.score(teste_x, teste_y) * 100

print("\nA acurácia do dummy stratified foi de %.2f%%" % acuracia)
print(20*'-')