# VALIDACAO_MODELO

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Importando dados
uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
dados = 

print(dados)

#Definindo x e y
x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

SEED = 1

# Dividindo em treino e teste
np.random.seed(SEED)
treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size = 0.25,stratify = y)

print(5*'+-','INÍCIO DO PROGRAMA',5*'-+')
quantidade_de_tracos = 80

print("\nTreinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))
print(quantidade_de_tracos*'-')

# DUMMY CLASSIFIER - Baseline
dummy_stratified = DummyClassifier(strategy='stratified')
dummy_stratified.fit(treino_x, treino_y)
acuracia_dummy = dummy_stratified.score(teste_x, teste_y) * 100

print(5*'+-','DUMMY CLASSIFIER',5*'-+')
print("A acurácia foi de %.2f%%" % acuracia_dummy)
print(quantidade_de_tracos*'-')

# ÁRVORE DE DECISÃO
dtc = DecisionTreeClassifier(max_depth = 2)
dtc.fit(treino_x, treino_y)
acuracia_dtc = dtc.score(teste_x, teste_y) * 100

print(5*'+-','DECISION TREE CLASSIFIER CLASSIFIER',5*'-+')
print("A acurácia foi de %.2f%%" % acuracia_dtc)
print(quantidade_de_tracos*'-')

# Validação cruzada
'''
dtc = DecisionTreeClassifier(max_depth = 2)
results = cross_validate(dtc,x,y,cv=5,return_train_score=False)
media_resultados = results['test_score'].mean()
desvio_padrao_resultados = results['test_score'].std()

acuracia_results_min = round((media_resultados - (2 * desvio_padrao_resultados)) * 100,2)
acuracia_results_max = round((media_resultados + (2 * desvio_padrao_resultados)) * 100,2)


print(f'Intervalo da acuracia do Decision Tree Classifier\ncom validação cruzada: [{acuracia_results_min},{acuracia_results_max}]')
print(20*'-')
'''


# Aleatoriedade com validação cruzada

def imprime_resultados(results):
    media_resultados = results['test_score'].mean().round(2)
    desvio_padrao_resultados = results['test_score'].std()
    acuracia_results_min = round((media_resultados - (2 * desvio_padrao_resultados)) * 100,2)
    acuracia_results_max = round((media_resultados + (2 * desvio_padrao_resultados)) * 100,2)
    variacao_do_intervalo = round(acuracia_results_max-acuracia_results_min,2)
    
    print(f'Média: {media_resultados * 100}')
    print(f'Intervalo da acuracia:\n\tMin: {acuracia_results_min}\n\tMax: {acuracia_results_max}')
    print(f'Variação do intervalo = {variacao_do_intervalo}')

print(5*'+-','DECISION TREE CLASSIFIER COM VALIDAÇÃO',5*'-+')

cv = KFold(n_splits = 5,shuffle= True)
dtc = DecisionTreeClassifier(max_depth = 2)
results = cross_validate(dtc,x,y,cv=cv,return_train_score=False)
imprime_resultados(results)

print(quantidade_de_tracos*'-')

# Aleatoriedade com validação cruzada estratificada

print(5*'+-','DECISION TREE CLASSIFIER COM VALIDAÇÃO ESTRATIFICADA',5*'-+')

scv = StratifiedKFold(n_splits = 5,shuffle= True)
dtc = DecisionTreeClassifier(max_depth = 2)
results = cross_validate(dtc,x,y,cv=scv,return_train_score=False)
imprime_resultados(results)

print(quantidade_de_tracos*'-')