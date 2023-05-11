# Hello World - Classificação 5 - CLASSIFICACAÇÃO DE USUÁRIOS BUSCA NA WEB

import csv
import pandas as pd
from sklearn.naive_bayes import MultinomialNB


'''
def carregar_buscas():
           
    X = []
    Y = []
    
    arquivo = open('/home/franciscofoz/Documents/GitHub/machine-learning-training/Datasets/busca.csv','r')
    leitor = csv.reader(arquivo)
    
    next(leitor)
    
    for home, busca, logado, comprou in leitor:
        
        dado = [int(home), busca, int(logado)]
        X.append(dado)
        Y.append([int(comprou)])
        
    return X, Y
'''

df = pd.read_csv('/home/franciscofoz/Documents/GitHub/machine-learning-training/Datasets/busca.csv')

Y_df = df['comprou']
X_df = df[['home','busca','logado']]

Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df

X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_de_treino = 0.9

tamanho_de_treino = int(porcentagem_de_treino * len(X))
tamanho_de_teste = len(Y) - tamanho_de_treino

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]


modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d==0]

total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)