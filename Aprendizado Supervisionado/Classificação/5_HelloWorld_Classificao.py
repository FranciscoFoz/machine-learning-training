# Hello World - Classificação 5 - CLASSIFICACAÇÃO DE USUÁRIOS BUSCA NA WEB

import csv


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
        
X, Y = carregar_buscas()

print(X)
print(Y)
