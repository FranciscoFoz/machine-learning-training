# Hello World - Classificação 4 - CLASSIFICACAÇÃO DE USUÁRIOS WEB

import requests
import csv
from sklearn.naive_bayes import MultinomialNB 

url_csv = 'https://raw.githubusercontent.com/alura-cursos/machine-learning-introducao-a-classificacao/master/acesso.csv'

def carregar_acessos(url):
    
    dados_raw = requests.get(url).content.decode('utf-8')
    
    X = []
    Y = []
    
    leitor = csv.reader(dados_raw.splitlines())
    next(leitor)
    for home, como_funciona, contato,comprou in leitor:
        
        dado = [int(home), int(como_funciona),int(contato)]
        X.append(dado)
        Y.append([int(comprou)])
        
    return X, Y
        
X, Y = carregar_acessos(url_csv)

treino_dados = X[:90]
treino_marcacoes = Y[:90]

teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)
diferencas = resultado - teste_marcacoes

acertos = (diferencas == 0).all(axis=1)

total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)



