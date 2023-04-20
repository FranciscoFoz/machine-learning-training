# Hello World - Classificação 4 - CLASSIFICACAÇÃO DE USUÁRIOS WEB

import requests
import csv 

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






