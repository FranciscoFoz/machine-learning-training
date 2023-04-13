# Dataset de Churn de Biblioteca

import numpy as np
import pandas as pd

tipo_usuario = ['Graduação', 'Pós-graduação','Docência']
proporcoes = [0.45, 0.45, 0.1] #90% de alunos e 10% de professores

institutos = ['Artes','Biologia','Computação','Economia','Estudos da Linguagem','Filosofia e Ciências Humanas','Física',
              'Geociências','Matemática, Estatística e Computação Científica','Química']

quantidade_total_usuarios = 18000
tamanho_dados_uns = round(0.95 * quantidade_total_usuarios)
tamanho_dados_zeros = round(0.05 * quantidade_total_usuarios)

def notas_distribuicao_normal(media,desvio_padrao,tamanho_dados):
    
    distribuicao_normal = np.random.normal(media,desvio_padrao, tamanho_dados).round()
    distribuicao_normal = np.clip(distribuicao_normal,0,5)
    return distribuicao_normal

data = {
    'nivel_matricula': np.random.choice(tipo_usuario, size=tamanho_dados_uns,p=proporcoes),
    'institutos': np.random.choice(institutos, size=tamanho_dados_uns),
    
    'visitou_biblioteca': np.random.randint(2, size=tamanho_dados_uns),
    'fez_emprestimo': np.random.randint(2, size=tamanho_dados_uns),
    'visitou_biblioteca_digital': np.random.randint(2, size=tamanho_dados_uns),
    'visitou_evento': np.random.randint(2, size=tamanho_dados_uns),
    'consultou_biblioteca': np.random.randint(2, size=tamanho_dados_uns),
    
    'nota_emprestimo': notas_distribuicao_normal(3,1.5,tamanho_dados_uns),
    'nota_evento': notas_distribuicao_normal(2.5,1,tamanho_dados_uns),
    'nota_infraestrutura': notas_distribuicao_normal(3.3,1.5,tamanho_dados_uns),
    'nota_acervo': notas_distribuicao_normal(4,1.5,tamanho_dados_uns),
    'nota_redes_sociais': notas_distribuicao_normal(2,2,tamanho_dados_uns)
}

data_zeros = {
    'nivel_matricula': np.random.choice(tipo_usuario, size=tamanho_dados_zeros,p=proporcoes),
    'institutos': np.random.choice(institutos, size=tamanho_dados_zeros),
    
    'visitou_biblioteca': np.zeros(tamanho_dados_zeros, dtype=int),
    'fez_emprestimo': np.zeros(tamanho_dados_zeros, dtype=int),
    'visitou_biblioteca_digital': np.zeros(tamanho_dados_zeros, dtype=int),
    'visitou_evento': np.zeros(tamanho_dados_zeros, dtype=int),
    'consultou_biblioteca': np.zeros(tamanho_dados_zeros, dtype=int),
    
    'nota_emprestimo': notas_distribuicao_normal(2.5,1.5,tamanho_dados_zeros),
    'nota_evento': notas_distribuicao_normal(2,1,tamanho_dados_zeros),
    'nota_infraestrutura': notas_distribuicao_normal(3,1.5,tamanho_dados_zeros),
    'nota_acervo': notas_distribuicao_normal(3.5,1.5,tamanho_dados_zeros),
    'nota_redes_sociais': notas_distribuicao_normal(1.5,2,tamanho_dados_zeros)
}

df_zeros = pd.DataFrame(data_zeros)
df_uns = pd.DataFrame(data)

df = pd.concat([df_zeros,df_uns])

# Marca a coluna de churn
df['churn'] = np.where((df['visitou_biblioteca']==0) & 
                       (df['fez_emprestimo']==0) & 
                       (df['visitou_biblioteca_digital']==0) & 
                       (df['visitou_evento']==0) & 
                       (df['consultou_biblioteca']==0), 'sim', 'não')

df.to_csv('Datasets/churn_biblioteca.csv',sep=';',index=False)
