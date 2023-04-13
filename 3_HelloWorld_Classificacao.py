# Hello World - Classificação 3 - PORCO OU CACHORRO?


from sklearn.naive_bayes import MultinomialNB

# Dados
    #é gordinho?
    #tem perna curta?
    #faz "au au"?
porco1 = [1,1,0]
porco2 = [1,1,0]
porco3 = [1,1,0]

cachorro1 = [1,1,1]
cachorro2 = [0,1,1]
cachorro3 = [0,1,1]

dados = [porco1,porco2,porco3,cachorro1,cachorro2,cachorro3]

marcacoes = [1,1,1,-1,-1,-1]
misterioso1 = [1,1,1]
misterioso2 = [1,0,0]
misterioso3 = [0,0,1]
teste = [misterioso1,misterioso2,misterioso3]
marcacoes_teste = [-1,1,-2]


# Modelo
modelo = MultinomialNB()
modelo.fit(dados,marcacoes)
resultado = modelo.predict(teste)

diferencas = resultado - marcacoes_teste
acertos = [d for d in diferencas if d==0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste)
taxa_de_acerto = (total_de_acertos / total_de_elementos) * 100


print('Resultado:',resultado)
print('Taxa de acerto', taxa_de_acerto)
