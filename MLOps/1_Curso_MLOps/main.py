from flask import Flask
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('https://raw.githubusercontent.com/alura-cursos/1576-mlops-machine-learning/aula-5/casas.csv')
df_tamanho = df[['tamanho','preco']]

X = df_tamanho.drop('preco',axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
modelo = LinearRegression()
modelo.fit(X_train,y_train)


app = Flask('__name__')

@app.route('/')
def home():
    return "Minha primeira API.<p>Hello, World!</p>" 


@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br',to='en')
    polaridade = tb_en.sentiment.polarity
    return f"polaridade: {polaridade}"


@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    preco = modelo.predict([[tamanho]])
    return str(preco)

app.run(debug=True)
