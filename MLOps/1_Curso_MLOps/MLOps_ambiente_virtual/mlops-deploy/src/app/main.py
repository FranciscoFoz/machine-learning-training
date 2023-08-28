from flask import Flask,request, jsonify
from flask_basicauth import BasicAuth 
from textblob import TextBlob
import pickle
from sklearn.linear_model import LinearRegression
import os

colunas = ['tamanho','ano','garagem']
#modelo = pickle.load(open('../../models/modelo.sav','rb'))
modelo = pickle.load(open('MLOps/1_Curso_MLOps/MLOps_ambiente_virtual/mlops-deploy/models/modelo.sav','rb'))


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API.<p>Hello, World!</p>" 


@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br',to='en')
    polaridade = tb_en.sentiment.polarity
    return f"polaridade: {polaridade}"


@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
    

