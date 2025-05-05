######## Exemplo de uso #########
import pandas as pd
import joblib

modelo = joblib.load(r'../API-modeloML/modeloFinal.pkl')
vetorizadorTFIDF = joblib.load(r'../API-modeloML/vetorizador.pkl')

texto = "texto qualquer licitação licitacao licitacao"
texto_tfidf = vetorizadorTFIDF.transform([texto])
feature_names = vetorizadorTFIDF.get_feature_names_out()
texto_df = pd.DataFrame(texto_tfidf.toarray(), columns=feature_names)

resultado = modelo.predict(texto_df)
print(resultado)
# 'contrato': 1
# 'licitacao': 2
# 'notaempenho': 3
# 'sancao': 4