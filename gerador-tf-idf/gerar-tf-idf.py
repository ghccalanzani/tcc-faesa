import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

stop_words_portugues = ['o', 'a', 'e', 'é', 'os', 'as', 'que', 'do', 'da', 'de', 'dos', 'das']

# Carregar o arquivo CSV com os textos e classificacoes
def processar_documentos_juridicos(arquivo_entrada):
    # Carregar o arquivo CSV
    print(f"Carregando o arquivo: {arquivo_entrada}")
    df = pd.read_csv(arquivo_entrada)

    # Separando textos e classificacoes
    textos = df.iloc[:, 0].tolist()
    classificacoes = df.iloc[:, 1].tolist()

    print(f"Total de documentos carregados: {len(textos)}")
    print(f"Distribuição de classificações: {pd.Series(classificacoes).value_counts().to_dict()}")

    # Criar o vetorizador TF-IDF
    print("Gerando matriz TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b[a-zA-ZÀ-ÖØ-öø-ÿ]{1,}\b',
                                       stop_words=stop_words_portugues)

    # Obter a matriz TF-IDF
    tfidf_matriz = tfidf_vectorizer.fit_transform(textos)

    # Converter para DataFrame
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matriz.toarray(), columns=tfidf_feature_names)

    # Salvar vetorizador com Joblib
    print("Salvando o vetorizador...")
    joblib.dump(tfidf_vectorizer, '../API-modeloML/vetorizador.pkl')

    # Salvar o arquivo TF-IDF
    tfidf_df.to_csv('./tfidf.csv', index=False)
    return 'Arquivo TF-IDF gerado com sucesso!'


if __name__ == "__main__":
    caminho_entrada = '../extracao-textos/documentos-classificados.csv'
    print(processar_documentos_juridicos(caminho_entrada))
    print("Processamento concluído!")