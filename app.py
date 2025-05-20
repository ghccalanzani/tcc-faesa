# Bibliotecas
import os
import pandas as pd
import joblib
import shutil
from pdfminer.high_level import extract_text

PASTA_ORIGEM = "input"
PASTA_DESTINO_BASE = "output"
CATEGORIAS = {
    1: 'contratos',
    2: 'licitacoes',
    3: 'notasempenho',
    4: 'sancoes'
}

print("Carregando modelo e vetorizador...")
modelo = joblib.load('API-modeloML/modeloFinal.pkl')
vetorizadorTFIDF = joblib.load('API-modeloML/vetorizador.pkl')

# Extrair primeira pag de cada pdf
def extrair_primeira_pagina(caminho_arquivo):
    try:
        texto = extract_text(caminho_arquivo, page_numbers=[0])
        texto = texto.replace('\n', ' ')
        return texto
    except Exception as e:
        print(f"Erro ao processar o arquivo {caminho_arquivo}: {str(e)}")
        return None


# Classificar documento usando o modelo
def classificar_documento(texto):
    try:
        texto_tfidf = vetorizadorTFIDF.transform([texto])
        feature_names = vetorizadorTFIDF.get_feature_names_out()
        texto_df = pd.DataFrame(texto_tfidf.toarray(), columns=feature_names)
        resultado = modelo.predict(texto_df)[0]
        return resultado
    except Exception as e:
        print(f"Erro na classificação: {str(e)}")
        return None


# Criar diretórios de destino
def criar_diretorios_destino():
    for categoria in CATEGORIAS.values():
        pasta_destino = os.path.join(PASTA_DESTINO_BASE, categoria)
        os.makedirs(pasta_destino, exist_ok=True)
    print(f"Diretórios criados em: {PASTA_DESTINO_BASE}")


# Processar todos os PDFs da pasta de origem
def processar_documentos():
    # Validar pasta de origem (documentos não classificados)
    if not os.path.exists(PASTA_ORIGEM):
        print(f"Erro: A pasta com documentos para serem classificados não existe.")
        return

    criar_diretorios_destino()

    # Listar arquivos PDF na pasta de origem
    arquivos_pdf = [arquivo for arquivo in os.listdir(PASTA_ORIGEM)
                    if arquivo.lower().endswith('.pdf')]

    if not arquivos_pdf:
        print(f"Nenhum arquivo PDF foi encontrado.")
        return

    print(f"Encontrados {len(arquivos_pdf)} arquivos PDF para processar...")

    # Criar contador de arquivos para cada categoria
    contadores = {categoria: 0 for categoria in CATEGORIAS.values()}
    total_processados = 0
    total_erros = 0

    # Processar cada arquivo
    for arquivo in arquivos_pdf:
        # Extrair texto
        caminho_origem = os.path.join(PASTA_ORIGEM, arquivo)
        texto = extrair_primeira_pagina(caminho_origem)
        if not texto:
            print(f"Erro: Não foi possível extrair texto do arquivo")
            total_erros += 1
            continue

        # Classificar documento
        resultado = classificar_documento(texto)
        if resultado is None:
            print(f"Erro: Não foi possível classificar o documento")
            total_erros += 1
            continue

        # Obter categoria
        categoria = CATEGORIAS.get(resultado)
        if not categoria:
            print(f"Erro: Categoria inválida: {resultado}")
            total_erros += 1
            continue

        # Definir pasta de destino
        pasta_destino = os.path.join(PASTA_DESTINO_BASE, categoria)
        caminho_destino = os.path.join(pasta_destino, arquivo)

        # Copiar arquivo para pasta de destino
        try:
            shutil.copy2(caminho_origem, caminho_destino)
            contadores[categoria] += 1
            total_processados += 1
            print(f"[OK] Classificado como '{categoria}' → {caminho_destino}")
        except Exception as e:
            print(f"[ERRO] Erro ao copiar arquivo: {str(e)}")
            total_erros += 1

    # Exibir resumo
    print("\nRESUMO DO PROCESSAMENTO\n")
    print(f"Total de arquivos processados: {total_processados}")
    print(f"Total de erros: {total_erros}")
    print("\nArquivos por categoria:")
    for categoria, quantidade in contadores.items():
        print(f"- {categoria.capitalize()}: {quantidade}")

# Executar o processamento
if __name__ == "__main__":
    processar_documentos()