import os
import pandas as pd
import joblib
import shutil
import json
from pdfminer.high_level import extract_text

# Categorias do modelo
CATEGORIAS = {
    1: 'contratos',
    2: 'licitacoes',
    3: 'notasempenho',
    4: 'sancoes'
}

ARQUIVO_REGISTRO = "arquivos_processados.txt"

# Carregar modelo e vetorizador
print("Carregando modelo e vetorizador...")
modelo = joblib.load('API-modeloML/modeloFinal.pkl')
vetorizadorTFIDF = joblib.load('API-modeloML/vetorizador.pkl')


def extrair_primeira_pagina(caminho_arquivo):
    try:
        texto = extract_text(caminho_arquivo, page_numbers=[0])
        return texto.replace('\n', ' ')
    except Exception as e:
        print(f"Erro ao processar o arquivo {caminho_arquivo}: {str(e)}")
        return None


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


def criar_diretorios_destino(pasta_destino_base):
    for categoria in CATEGORIAS.values():
        pasta_destino = os.path.join(pasta_destino_base, categoria)
        os.makedirs(pasta_destino, exist_ok=True)


def carregar_registro_processados():
    if os.path.exists(ARQUIVO_REGISTRO):
        with open(ARQUIVO_REGISTRO, "r", encoding="utf-8") as f:
            return set(linha.strip() for linha in f.readlines())
    return set()


def salvar_registro_processado(nome_arquivo):
    with open(ARQUIVO_REGISTRO, "a", encoding="utf-8") as f:
        f.write(f"{nome_arquivo}\n")


def carregar_contadores(pasta_destino_base):
    contadores = {}

    for categoria in CATEGORIAS.values():
        pasta_categoria = os.path.join(pasta_destino_base, categoria)
        if not os.path.exists(pasta_categoria):
            os.makedirs(pasta_categoria)
            contadores[categoria] = 0
        else:
            # Contar quantos arquivos PDF existem nessa pasta
            arquivos = [arq for arq in os.listdir(pasta_categoria) if arq.lower().endswith(".pdf")]
            contadores[categoria] = len(arquivos)

    return contadores


def gerar_nome_arquivo(categoria, contadores):
    contadores[categoria] += 1
    nome_formatado = f"{categoria[:-1].capitalize()}-{contadores[categoria]:02d}.pdf"
    return nome_formatado



def processar_documentos(pasta_origem, pasta_destino_base):
    if not os.path.exists(pasta_origem):
        return {"erro": f"A pasta de origem não existe: {pasta_origem}"}

    criar_diretorios_destino(pasta_destino_base)
    processados = carregar_registro_processados()
    contadores = carregar_contadores(pasta_destino_base)

    arquivos_pdf = [arq for arq in os.listdir(pasta_origem) if arq.lower().endswith(".pdf")]
    if not arquivos_pdf:
        return {"erro": "Nenhum arquivo PDF encontrado."}

    resultado_final = {
        "processados": 0,
        "erros": 0,
        "por_categoria": {cat: 0 for cat in CATEGORIAS.values()}
    }

    for arquivo in arquivos_pdf:
        if arquivo in processados:
            print(f"[IGNORADO] Já processado: {arquivo}")
            continue

        caminho_origem = os.path.join(pasta_origem, arquivo)
        texto = extrair_primeira_pagina(caminho_origem)
        if not texto:
            resultado_final["erros"] += 1
            continue

        resultado = classificar_documento(texto)
        if resultado is None:
            resultado_final["erros"] += 1
            continue

        categoria = CATEGORIAS.get(resultado)
        if not categoria:
            resultado_final["erros"] += 1
            continue

        pasta_destino = os.path.join(pasta_destino_base, categoria)
        novo_nome = gerar_nome_arquivo(categoria, contadores)
        caminho_destino = os.path.join(pasta_destino, novo_nome)

        try:
            shutil.copy2(caminho_origem, caminho_destino)
            salvar_registro_processado(arquivo)
            resultado_final["por_categoria"][categoria] += 1
            resultado_final["processados"] += 1
            print(f"[OK] {arquivo} → {novo_nome}")
        except Exception as e:
            print(f"[ERRO] Erro ao copiar {arquivo}: {str(e)}")
            resultado_final["erros"] += 1

    return resultado_final


# Execução local
if __name__ == "__main__":
    resultado = processar_documentos("input", "output")
    print("\nResumo:")
    print(resultado)
