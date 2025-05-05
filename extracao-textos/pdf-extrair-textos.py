#Bibliotecas
import os
import pandas as pd
import csv
from pdfminer.high_level import extract_text

#Extrair texto da primeira pag do pdf
def extrair_primeira_pagina(caminho_arquivo):
    try:
        texto = extract_text(caminho_arquivo, page_numbers=[0])
        texto = texto.replace('\n', ' ')
        return texto
    except Exception as e:
        return f"Erro ao processar o arquivo: {str(e)}"

# Processar os PDFs do dir e adicionar ao df
def processar_diretorio(diretorio, categoria):
    dados = []

    # Validar diretorio
    if not os.path.exists(diretorio):
        print(f"O diretorio {diretorio} não existe.")
        return dados

    # Listar os PDFs no diretorio
    arquivos_pdf = [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio)
                    if arquivo.lower().endswith('.pdf')]

    # Processar cada PDF
    for arquivo in arquivos_pdf:
        texto = extrair_primeira_pagina(arquivo)
        dados.append({'texto': texto, 'categoria': categoria})
        print(f"Processado: {os.path.basename(arquivo)}")

    return dados

# Processar os diretórios
dados_contratos = processar_diretorio("../documentos-juridicos/contratos", 'contrato')
dados_licitacoes = processar_diretorio("../documentos-juridicos/licitacoes", 'licitacao')
dados_notasempenho = processar_diretorio("../documentos-juridicos/notasempenho", 'notaempenho')
dados_sancoes = processar_diretorio("../documentos-juridicos/sancoes", 'sancao')

# Combinar os dados
todos_dados = dados_contratos + dados_licitacoes + dados_notasempenho + dados_sancoes

# Criar Dataframe
df = pd.DataFrame(todos_dados)

# Salvar no CSV
df.to_csv('./documentos-classificados.csv', index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')

print(f"Arquivo CSV criado")
print(f"Total de documentos processados: {len(todos_dados)}")
print(f"- Contratos: {len(dados_contratos)}")
print(f"- Licitações: {len(dados_licitacoes)}")
print(f"- Notas de Empenho: {len(dados_notasempenho)}")
print(f"- Sanções: {len(dados_sancoes)}")