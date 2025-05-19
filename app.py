import os
import fitz  # PyMuPDF
import joblib
import shutil

# Caminhos principais
BASE_DIR = "documentos-juridicos"
PASTA_ORIGEM = os.path.join(BASE_DIR, "downloads")

PASTAS_DESTINO = {
    1: "contratos", #Original 126
    2: "licitacoes",#Original 176
    3: "notas-empenho",#Original 06-77
    4: "sancoes"#Original 29
}

NOMES_ARQUIVO = {
    1: "Contrato",
    2: "Licitacao",
    3: "notaEmpenho",
    4: "Sancao"
}

# Carrega modelo e vetorizador
modelo = joblib.load("API-modeloML/modeloFinal.pkl")
vetorizador = joblib.load("API-modeloML/vetorizador.pkl")

# Função para extrair texto do PDF
def extrair_texto_pdf(caminho_pdf):
    texto = ""
    with fitz.open(caminho_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto.strip()

# Garante existência das pastas de destino
for pasta in PASTAS_DESTINO.values():
    os.makedirs(os.path.join(BASE_DIR, pasta), exist_ok=True)

# Verifica se pasta de origem existe
if not os.path.exists(PASTA_ORIGEM):
    print(f"[ERRO] Pasta de origem '{PASTA_ORIGEM}' não encontrada.")
    exit()

# Processa cada PDF da pasta 'downloads'
for arquivo in os.listdir(PASTA_ORIGEM):
    if arquivo.lower().endswith(".pdf"):
        caminho_arquivo = os.path.join(PASTA_ORIGEM, arquivo)
        print(f"\n📄 Lendo: {caminho_arquivo}")

        texto = extrair_texto_pdf(caminho_arquivo)

        if not texto:
            print("  [!] PDF vazio ou ilegível. Ignorado.")
            continue

        # Vetoriza e classifica
        texto_vetorizado = vetorizador.transform([texto])
        resultado = modelo.predict(texto_vetorizado)[0]

        pasta_destino = os.path.join(BASE_DIR, PASTAS_DESTINO[resultado])
        nome_base = NOMES_ARQUIVO[resultado]

        # Conta arquivos com mesmo prefixo
        existentes = [f for f in os.listdir(pasta_destino) if f.startswith(nome_base) and f.endswith(".pdf")]
        numero = len(existentes) + 1
        novo_nome = f"{nome_base}-{numero:02d}.pdf"
        caminho_novo = os.path.join(pasta_destino, novo_nome)

        # Copia o arquivo (não move)
        shutil.copy2(caminho_arquivo, caminho_novo)
        print(f"  [✓] Classificado como '{nome_base}', copiado para: {caminho_novo}")