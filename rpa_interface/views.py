# classificador/views.py

from django.shortcuts import render
from .forms import PastaForm
from app import processar_documentos  # Certifique-se que app.py está no mesmo nível do manage.py ou em PYTHONPATH


def classificar_documentos_view(request):
    resultado = None
    erro = None

    if request.method == 'POST':
        form = PastaForm(request.POST)
        if form.is_valid():
            pasta_origem = form.cleaned_data['pasta_origem']
            pasta_destino = form.cleaned_data['pasta_destino']

            try:
                resultado = processar_documentos(pasta_origem, pasta_destino)
                if "erro" in resultado:
                    erro = resultado["erro"]
                    resultado = None
            except Exception as e:
                erro = f"Ocorreu um erro durante o processamento: {str(e)}"
    else:
        form = PastaForm()

    return render(request, '../templates/index.html', {
        'form': form,
        'resultado': resultado,
        'erro': erro
    })
