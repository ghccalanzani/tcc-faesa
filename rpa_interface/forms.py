from django import forms

class PastaForm(forms.Form):
    pasta_origem = forms.CharField(label='Pasta de Origem', max_length=300, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Digite o caminho da pasta com os PDFs'
    }))
    pasta_destino = forms.CharField(label='Pasta de Destino', max_length=300, widget=forms.TextInput(attrs={
        'class': 'form-control',
        'placeholder': 'Digite o caminho da pasta onde salvar os arquivos classificados'
    }))
