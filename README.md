# BART-large-CNN

# Introdução

O modelo BART (Bidirectional and Auto-Regressive Transformers), na sua variante BART-large-CNN, é um poderoso modelo de machine learning desenvolvido pela Meta AI (antiga Facebook AI). Ele é baseado na arquitetura Transformer e foi projetado para tarefas de processamento de linguagem natural, com destaque para resumo de texto.

O diferencial do BART é sua capacidade de combinar os métodos de codificação bidirecional (como no BERT) e geração de texto auto-regressiva (como no GPT), tornando-o altamente eficaz em tarefas que exigem compressão semântica e geração textual. A variante large do BART foi treinada com um grande volume de dados e configurações avançadas, o que amplia sua precisão e flexibilidade.

O BART-large-CNN foi ajustado especificamente para o resumo de textos longos, utilizando o conjunto de dados CNN/Daily Mail. Essa especialização o torna adequado para aplicações como sumarização de artigos de notícias, relatórios, e outros textos extensos. Ele é amplamente utilizado por pesquisadores e profissionais que buscam soluções robustas em Natural Language Processing (NLP).

O modelo pode ser acessado e implementado diretamente pela biblioteca Transformers, da Hugging Face, facilitando sua integração em diferentes projetos.

# Motivação do BART-large-CNN:

A motivação para desenvolver o BART foi criar um modelo versátil que combinasse o melhor dos dois mundos em arquiteturas de processamento de linguagem natural:

- **Codificação bidirecional:** A abordagem usada no BERT é altamente eficaz para entender o contexto de palavras em ambas as direções dentro de uma sentença. Isso é crucial para tarefas como análise de sentimento e classificação de texto.
- **Geração auto-regressiva:** Modelos como o GPT se destacam na criação de texto fluido, aproveitando informações aprendidas em sequência.
O BART unifica essas duas abordagens para superar as limitações de modelos exclusivamente de codificação ou de geração. Ele foi projetado com o intuito de lidar com uma variedade de tarefas, desde tradução e preenchimento de lacunas até resumo de texto.

Para o BART-large-CNN, a motivação específica foi criar um modelo de resumo de texto eficiente que pudesse lidar com entradas longas, como artigos jornalísticos, mantendo a precisão semântica e a clareza.

# Principais características:

- **Versatilidade:** Embora especializado em sumarização, pode ser adaptado para outras tarefas de NLP, como tradução e geração de texto.
- **Pré-treinamento robusto:** Utiliza estratégias como máscaras de palavras e reorganização de sentenças para melhor compreensão e geração.
- **Alto desempenho:** Lida com entradas longas e complexas, preservando contexto e consistência.
- **Integração fácil:** Disponível na biblioteca Hugging Face Transformers, facilitando seu uso em diversos projetos.

# Funcionamento

O funcionamento do BART-large-CNN combina etapas de pré-treinamento e ajuste fino, utilizando uma arquitetura Transformer com codificador e decodificador.

1. **Arquitetura Transformer Híbrida**

   O BART adota uma abordagem clássica de Transformer com dois componentes principais, codificador e decodificador.
   
2. **Pré-treinamento**

   O modelo é treinado inicialmente para reconstruir texto original a partir de versões distorcidas. Isso melhora sua capacidade de entender e gerar texto.
   
3. **Ajuste Fino (Fine-Tuning)**

   Após o pré-treinamento, o modelo é ajustado para tarefas específicas. No caso do BART-large-CNN, ele foi ajustado com o conjunto de dados CNN/Daily Mail, que contém    textos longos (artigos jornalísticos) e seus respectivos resumos.
   
5. **Processo de Inferência**
  
6. **Mecanismo de Atenção (Attention)**
   
   O modelo usa mecanismos de atenção para identificar as partes mais importantes do texto de entrada. Isso é crucial para determinar quais informações devem ser 
   destacadas no resumo e quais podem ser omitidas.

# Casos de Uso do BART-large-CNN:
- **Jornalismo e Mídia:** Resumo de notícias e artigos para facilitar o consumo de informações.
- **Educação:** Condensação de documentos científicos, relatórios e textos acadêmicos.
- **Negócios:** Geração de resumos de relatórios e análises para executivos.
- **Assistentes Virtuais:** Fornecimento de resumos de textos longos em interações com usuários.
- **Marketing:** Criação de textos concisos para redes sociais ou blogs.
- **Jurídico:** Resumo de contratos e documentos legais para maior acessibilidade.

# Limitações do BART-large-CNN (Resumo):
- **Dependência de dados:** Desempenho reduzido fora do domínio de notícias.
- **Resumos genéricos:** Pode gerar textos superficiais ou com lacunas de informação.
- **Limite de entrada:** Incapaz de processar textos muito longos sem segmentação.
- **Inconsistências:** Erros contextuais ou inclusão de informações irrelevantes.
- **Necessidade de ajuste:** Requer treinamento adicional para novos domínios.
- **Demanda computacional:** Exige hardware potente, como GPUs, para execução eficiente.

# Como utilizar

1. **Instalar Dependências**

Certifique-se de ter o Python instalado e configure a biblioteca Transformers:
```bash
pip install transformers
```

Se for usar processamento de texto mais eficiente, instale também o PyTorch ou TensorFlow (escolha uma):
```bash
pip install torch   # Para PyTorch
```
ou
```bash
pip install tensorflow  # Para TensorFlow
```

2. **Carregar o Modelo**

Carregue o modelo e o tokenizer para processar o texto de entrada:
```bash
from transformers import BartForConditionalGeneration, BartTokenizer

# Carregar o modelo BART-large-CNN
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
```

3. **Preparar o Texto de Entrada**

Codifique o texto a ser resumido:
```bash
# Texto original (entrada)
input_text = """
    Seu artigo ou texto longo aqui. O modelo irá processar e gerar um resumo baseado nesse conteúdo.
"""

# Tokenizar o texto
inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
```

4. **Gerar o Resumo**

Use o modelo para gerar o texto resumido:
```bash
# Gerar resumo
summary_ids = model.generate(inputs, max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

# Decodificar e exibir o resumo
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Resumo:\n", summary)
```

### **Requisitos de Hardware**

O BART-large-CNN pode ser executado em CPU, mas GPUs são recomendadas para acelerar o processamento, especialmente com textos longos.

# Problemas comuns e soluções
- **Erro de Memória (Out of Memory / OOM)**

  Reduzir o Tamanho de Entrada: Utilize o parâmetro max_length ao tokenizar os textos para limitar o tamanho da entrada. Além disso, considere dividir o texto em segmentos menores.
```bash
  inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
```
- **Texto Cortado ou Truncado**

  Divida o texto em partes menores e resuma cada uma separadamente. Depois, combine os resumos em um único documento, se necessário.
```bash
input_text_parts = [input_text[i:i+1024] for i in range(0, len(input_text), 1024)]
summaries = [model.generate(tokenizer.encode(part, return_tensors="pt")) for part in input_text_parts]
```
- **Resumos Superficiais ou Irrelevantes**

  Utilize parâmetros como num_beams e length_penalty para melhorar a qualidade do resumo.
```bash
summary_ids = model.generate(inputs, num_beams=4, length_penalty=2.0, early_stopping=True)
```
- **Saída com Erros de Codificação ou Caracteres Especiais**

  Antes de passar o texto para o modelo, remova ou normalize caracteres especiais e símbolos indesejados.
```bash
input_text = input_text.replace("\n", " ").replace("\t", " ")
```
- **Desempenho Lento (Baixa Velocidade de Execução)**

  Processar múltiplos textos em paralelo (batch inference) pode aumentar a eficiência. Isso pode ser feito passando vários textos para o modelo em um único batch.
```bash
batch_input = tokenizer([text1, text2, text3], return_tensors="pt", padding=True, truncation=True)
batch_summary = model.generate(batch_input['input_ids'])
```
- **Erro ao Carregar o Modelo ("Model Not Found")**

1. Verificar Conexão de Internet: Certifique-se de que a máquina tenha acesso à internet para baixar os arquivos do modelo.
2. Verificar Nome do Modelo: Confirme se o nome do modelo está correto e se ele está disponível na Hugging Face.
```bash
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
```
- **Resultados de Geração de Texto Não Natural**

  Use parâmetros como temperature, top_k, ou top_p para melhorar a qualidade e a naturalidade da geração.
```bash
summary_ids = model.generate(inputs, temperature=0.7, top_k=50, top_p=0.95)
```

# Licença

O modelo BART-large-CNN é disponibilizado sob a Licença Apache 2.0 pela Hugging Face. A Licença Apache 2.0 é uma licença de código aberto permissiva, que permite que você use, modifique e distribua o modelo, com algumas condições.
