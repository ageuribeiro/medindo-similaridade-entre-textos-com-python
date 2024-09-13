# Treinamento básico que abordar a similaridade dos dados - NLP

## Medida de Similaridade entre textos - Containment

<hr>

Neste notebook, iremos implementar uma função containment. A tal função irá comparar dois textos e analisar a similaridade dos mesmos com relação aos seus n-gramas de interação. Primeiramente iremos entender o conceito de vocabulário, n-gramas para posteriormente implementar a função.

<hr>

## Contar N-Grama

* É uma sequência de n-elementos dentro de uma frase:
    * Palavras
    * Letras
    * Símbolos
    * Classificação gramatical
    * etc

Primeiramente temos que contar as ocorrências de n-gramas dos nosso textos. Usaremos o CountVectorizer para converter o conjuntos de dados
de textos em uma matriz de contagem.

No código abaixo, podemos variar o valor de n e utilizar o CountVectorizer para contat as ocorrências de n-gramas. Podemos notar que na célula abaixo estamos criando um vocabulário através da utilização do ConuntVectorizer e, posteriormente iremos analizae a matriz de contagem.

```bash
import numpy as np
import sklearn
```

# Unigrama

A execução do exemplo imprime o vocabulário. Podemos ver que existem 8 palavras no vocabulário e, portanto, vetores codificados também possuem um comprimento de 8.

```bash
from sklearn.feature_extraction.text import CountVectorizer

texto_a_ser_comparado = "Suponha que esse seja o texto que desejo comparar"
texto_fonte = "Suponha que esse seja o texto principal"

# Número de n-gramas
n = 1

# Instanciando o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# Cria um dicionario de n-gramas
vocab2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# Printar dicionário de palavras: index
print(vocab2int)
```

### Saída do Unigrama
```bash
{'suponha': 6, 'que': 4, 'esse': 2, 'seja': 5, 'texto': 7, 'desejo': 1, 'comparar': 0, 'principal': 3}
```

# Bigrama

O mesmo vale para o caso de dec bigramas. Temos 8 bigramas no vocabulário e, os vetores são codificados com comprimento.

```bash
# Número de n_gramas
n = 2

# Instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# Cria um dicionário de n-gramas
vocab2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# printar o dicionario de palavras index
print(vocab2int)
```
### Saída do Bigrama
```bash
{'suponha que': 5, 'que esse': 3, 'esse seja': 1, 'seja texto': 4, 'texto que': 7, 'que desejo': 2, 'desejo comparar': 0, 'texto principal': 6}
```


# Trigrama

```bash
# Número de n_gramas
n = 3

# Instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# Cria um dicionário de n-gramas
vocab2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

# printar o dicionario de palavras index
print(vocab2int)
```
### Saída do Trigrama
```bash
{'suponha que esse': 5, 'que esse seja': 2, 'esse seja texto': 0, 'seja texto que': 4, 'texto que desejo': 6, 'que desejo comparar': 1, 'seja texto principal': 3}
```

## As palavras do vocabulário

Note que o artigo "o" das frases <span style="background-color: silver; color: black; ">texto_a_ser_comparado</span> e <span style="color: red;">texto_fonte</span> não aparece no vocabulários. Note que todas as frases encontram-se em minúsculos. Isso ocorre devido ao fato de que quando passamos o parâmetro analyzer = "word", estamos considerando em nossa análise palavras com dois ou mais caracteres e consequetemente ignorando as palavras com apenas um caraterer. Excluir esses caracteres (artigos) é um comportamento padrão e desejado por muitas análises de texto devido a sua irrelevancia, em grande parte das análises textuais.

Caso você precise desconsiderar o padrão default do CountVectorizer e adicionar palavras com caracteres únicos em sua análise, você pode adicionar o argumento token_pattern passando REGEX de seleção. Essa expressão regular (REGEX) define palavra como tendo uma ou mais caracteres.


# Array de N-Gramas

Vamos utilizar o CountVectorizer para criar um array com as  contagens de n-gramas. além disso, vamos criar duas frases que desejamos analizar, e transformar cada texto em um vetor numérico representando o ocorrência de palavras.

Notar que cada linha representa um texto e cada coluna ou index representa o termos do vocabulário. Iremos ver isso claramente no mapeamento abaixo.

* texto_a_ser_comparado = "Suponha que esse seja o texto que desejo comparar".

* texto_fonte = "Suponha que esse seja o texto principal".

```bash
# Número de n_gramas
n = 1

# Instancia o contador de n-gramas
counts = CountVectorizer(analyzer='word', ngram_range=(n,n))

# Cria um matriz de contagem de n-gramas para os dois textos
n_gramas = counts.fit_transform([texto_a_ser_comparado, texto_fonte])

# Cria um dicionário de n-gramas
vocab2int = counts.fit([texto_a_ser_comparado, texto_fonte]).vocabulary_

n_gramas_array = n_gramas.toarray()

# printar o dicionario de palavras index
print('Vetor de n-gramas:\n\n', n_gramas_array)
print()
print('Dicionario de n-gramas(unigrama):\n\n',vocab2int)
```

### Saída do vetor de n-gramas
```bash
Vetor de n-gramas:

 [[1 1 1 0 2 1 1 1]
 [0 0 1 1 1 1 1 1]]

Dicionario de n-gramas(unigrama):

 {'suponha': 6, 'que': 4, 'esse': 2, 'seja': 5, 'texto': 7, 'desejo': 1, 'comparar': 0, 'principal': 3}
```

```bash
texto_a_ser_comparado = "Suponha que esse seja o texto que desejo comparar"
texto_fonte = "Suponha que esse seja o texto principal"
```

Acima temos os vetores que codificam cada texto. Na linha superior temos o n-gramas do texto_a_ser_comparado e na linha inferior temos a  codificação do textO_fonte. Podemos analisar se os textos possuem n_gramas em comum através de suas colunas. por exemplo, ambos possuem a palavra texto (inddice 7 - ultima coluna). O mesmo vale para os unigramas [essa],[seja],[que]e[suponha]. Notar que o unigrama [que] ocorre duas vezes no segundo texto.

```bash
n_gramas
```


### print
```bash
array([[1, 1, 1, 0, 2, 1, 1, 1],
       [0, 0, 1, 1, 1, 1, 1, 1]])
```

```bash
intersection_list = np.amin(n_gramas.toarray(), axis = 0)
intersection_list
```
### print
```bash
array([0, 0, 1, 0, 1, 1, 1, 1])
```

```bash
intersection_count = np.sum(intersection_list)
intersection_count
```

### print
```bash
5
```

```bash
index_A = 0
A_count = np.sum(n_gramas.toarray()[index_A])
A_count
```

### print
```bash
8
```

```bash
intersection_count/A_count
```

```bash
0.625
```
