# Solução case Rio de Janeiro Airbnb


### Como foi a definição da sua estratégia de modelagem?

R: Escolhi abordar o case como um problema de regressão para prever o preço de estadia. Dessa forma, procurei primeiro analisar os dados disponíveis e avaliar quais poderiam ser bons preditores.

O data set como um todo precisou ser bastante trabalhado antes de se aplicar algum tipo de modelo. Muitos dados números estavam em formato de texto (ex. o preço da estadia), existiam muitos dados faltantes e categorizações que precisavam ser melhor estruturadas,

Muitas colunas significavam a mesma informação e podem ser eliminadas, ou caíria num sério problema de multi colineariedade. 

 * No caso de período mínimo e máximo de estadia, ficaremos com `['minimum_nights', 'maximum_nights']` por agregarem a informação mais relevante. Serão eliminadas:

```python
['minimum_minimum_nights','maximum_minimum_nights', 'minimum_maximum_nights',
    'maximum_maximum_nights', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']
```

* `number_of_reviews` agrega de forma satisfatória a mesma informação que  

```python
['number_of_reviews_ltm','number_of_reviews_l30d']
```

* `neighbourhood_cleansed` está completo e informa de forma lima a localização do imóvel  

```python
['neighbourhood']
```

* `review_scores_rating` possui agrega de forma satisfatória a mesma informação que  

```python
['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location','review_scores_value','reviews_per_month]
```

Outras colunas não apresentavam informações que fossem relevantes para um modelo de regressão 

* Colunas contendo links não são relevantes para um modelo de precificação

```python
['listing_url', 'picture_url', 'host_url', 'host_thumbnail_url', 'host_picture_url']
```

* Colunas com descrições textuais que não impactam no modelo de precificação

```python
['description', 'neighborhood_overview']
```

* Considero que disponibilidade não impacta na precificação, uma vez que o valor da acomodação não muda na plataforma por estar indisponível 

```python
['has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365']
```

* Colunas sem nenhum registro e sem alternativas de como serem preenchidas

```python
['neighbourhood_group_cleansed', 'bathrooms', 'calendar_updated', 'license']
```

* Colunas diversas que são apenas de registro do banco de dados do airbnb

```python
['scrape_id', 'last_scraped', 'calendar_last_scraped', 'name','first_review', 'last_review']
```

* Estas Informações sobre o host ou possuiam quantidades enormes de dados faltantes, ou considerei de pouco valor para avaliar o preço de uma acomodação

```python
['host_id', 'host_name', 'host_about', 'host_neighbourhood', 'calculated_host_listings_count',
 'calculated_host_listings_count_entire_homes','calculated_host_listings_count_private_rooms',
 'calculated_host_listings_count_shared_rooms','host_location', 'host_response_time',
 'host_response_rate', 'host_acceptance_rate','host_verifications']
```



Depois de selecionar as principais features para o modelo, ainda era exigido um grande esforço de aprimoramento dos dados. A coluna amenidades, por exemplo, me parecia ser de grande valor para uma precificação. me parece razoável supor que alguém aceite pagar mais por uma hospedagem que inclua estacionamento, academia, garagem etc.

A coluna possuia um json com todas as amenidades disponíveis no anuncio. Selecionei as que além de bastante frequentes, me pareceram ser de maior relevância. Sendo elas: ```['kitchen', 'wifi', 'free parking', 'refrigerator','gym']```

As informações sobre disponibilidade de banheiros estavam em formato de texto e exigiram um tratamento. 

Por fim, o preço (vairável alvo) apresentava uma distribuição com calda muito longa. Com valores registrados na casa de centenas de milhares de reais e alguns zerados.  Porém 95% dos dados de preços são inferiores a $1925.50. Além disso, observei que uma transformação log nos dados os aproximava de uma distribuição mais simétrica e melhor comportada.

Procurei avaliar se existia algum tipo de sazonalidade, e embora os dados indiquem que exista sazonalidade dentro da semana (vários picos repetidos quando plotada a série diária de preços), a série total era curta demais para se estimar sazonalidade semanal ou mensal.

Conhecendo o Rio de Janeiro e a diferença de segurança e renda entre seus bairros, investiguei o preço mediano cobrando na hospedagem e a contagem de anúncios plotando um mapa (dados geográficos diponíveis no dataset do airbnb)

Foi possível ver que os bairros oferecem um bom critírio para precificar o anuncio. Porém, existiam 151 bairros com anuncios. Alguns com muito poucos pontos para se treinar um modelo. Decidi, portanto, por criar um modelo auxiliar de clusterização de forma a agrupar as coordenadas geográficas dos anuncios por proximidade.

Com um algoritmo de K-means, auxiliado pela análise da Elbow Curve, cheguei a um valor de 12 grupos para as localidades. 

Com essa preparação dos dados em mãos, pude fazer o modelo preditivo.

A regressão escolhida foi uma log-linear $ln(y) = \alpha X + \beta$



### Como foi definida a função de custo utilizada?

A função de custo escolhida foi a tradicional "Mínimos quadrados", ou "Ordinary least squares" - OLS.

Essa é a função objetivo default na maioria dos pacotes de regressão linear. 

Sem análises mais aprofundadas de como os dados se comportam, nem com problemas de convergência, não caberia pensar em algum outro tipo de função.



### Qual foi o critério utilizado na seleção do modelo final?

A análise final do modelo foi por meio de análise de p-valor dos coeficientes e intervalos de confiança. Além disso obervei as medidas indicativas de qualidade das features escolhidas

### Qual foi o critério utilizado para validação do modelo? Por que escolheu utilizar este método?

Fiz uma separação em dataset de treino e dataset de teste. É um critério simples, porém efetivo para modelos lineares. O primeiros resultados já mostram que existem potenciais colineariedades.

### Quais evidências você possui de que seu modelo é suficientemente bom?

Não considero que o modelo seja muito bom. A tabela de resultados mostra que algumas features do modelo apresentaram p-valor muito elevado para seus coeficientes.

Ao que parece a quantidade de clusters ficou acima do necessário. Caberia testar reduzir para 11 ou 10.

As features relacionadas a amenidades também apresentaram resultados ruim. Seus coeficientes numericamente iguais a zero mostram que foram más escolhas de preditores. 

Vale ressaltar que esse modelo de regressão simples não incorpora nenhum tipo de raciocínio apriorístico. Poderia-se construir um modelo onde restrigimos o sinal da contribuição de cada feature do modelo. Isso evitaria resultados contra intuitivos ou que neguem premissas razoáveis (ex. supor que acomodações maiores são mais caras)
