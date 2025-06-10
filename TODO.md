### TODO's

Study the class implementation

Explore data to see what happens when the model is mistaken

Remove magic numbers and parameters, use config files (airflow??)

Study the possibility of including sentiment analysis of news as part of the inference
--------------------------------

## O que a aplicação deve fazer

- Pega uma ação
- Treina em dados históricos
- Salva melhor modelo para inferencia
- - A partir desse momento, deve:
- - Pegar dados do ultimo dia (ou ultimos N dias) para fazer fine tuning no modelo
- - Manter 2 modelos: o mais recente e o de melhor performance