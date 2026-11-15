# AI_LOG

Na Parte 2, usamos o Codex para adaptar o piso aos tres
orcamentos, separar as saidas por seed e automatizar as oito execucoes pendentes.
Ele preparou a consolidacao do F1 macro e da acuracia, media e desvio entre
seeds, curva de orcamento e ganho aluno menos piso por classe, com testes de
estatisticas e de alinhamento dos IDs. Mantivemos tau=0,9, escolhido no val
do experimento original, e os mesmos numeros de epocas e amostras. O pacote
oficial foi usado localmente, e o codigo passou a aproveitar a GPU Apple MPS.

Usamos o Claude Code dentro do VS Code. As decisões de método foram nossas, depois de discutir as opções com ele.

Ele ajudou a montar o pipeline sem vazar val nem test_public na escolha do modelo. O professor nunca vê o val, o prompt foi fixado antes de olhar qualquer resultado, e a época e o tau são escolhidos no val, com o test entrando só no fim.

Quando a GPU do Kaggle e a do Colab ficaram sem vaga, ele mediu a velocidade da nossa CPU. O aluno levaria mais de 20 horas, então rodamos cada tau num notebook de CPU separado no Kaggle, em paralelo e com menos épocas.

Quando o aluno tirou F1 0 em eastern_cottontail, o Claude conferiu se era bug. Pelas confusões, os coelhos viram gambá e as galinhas viram coelho, o que parece atalho de fundo. Vamos investigar isso com Grad-CAM na Parte 3.

Na Parte 3, pedimos ao Codex a continuidade da análise do modelo da Trilha C,
mantendo o orçamento de 5 rótulos por classe e a escolha de checkpoint pelo val.
Indicamos os pesos disponíveis; como faltava o checkpoint original da seed 0
nesse orçamento, a ferramenta reproduziu o treino em arquivos separados,
preservando os resultados anteriores. O Codex implementou e testou o Grad-CAM,
gerou a galeria de três erros e cinco acertos e redigiu diagnósticos a partir
da inspeção das imagens e dos mapas. Essas interpretações são exploratórias:
não criam rótulos de treino nem demonstram, sozinhas, a causa de cada previsão.
