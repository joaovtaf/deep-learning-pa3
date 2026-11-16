# AI_LOG

Usamos o Claude Code no VS Code nas Partes 0 e 1 e o Codex nas Partes 2 e 3. As decisões de método foram nossas, depois de discutir as opções com eles.

O Claude ajudou a montar o pipeline sem vazar val nem test_public na escolha do modelo. O professor nunca vê o val, o prompt foi fixado antes de olhar qualquer resultado, e a época e o tau são escolhidos no val, com o test entrando só no fim.

Quando a GPU do Kaggle e a do Colab ficaram sem vaga, ele mediu a velocidade da nossa CPU. O aluno levaria mais de 20 horas, então rodamos cada tau num notebook de CPU separado no Kaggle, em paralelo e com menos épocas.

Quando o aluno tirou F1 0 em eastern_cottontail, o Claude conferiu se era bug. Pelas confusões, os coelhos viravam gambá e as galinhas viravam coelho, o que parecia atalho de fundo. Esse caso de coelho virando gambá aparece de novo na galeria da Parte 3.

Na Parte 2, o Codex adaptou o piso para os três orçamentos, separou as saídas por seed e automatizou as oito execuções que faltavam, já usando a GPU MPS do Mac. Ele também fez a consolidação com média e desvio entre seeds, a curva de orçamento e o ganho por classe. Mantivemos tau = 0,9 e o mesmo número de épocas e amostras das Partes 0 e 1.

Na Parte 3, pedimos ao Codex para continuar a análise com o orçamento de 5 rótulos e a seed escolhida no val. O checkpoint original da seed 0 não estava salvo, então ele reproduziu esse treino em arquivos separados, sem apagar os resultados anteriores. Ele implementou o Grad-CAM, montou a galeria de três erros e cinco acertos e escreveu os diagnósticos a partir das imagens e dos mapas. Esses diagnósticos são exploratórios: o mapa sugere onde o modelo olhou, mas sozinho não prova a causa de cada previsão.
