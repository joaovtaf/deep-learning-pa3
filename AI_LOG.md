# AI_LOG

Usamos o Claude Code dentro do VS Code. As decisões de método foram nossas, depois de discutir as opções com ele.

Ele ajudou a montar o pipeline sem vazar val nem test_public na escolha do modelo. O professor nunca vê o val, o prompt foi fixado antes de olhar qualquer resultado, e a época e o tau são escolhidos no val, com o test entrando só no fim.

Quando a GPU do Kaggle e a do Colab ficaram sem vaga, ele mediu a velocidade da nossa CPU. O aluno levaria mais de 20 horas, então rodamos cada tau num notebook de CPU separado no Kaggle, em paralelo e com menos épocas.

Quando o aluno tirou F1 0 em eastern_cottontail, o Claude conferiu se era bug. Pelas confusões, os coelhos viram gambá e as galinhas viram coelho, o que parece atalho de fundo. Vamos investigar isso com Grad-CAM na Parte 3.
