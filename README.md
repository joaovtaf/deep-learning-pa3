# PA3 de Aprendizado Profundo: poucos rótulos no ENA24

Fizemos a trilha C (professor e aluno). O professor é o BioCLIP 2 em zero-shot, que rotula as 6505 imagens do train. O aluno é uma ResNet-18 partindo do ImageNet, treinada nos pseudo-rótulos do professor junto com os 105 rótulos de `rotulos_5.csv`. Para o ruído dos pseudo-rótulos usamos filtro por confiança: só entra pseudo-rótulo com confiança >= tau, e o tau é escolhido no val.

Precisa de Python 3 com `torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib` e `pillow`. O `open_clip_torch` só precisa para rodar o professor.

```
pip install torch torchvision pandas scikit-learn matplotlib pillow open_clip_torch
```

Os dados (`pa3_dados.zip`) vão descompactados na raiz do repositório, com `images/`, `labels/` e `classes.csv`. Se estiverem em outro lugar, aponta a variável `PA3_DADOS` para a pasta.

## Como rodar

```
python professor.py        # BioCLIP 2 no train e no test_public, gera resultados/professor_*.csv
python parte0.py           # piso: CNN do zero e ResNet-18 do ImageNet so com rotulos_5
python aluno.py            # aluno para cada tau, escolhe o tau no val
python parte1_analise.py   # numeros e figuras da parte 1
```

O professor rodou na nossa máquina, em CPU, porque o enunciado só deixa usar foundation model no Colab ou na máquina da dupla. Demorou umas 3 horas. Os CSVs dele já estão em `resultados/`, então dá para pular esse passo. O val nunca passa pelo professor.

O resto rodou em notebooks do Kaggle. A GPU do Kaggle e a do Colab ficaram sem vaga, então o aluno rodou em CPU, um notebook para cada tau, e por isso treina só 6 épocas de 3000 amostras. Cada tau grava seus próprios arquivos, e o `--juntar` escolhe o tau no val depois que todos terminam:

```
python aluno.py --taus 0.7   # um desses para cada tau (0, 0.5, 0.7, 0.9)
python aluno.py --juntar
```

## Resultados até agora (test_public)

| modelo | F1 macro | acurácia |
|---|---|---|
| piso (a), CNN do zero | 0,282 | 0,317 |
| piso (b), ResNet-18 ImageNet | 0,425 | 0,461 |
| professor, BioCLIP 2 zero-shot | 0,560 | 0,610 |
| aluno sem filtro (tau = 0) | 0,470 | 0,501 |
| aluno, tau = 0,9 escolhido no val | 0,573 | 0,636 |
| teto do monitor | 0,686 | |

O piso (b) decora rápido. Com 105 imagens a acurácia de treino passa de 98% na época 7, e a perda de val não cai mais depois da época 4. O professor erra quase tudo de bicho doméstico (dog, chicken, domestic_cat) e manda bird para american_crow. O aluno acerta essas classes porque tem os rótulos verdadeiros delas, mas piora em eastern_cottontail e red_fox, onde o professor ia bem. De dia o aluno ganha do professor (F1 0,518 contra 0,423) e de noite perde (0,464 contra 0,648). O detalhe por classe está em `resultados/parte1_por_classe.csv` e as figuras em `figs/`.
