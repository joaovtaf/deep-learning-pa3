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

## Parte 2: curva de orcamento

Depois de descompactar o pacote oficial, indique a pasta que contem `images/`,
`labels/` e `classes.csv` (o zip cria uma pasta `pa3_dados/`):

```bash
export PA3_DADOS="$PWD/pa3_dados"
python parte2.py
```

O comando reaproveita as predicoes ja salvas e executa somente os experimentos
pendentes. Sao cinco execucoes por metodo: orcamentos 1 e 20 com seed 0, e
orcamento 5 com seeds 0, 1 e 2. A CNN do zero nao precisa ser repetida.
O piso usa 50 epocas, e o aluno usa 6 epocas com 3000 amostras por epoca,
mantendo o protocolo dos experimentos originais. A melhor epoca de cada treino
e escolhida pelo F1 macro no `val`.

O filtro do aluno fica fixo em **tau=0,9**, escolhido anteriormente no `val`
do orcamento 5, seed 0. Nao escolhemos novos limiares pelo `test_public`.
O professor e seus pseudo-rotulos sao reutilizados; as imagens com rotulo
verdadeiro em cada orcamento sao excluidas do conjunto de pseudo-rotulos.
As saidas de treino incluem orcamento e seed para nao sobrescrever experimentos.

Para apenas recalcular as metricas e os graficos, sem treinar:

```bash
python parte2.py --consolidar
python -m unittest test_parte2.py
```

Saidas:

- `curva_orcamento.csv`: metodo, orcamento (rotulos por classe), seed, F1 macro e acuracia.
- `resultados/parte2_resumo.csv`: media e desvio amostral (`ddof=1`); o desvio
  fica vazio quando ha uma unica seed.
- `figs/curva_orcamento.png`: eixo log com 21, 105 e 420 rotulos, barras de
  desvio entre seeds, professor zero-shot e teto do monitor (0,686).
- `resultados/parte2_por_classe_seeds.csv` e `parte2_por_classe.csv`: F1 do piso,
  F1 do aluno e ganho aluno menos piso no orcamento 5, por seed e agregado.
- `figs/parte2_ganho_por_classe.png`: ganho medio e desvio das diferencas
  pareadas por seed.
- `resultados/parte2_pendencias.csv`: combinacoes que ainda precisam rodar.

Enquanto faltarem execucoes, os graficos recebem a marca `PARCIAL` e a
consolidacao retorna codigo 1. Nenhum numero ausente e preenchido ou estimado.
A GPU CUDA ou Apple MPS e usada automaticamente quando disponivel; caso
contrario, o treino usa CPU. `PA3_WORKERS=0` desativa processos de leitura
paralelos quando necessario no ambiente local.

As duas execucoes originais de orcamento 5, seed 0, foram reaproveitadas da CPU;
as oito novas execucoes usaram Apple MPS. Ambiente local: Python 3.13,
PyTorch 2.11.0, torchvision 0.26.0, pandas 2.3.3, scikit-learn 1.7.1,
Pillow 12.1.1 e matplotlib 3.10.3. Resultados estocasticos podem variar entre
dispositivos.

### Resultados da curva (test_public)

| Rotulos/classe | Seeds | Piso: F1 macro | Aluno: F1 macro | Piso: acuracia | Aluno: acuracia |
|---|---|---|---|---|---|
| 1 | 1 | 0,198 | 0,487 | 0,229 | 0,567 |
| 5 | 3 | 0,431 +/- 0,007 | 0,591 +/- 0,020 | 0,463 +/- 0,008 | 0,646 +/- 0,009 |
| 20 | 1 | 0,670 | 0,615 | 0,689 | 0,677 |

Professor zero-shot: F1 macro 0,560 e acuracia 0,610. Teto de referencia do
monitor: F1 macro 0,686. As dez execucoes estao completas.

O ganho do aluno sobre o piso foi +0,289 com 1 rotulo/classe e +0,160 com
5 rotulos/classe. Com 20, o piso foi melhor: diferenca aluno menos piso de
-0,055. Portanto, neste protocolo, os pseudo-rotulos ajudaram mais quando
havia menos rotulos humanos; nao houve vantagem no orcamento 20. Isso e
compativel com uma limitacao dos pseudo-rotulos, mas o experimento nao isola
essa causa: o piso treina 50 epocas e o aluno 6, com estrategias de amostragem
diferentes. Nao mudamos o metodo depois de observar os numeros do teste.

### Ganho por classe no orcamento 5

Nas tres seeds, o aluno obteve F1 macro **0,591 +/- 0,020**, contra
**0,431 +/- 0,007** do piso: ganho medio de 0,160. Os maiores ganhos de F1
por classe foram `american_crow` (+0,550), `american_black_bear` (+0,479),
`northern_raccoon` (+0,441), `woodchuck` (+0,364) e `striped_skunk` (+0,341).

Nao houve ganho em `bird` (F1 zero nos dois modelos) nem em
`eastern_chipmunk` (F1 medio 0,967 nos dois). A maior perda ocorreu em
`red_fox` (-0,126). Tambem houve perdas medias pequenas em
`eastern_gray_squirrel` (-0,009), `virginia_opossum` (-0,006) e
`eastern_cottontail` (-0,005); nesses casos, o desvio entre seeds e maior
que a diferenca media. Os valores completos e os desvios das diferencas
pareadas estao em `resultados/parte2_por_classe.csv`.
O ganho de `horse` (+0,117) deve ser interpretado com cuidado: ha apenas
quatro imagens dessa classe no teste.

## Resultados das Partes 0 e 1 (seed 0, test_public)

| modelo | F1 macro | acurácia |
|---|---|---|
| piso (a), CNN do zero | 0,282 | 0,317 |
| piso (b), ResNet-18 ImageNet | 0,425 | 0,461 |
| professor, BioCLIP 2 zero-shot | 0,560 | 0,610 |
| aluno sem filtro (tau = 0) | 0,470 | 0,501 |
| aluno, tau = 0,9 escolhido no val | 0,573 | 0,636 |
| teto do monitor | 0,686 | |

O piso (b) decora rápido. Com 105 imagens a acurácia de treino passa de 98% na época 7, e a perda de val não cai mais depois da época 4. O professor erra quase tudo de bicho doméstico (dog, chicken, domestic_cat) e manda bird para american_crow. O aluno acerta essas classes porque tem os rótulos verdadeiros delas, mas piora em eastern_cottontail e red_fox, onde o professor ia bem. De dia o aluno ganha do professor (F1 0,518 contra 0,423) e de noite perde (0,464 contra 0,648). O detalhe por classe está em `resultados/parte1_por_classe.csv` e as figuras em `figs/`.
