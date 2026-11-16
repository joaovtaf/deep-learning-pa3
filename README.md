# PA3 de Aprendizado Profundo: poucos rótulos no ENA24

Fizemos a trilha C (professor e aluno). O professor é o BioCLIP 2 em zero-shot, que rotula as 6505 imagens do train. O aluno é uma ResNet-18 partindo do ImageNet, treinada nos pseudo-rótulos do professor junto com os rótulos verdadeiros do orçamento. Para o ruído dos pseudo-rótulos usamos filtro por confiança: só entra pseudo-rótulo com confiança >= tau, e o tau é escolhido no val.

Precisa de Python 3.11 ou mais novo (o `parte3.py` usa `hashlib.file_digest`) com `torch`, `torchvision`, `pandas`, `scikit-learn`, `matplotlib` e `pillow`. O `open_clip_torch` só é necessário para rodar o professor. Na nossa máquina rodamos com Python 3.14, PyTorch 2.12, torchvision 0.27 e scikit-learn 1.9. No Mac das Partes 2 e 3, com Python 3.13, PyTorch 2.11.0, torchvision 0.26.0, pandas 2.3.3, scikit-learn 1.7.1, Pillow 12.1.1 e matplotlib 3.10.3. Os notebooks do Kaggle usaram o ambiente padrão deles.

```
pip install torch torchvision pandas scikit-learn matplotlib pillow open_clip_torch
```

Os dados (`pa3_dados.zip`) vão descompactados na raiz do repositório, com `images/`, `labels/` e `classes.csv`. Se ficarem em outra pasta (o zip cria `pa3_dados/`), aponta a variável `PA3_DADOS` para ela. Se a leitura em paralelo der problema na sua máquina, `PA3_WORKERS=0` desliga os workers.

## Modelo entregue

O modelo entregue é o `modelo_final.pt`, uma ResNet-18 com 11.187.285 parâmetros. É o aluno do orçamento de 5 rótulos por classe, com tau = 0,9 e seed 0, escolhido no val entre as três seeds. Ele roda em CPU, sem internet, e classifica as 811 imagens do test_public em pouco mais de 1 minuto num notebook:

```
python predict.py --imagens images/test_public --saida preds.csv
python predict.py --imagens images/test_public --saida preds.csv --rotulos labels/test_public.csv   # avalia
```

O CSV de saída tem as colunas `image_id,label`, com os nomes de `classes.csv`. Com `--rotulos`, o script também imprime o F1 macro e a acurácia. No test_public deu F1 macro 0,589 e acurácia 0,656.

## Como treinar

```
python professor.py        # BioCLIP 2 no train e no test_public, gera resultados/professor_*.csv
python parte0.py           # piso: CNN do zero e ResNet-18 do ImageNet so com rotulos_5
python aluno.py            # aluno para cada tau, escolhe o tau no val
python parte1_analise.py   # numeros e figuras da parte 1
python parte2.py           # curva de orcamento (1, 5 e 20 rotulos por classe)
python parte3.py --preparar   # escolhe a seed no val e copia os pesos para modelo_final.pt
python parte3.py           # galeria de Grad-CAM
```

Cuidado com o `parte3.py --preparar`: ele escolhe a seed com os pesos que encontrar em `pesos/`, que não é versionada, e sobrescreve o `modelo_final.pt`. Num clone novo ele treina de novo e o modelo sai diferente do entregue. Para só avaliar o modelo entregue, use o `predict.py`.

O professor rodou na nossa máquina, em CPU, porque o enunciado só deixa usar foundation model no Colab ou na máquina da dupla. Demorou umas 3 horas. O prompt foi fixado antes de ver qualquer resultado: `"a camera trap photo of a {classe}."`, com o nome da classe trocando `_` por espaço. Os CSVs dele já estão em `resultados/`, então dá para pular esse passo. O val nunca passa pelo professor.

O aluno das Partes 0 e 1 rodou em notebooks de CPU do Kaggle, porque a GPU do Kaggle e a do Colab ficaram sem vaga. Por isso ele treina só 6 épocas de 3000 amostras. Cada tau rodou num notebook separado e grava seus próprios arquivos, e o `--juntar` escolhe o tau no val no fim (`python aluno.py --taus 0.7` para cada tau e depois `python aluno.py --juntar`). As oito execuções novas da Parte 2 e a seed 0 reproduzida da Parte 3 rodaram num Mac com GPU Apple MPS. A semente não fixa o determinismo do backend, então o mesmo treino pode mudar um pouco de um dispositivo para outro.

## Partes 0 e 1 (seed 0, test_public)

| modelo | F1 macro | acurácia |
|---|---|---|
| piso (a), CNN do zero | 0,282 | 0,317 |
| piso (b), ResNet-18 ImageNet | 0,425 | 0,461 |
| professor, BioCLIP 2 zero-shot | 0,560 | 0,610 |
| aluno sem filtro (tau = 0) | 0,470 | 0,501 |
| aluno, tau = 0,9 escolhido no val | 0,573 | 0,636 |
| teto do monitor | 0,686 | |

O piso (b) decora rápido. Com 105 imagens a acurácia de treino passa de 98% na época 7, e a perda de val não cai mais depois da época 4. O professor erra quase tudo de bicho doméstico (dog, chicken, domestic_cat) e manda bird para american_crow. Nessas classes o aluno ganha do professor porque os 5 rótulos verdadeiros corrigem um erro que o professor repete. Ele quase nunca gera pseudo-rótulo de dog (27 imagens do train), e o sampler balanceado faz esses poucos exemplos pesarem. O dog vai de 0,03 para 0,79 e o chicken de 0,17 para 0,56. Em domestic_cat o ganho é pequeno (de 0,14 para 0,34). O aluno piora em eastern_cottontail e red_fox, onde o professor ia bem. De dia o aluno ganha do professor (F1 0,518 contra 0,423) e de noite perde (0,464 contra 0,648). Dia e noite foram separados pela diferença entre os canais RGB, já que infravermelho sai praticamente cinza. No test_public a separação é limpa: 391 imagens têm diferença menor que 1 e nenhuma fica entre 1 e 6. O F1 de cada recorte usa só as classes que aparecem nele, então a comparação vale entre modelos dentro do mesmo recorte, e não entre o dia e a noite. O detalhe por classe está em `resultados/parte1_por_classe.csv` e as figuras em `figs/`.

## Parte 2: curva de orçamento

Rodamos o piso (b) e o aluno com 1 e 20 rótulos por classe (seed 0) e com 5 rótulos (seeds 0, 1 e 2). O piso treina 50 épocas e o aluno 6 épocas de 3000 amostras, igual às Partes 0 e 1, com a melhor época escolhida pelo F1 macro no val. O tau ficou fixo em 0,9, que foi o escolhido no val do orçamento 5 com seed 0, e as imagens que têm rótulo verdadeiro em cada orçamento saem dos pseudo-rótulos. O `parte2.py` reaproveita o que já rodou e só treina o que falta. Com `python parte2.py --consolidar` ele só refaz as tabelas e os gráficos.

| rótulos por classe | seeds | piso F1 macro | aluno F1 macro | piso acurácia | aluno acurácia |
|---|---|---|---|---|---|
| 1 | 1 | 0,198 | 0,487 | 0,229 | 0,567 |
| 5 | 3 | 0,431 ± 0,007 | 0,591 ± 0,020 | 0,463 ± 0,008 | 0,646 ± 0,009 |
| 20 | 1 | 0,670 | 0,615 | 0,689 | 0,677 |

Os números por execução estão em `curva_orcamento.csv`, e o gráfico com eixo log, o professor (0,560) e o teto do monitor (0,686) está em `figs/curva_orcamento.png`. O desvio é o amostral entre seeds. As curvas de treino e as matrizes de confusão de cada piso ficam em `figs/piso_r*_s*_*.png`. O ponto da seed 0 do aluno na curva é o treino original das Partes 0 e 1 (0,573 no test). O modelo entregue é a mesma configuração treinada de novo na Parte 3 (0,589), então ele não é exatamente nenhum dos pontos da curva.

O aluno ganha do piso por 0,289 com 1 rótulo por classe e por 0,160 com 5. Com 20 o piso ganha, e o aluno fica 0,055 abaixo. Os pseudo-rótulos ajudam mais quando há poucos rótulos humanos, mas esse experimento não isola a causa, porque o piso treina 50 épocas e o aluno 6, com amostragem diferente. Não mudamos nada no método depois de ver os números do test.

No orçamento de 5, os maiores ganhos do aluno sobre o piso foram em american_crow (+0,550), american_black_bear (+0,479), northern_raccoon (+0,441), woodchuck (+0,364) e striped_skunk (+0,341). Não houve ganho em bird, que fica com F1 zero nos dois, nem em eastern_chipmunk, que fica em 0,967 nos dois. A maior perda foi em red_fox (-0,126). Em eastern_gray_squirrel (-0,009), virginia_opossum (-0,006) e eastern_cottontail (-0,005) a perda média é menor que o desvio entre seeds. O ganho em horse (+0,117) vale pouco, porque são só 4 imagens no test. O detalhe está em `resultados/parte2_por_classe.csv` e `figs/parte2_ganho_por_classe.png`.

## Parte 3: galeria de falhas e atalhos

O modelo da galeria é o mesmo `modelo_final.pt`. O checkpoint original da seed 0 não estava na máquina onde a Parte 3 rodou, então o `parte3.py --preparar` treinou essa seed de novo com o mesmo protocolo, em arquivos com sufixo `reproduzido`. Entre as seeds 0, 1 e 2 do orçamento 5, a seed 0 reproduzida teve o melhor F1 no val (0,599, contra 0,550 e 0,551). O treino original da seed 0 tinha 0,573 no val e também ganharia. O F1 do reproduzido no test (0,589) é um pouco diferente do 0,573 da Parte 1. A escolha fica registrada em `resultados/parte3_selecao.csv`, e o `resultados/parte3_modelo.json` guarda o SHA-256 dos pesos, que a galeria confere antes de rodar.

O Grad-CAM ([Selvaraju et al.](https://arxiv.org/abs/1610.02391)) pondera os canais da `layer4` pela média dos gradientes do logit da classe prevista e passa uma ReLU. Na entrada de 320 x 320 o mapa tem 10 x 10, e a interpolação só serve para sobrepor na imagem. Cada mapa é normalizado entre 0 e 1, e amarelo quer dizer ativação relativa maior. A confiança nas figuras é a maior probabilidade do softmax, sem calibração.

Os exemplos e os diagnósticos foram escolhidos com o Codex, olhando as imagens e os mapas, e ficam fixos em `parte3_exemplos.csv` (`python parte3.py --candidatos` gera as folhas de candidatos de novo). A escolha não muda pesos nem hiperparâmetros, e os casos servem de ilustração, sem medir quantas vezes cada comportamento aparece no test. As figuras estão em `figs/parte3/` ([erros](figs/parte3/galeria_erros.png) e [acertos](figs/parte3/galeria_acertos.png)), e o `resultados/parte3_galeria.csv` tem os diagnósticos.

| exemplo | verdade | previsão | confiança | onde o mapa olha |
|---|---|---|---|---|
| [erro 1](figs/parte3/erro_b4ba3ffd9b742f4f.png) | bird | chicken | 96,38% | cerca e poste, longe da ave |
| [erro 2](figs/parte3/erro_50de954e62991d20.png) | eastern_cottontail | virginia_opossum | 96,31% | tronco e solo, animal na borda direita |
| [erro 3](figs/parte3/erro_69b9c70abd6097b3.png) | bobcat | grey_fox | 71,65% | cabeça e ombro do animal |
| [acerto 1](figs/parte3/acerto_8d4560b32a56b673.png) | american_black_bear | american_black_bear | 99,95% | misto, patas e solo |
| [acerto 2](figs/parte3/acerto_4fa6a1fb28437176.png) | american_crow | american_crow | 99,98% | corpo das aves |
| [acerto 3](figs/parte3/acerto_410d9af7caba9688.png) | domestic_cat | domestic_cat | 99,11% | fundo à direita, perto das luzes |
| [acerto 4](figs/parte3/acerto_fe0f9ec39c4b2192.png) | chicken | chicken | 97,36% | cerca e poste |
| [acerto 5](figs/parte3/acerto_dcbc4cb8d3a0c6b6.png) | eastern_chipmunk | eastern_chipmunk | 99,95% | reservatório e parede |

Dos cinco acertos, três olham principalmente para o cenário, um para o animal e um fica misto. O cercado aparece forte tanto no erro em que bird vira chicken quanto no acerto de chicken, o que sugere (sem provar, porque o Grad-CAM não mostra causa) que o modelo associa aquele cenário à classe. O erro de bobcat para grey_fox foca no próprio animal, então nem toda confusão vem do fundo.
