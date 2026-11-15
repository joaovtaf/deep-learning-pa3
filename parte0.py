# Piso supervisionado: (a) CNN pequena do zero, (b) ResNet-18 do ImageNet.
import argparse
import os

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from comum import *

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas_a", type=int, default=100)
    ap.add_argument("--epocas_b", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orcamento", type=int, choices=[1, 5, 20], default=5)
    ap.add_argument("--so_b", action="store_true", help="treina apenas o piso ImageNet (Parte 2)")
    args = ap.parse_args()
    for d in ["resultados", "figs", "pesos"]:
        os.makedirs(os.path.join(AQUI, d), exist_ok=True)

    ids, y = le(f"rotulos_{args.orcamento}")
    idv, yv = le("val")
    idt, yt = le("test_public")
    tr = Imagens("train", ids, y, TREINO)
    va = Imagens("val", idv, yv, TESTE)
    te = Imagens("test_public", idt, yt, TESTE)

    tag = f"r{args.orcamento}_s{args.seed}"
    legado = args.orcamento == 5 and args.seed == 0 and not args.so_b
    linhas = []
    versoes = [("a_cnn_zero", cnn_pequena, args.epocas_a, 1e-3),
               ("b_resnet18_imagenet", resnet18, args.epocas_b, 3e-4)]
    for nome, cria, ep, lr in versoes[1:] if args.so_b else versoes:
        semente(args.seed)
        model = cria()
        print(nome, "parametros:", n_params(model), flush=True)
        hist = treina(model, tr, va, ep, lr)
        prefixo = f"parte0_{nome}" if legado else f"parte0_{nome}_{tag}"
        hist.to_csv(os.path.join(AQUI, "resultados", f"{prefixo}_historico.csv"), index=False)
        # test_public so depois da epoca ja escolhida no val
        p = prediz(model, te)
        f1, acc = metricas(yt, p.argmax(1))
        linhas.append(dict(modelo=nome, epoca_val=int(hist.f1_val.idxmax()) + 1, f1_val=hist.f1_val.max(),
                           f1_macro=f1, acuracia=acc))
        print(linhas[-1], flush=True)
        pd.DataFrame(dict(image_id=idt, label=[CLASSES[k] for k in yt], pred=[CLASSES[k] for k in p.argmax(1)],
                          conf=p.max(1))).to_csv(os.path.join(AQUI, "resultados", f"{prefixo}_test.csv"), index=False)

    pd.DataFrame(linhas).to_csv(os.path.join(AQUI, "resultados", "parte0.csv" if legado else f"piso_{tag}.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(AQUI, "pesos", "piso_r5.pt" if legado else f"piso_{tag}.pt"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoca, hist.perda_treino, label="treino")
    ax[0].plot(hist.epoca, hist.perda_val, label="val")
    ax[0].set_title("perda")
    ax[1].plot(hist.epoca, hist.acc_treino, label="acc treino")
    ax[1].plot(hist.epoca, hist.acc_val, label="acc val")
    ax[1].plot(hist.epoca, hist.f1_val, label="F1 macro val")
    ax[1].set_title("acuracia / F1")
    for a in ax:
        a.set_xlabel("epoca")
        a.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figs", "parte0b_curva.png" if legado else f"piso_{tag}_curva.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay.from_predictions(yt, p.argmax(1), labels=range(len(CLASSES)), display_labels=CLASSES,
                                            normalize="true", values_format=".2f", ax=ax, colorbar=False,
                                            xticks_rotation=90)
    ax.set_title(f"Piso (b), {args.orcamento} rotulos/classe, seed {args.seed}, test_public")
    fig.tight_layout()
    fig.savefig(os.path.join(AQUI, "figs", "parte0b_confusao.png" if legado else f"piso_{tag}_confusao.png"), dpi=120)
