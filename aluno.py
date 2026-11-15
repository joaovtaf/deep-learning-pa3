# Aluno: ResNet-18 treinada nos rotulos verdadeiros do orcamento + pseudo-rotulos do professor com conf >= tau.
# tau=0 e o aluno sem tratamento de ruido nenhum. Cada tau grava seus arquivos e o --juntar escolhe o tau no val,
# assim da pra rodar cada tau numa maquina diferente.
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch

from comum import *


def roda(tau, args, tag, ids_r, y_r, prof, va, te):
    semente(args.seed)
    sel = prof[prof.conf >= tau]
    ids = ids_r + sel.image_id.tolist()
    y = y_r + sel.pred.map(IDX).tolist()
    cont = np.bincount(y, minlength=len(CLASSES))
    # sampler balanceado, senao o urso domina os pseudo-rotulos
    sampler = torch.utils.data.WeightedRandomSampler(1.0 / cont[y], args.amostras or len(y), replacement=True)
    model = resnet18()
    hist = treina(model, Imagens("train", ids, y, TREINO), va, args.epocas, 3e-4, sampler=sampler)
    pd.DataFrame([dict(tau=tau, n_pseudo=len(sel), cobertura=len(sel) / len(prof),
                       epoca_val=int(hist.f1_val.idxmax()) + 1, f1_val=hist.f1_val.max())]) \
        .to_csv(R(f"aluno_{tag}_t{tau}_val.csv"), index=False)
    hist.to_csv(R(f"aluno_{tag}_t{tau}_historico.csv"), index=False)
    # o test e predito pra todo tau, mas so o escolhido no val (e o tau=0) vai pra tabela
    p = prediz(model, te)
    pd.DataFrame(dict(image_id=te.ids, label=[CLASSES[k] for k in te.y], pred=[CLASSES[k] for k in p.argmax(1)],
                      conf=p.max(1))).to_csv(R(f"aluno_{tag}_t{tau}_test.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(AQUI, "pesos", f"aluno_{tag}_t{tau}.pt"))


def juntar(tag):
    val = pd.concat([pd.read_csv(f) for f in glob.glob(R(f"aluno_{tag}_t*_val.csv"))]).sort_values("tau")
    print(val.to_string(index=False))
    val.to_csv(R(f"aluno_{tag}_val.csv"), index=False)
    escolhido = val.tau.iloc[val.f1_val.values.argmax()]
    print("tau escolhido no val:", escolhido)
    teste = []
    for tau in sorted({0.0, escolhido} & set(val.tau)):
        d = pd.read_csv(R(f"aluno_{tag}_t{tau}_test.csv"))
        f1, acc = metricas(d.label, d.pred)
        teste.append(dict(tau=tau, escolhido=tau == escolhido, f1_macro=f1, acuracia=acc))
    print(pd.DataFrame(teste).to_string(index=False))
    pd.DataFrame(teste).to_csv(R(f"aluno_{tag}_test.csv"), index=False)
    shutil.copy(os.path.join(AQUI, "pesos", f"aluno_{tag}_t{escolhido}.pt"), os.path.join(AQUI, "pesos", f"aluno_{tag}.pt"))


def R(f):
    return os.path.join(AQUI, "resultados", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0, 0.5, 0.7, 0.9])
    ap.add_argument("--orcamento", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    # 6 epocas de 3000 amostras porque so tinha CPU (a GPU do Kaggle e do Colab ficou sem vaga)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--amostras", type=int, default=3000, help="amostras por epoca (0 = tamanho do conjunto)")
    ap.add_argument("--juntar", action="store_true", help="so escolhe o tau no val com os arquivos ja gerados")
    args = ap.parse_args()
    tag = f"r{args.orcamento}_s{args.seed}"
    if not args.juntar:
        for d in ["resultados", "pesos"]:
            os.makedirs(os.path.join(AQUI, d), exist_ok=True)
        ids_r, y_r = le(f"rotulos_{args.orcamento}")
        prof = pd.read_csv(R("professor_train.csv"))
        prof = prof[~prof.image_id.isin(ids_r)]  # onde tem rotulo de verdade ele ganha do pseudo
        idv, yv = le("val")
        idt, yt = le("test_public")
        va = Imagens("val", idv, yv, TESTE)
        te = Imagens("test_public", idt, yt, TESTE)
        te.ids = idt
        for tau in args.taus:
            roda(tau, args, tag, ids_r, y_r, prof, va, te)
    if args.juntar or len(args.taus) > 1:
        juntar(tag)
