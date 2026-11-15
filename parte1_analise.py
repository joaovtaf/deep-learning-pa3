# Analise da parte 1 a partir dos CSVs: professor no test_public, ruido, dia/noite e aluno x professor.
import os

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, f1_score

from comum import AQUI, CLASSES, RAIZ, metricas

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = lambda f: os.path.join(AQUI, "resultados", f)
F = lambda f: os.path.join(AQUI, "figs", f)
os.makedirs(os.path.join(AQUI, "figs"), exist_ok=True)


def noite(i):
    # infravermelho sai praticamente cinza, os 3 canais quase iguais
    a = np.asarray(Image.open(os.path.join(RAIZ, "images", "test_public", i + ".jpg")).convert("RGB").resize((64, 64)), float)
    return np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 4


def f1_classe(d):
    return f1_score(d.label, d.pred, labels=CLASSES, average=None, zero_division=0)


gab = pd.read_csv(os.path.join(RAIZ, "labels", "test_public.csv"))
gab["noite"] = gab.image_id.map(noite)
prof = gab.merge(pd.read_csv(R("professor_test_public.csv")), on="image_id")

def dia_noite(d, quem):
    linhas = []
    for parte, s in [("tudo", d), ("dia", d[~d.noite]), ("noite", d[d.noite])]:
        # so as classes que aparecem no subconjunto, senao classe ausente entra como F1 zero
        f1 = f1_score(s.label, s.pred, labels=sorted(s.label.unique()), average="macro", zero_division=0)
        linhas.append(dict(modelo=quem, conjunto=parte, n=len(s), f1_macro=f1, acuracia=(s.label == s.pred).mean()))
    return pd.DataFrame(linhas)


dn = dia_noite(prof, "professor")
print(dn)
dn.to_csv(R("professor_metricas.csv"), index=False)

fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay.from_predictions(prof.label, prof.pred, labels=CLASSES, normalize="true", values_format=".2f",
                                        ax=ax, colorbar=False, xticks_rotation=90)
ax.set_title("Professor (BioCLIP 2 zero-shot) no test_public, normalizada por linha")
fig.tight_layout()
fig.savefig(F("professor_confusao.png"), dpi=120)

# precisao e cobertura dos pseudo-rotulos em funcao de tau, rodando no test_public como se nao tivesse rotulo
taus = np.round(np.arange(0, 1, 0.1), 1)
pc = []
for t in taus:
    s = prof[prof.conf >= t]
    for c in CLASSES:
        pc.append(dict(tau=t, classe=c, cobertura=(s.label == c).sum() / (prof.label == c).sum(),
                       precisao=(s[s.pred == c].label == c).mean() if (s.pred == c).any() else np.nan))
pc = pd.DataFrame(pc)
pc.to_csv(R("professor_precisao_cobertura.csv"), index=False)
glob_ = pd.DataFrame([dict(tau=t, cobertura=(prof.conf >= t).mean(), precisao=(prof[prof.conf >= t].pred ==
                                                                               prof[prof.conf >= t].label).mean())
                      for t in taus])
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(glob_.tau, glob_.precisao, "o-", label="precisao")
ax.plot(glob_.tau, glob_.cobertura, "o-", label="cobertura")
ax.set_xlabel("tau")
ax.legend()
ax.set_title("pseudo-rotulos do professor no test_public")
fig.tight_layout()
fig.savefig(F("professor_tau.png"), dpi=120)

# pra onde vao os pseudo-rotulos no train
tr = pd.read_csv(R("professor_train.csv"))
cont = tr.pred.value_counts().reindex(CLASSES, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(CLASSES, cont.values)
ax.tick_params(axis="x", rotation=90)
ax.set_title("pseudo-rotulos do professor no train (6505 imagens)")
fig.tight_layout()
fig.savefig(F("professor_train_distribuicao.png"), dpi=120)

if os.path.exists(R("aluno_r5_s0_test.csv")):
    t = pd.read_csv(R("aluno_r5_s0_test.csv"))
    tau = t.tau[t.escolhido].iloc[0]
    aluno = gab.merge(pd.read_csv(R(f"aluno_r5_s0_t{tau}_test.csv")).drop(columns="label"), on="image_id")
    piso = gab.merge(pd.read_csv(R("parte0_b_resnet18_imagenet_test.csv")).drop(columns="label"), on="image_id")
    comp = pd.DataFrame(dict(classe=CLASSES, n_test=gab.label.value_counts().reindex(CLASSES).values,
                             piso=f1_classe(piso), professor=f1_classe(prof), aluno=f1_classe(aluno)))
    comp["aluno_menos_professor"] = comp.aluno - comp.professor
    comp = comp.sort_values("aluno_menos_professor", ascending=False)
    comp.to_csv(R("parte1_por_classe.csv"), index=False)
    print(comp.round(3).to_string(index=False))
    dn = pd.concat([dia_noite(piso, "piso"), dia_noite(prof, "professor"), dia_noite(aluno, "aluno")])
    print(dn.round(3).to_string(index=False))
    dn.to_csv(R("parte1_dia_noite.csv"), index=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(comp))
    for k, nome in enumerate(["piso", "professor", "aluno"]):
        ax.bar(x + (k - 1) * 0.27, comp[nome], 0.27, label=nome)
    ax.set_xticks(x, comp.classe, rotation=90)
    ax.set_ylabel("F1 no test_public")
    ax.legend()
    fig.tight_layout()
    fig.savefig(F("parte1_por_classe.png"), dpi=120)
