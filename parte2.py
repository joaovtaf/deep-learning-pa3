"""Executa os orcamentos pendentes e reproduz as tabelas e figuras da Parte 2."""
import argparse
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score

from comum import AQUI, RAIZ, CLASSES, DEV, metricas

ROOT = Path(AQUI)
RESULTADOS = ROOT / "resultados"
EXECUCOES = [(1, 0), (5, 0), (5, 1), (5, 2), (20, 0)]
TAU = 0.9  # escolhido no val de r5_s0; fixo em todos os novos experimentos
TETO = 0.686


def arquivo(metodo, orcamento, seed):
    if metodo == "piso":
        nome = f"parte0_b_resnet18_imagenet_r{orcamento}_s{seed}_test.csv"
        if (orcamento, seed) == (5, 0) and not (RESULTADOS / nome).exists() and (
                RESULTADOS / "parte0_b_resnet18_imagenet_test.csv").exists():
            nome = "parte0_b_resnet18_imagenet_test.csv"
    else:
        nome = f"aluno_r{orcamento}_s{seed}_t{TAU}_test.csv"
    return RESULTADOS / nome


def le_predicoes(path):
    d = pd.read_csv(path, dtype={"image_id": str})
    if d.empty or d.image_id.isna().any() or d.image_id.duplicated().any():
        raise ValueError(f"IDs vazios ou repetidos: {path}")
    if not d.label.isin(CLASSES).all() or not d.pred.isin(CLASSES).all():
        raise ValueError(f"Classe ausente ou desconhecida: {path}")
    return d.set_index("image_id").sort_index()


def consolidar():
    RESULTADOS.mkdir(exist_ok=True)
    (ROOT / "figs").mkdir(exist_ok=True)
    linhas, faltam, predicoes = [], [], {}
    gabarito = None
    for metodo in ["piso", "aluno"]:
        for orcamento, seed in EXECUCOES:
            path = arquivo(metodo, orcamento, seed)
            if not path.exists():
                faltam.append(dict(metodo=metodo, orcamento=orcamento, seed=seed))
                continue
            d = le_predicoes(path)
            if gabarito is None:
                gabarito = d.label
            elif not d.label.equals(gabarito):
                raise ValueError(f"Testes com IDs ou rotulos diferentes: {path}")
            f1, acc = metricas(d.label, d.pred)
            linhas.append(dict(metodo=metodo, orcamento=orcamento, seed=seed,
                               f1_macro=f1, acuracia=acc))
            predicoes[metodo, orcamento, seed] = d
    curva = pd.DataFrame(linhas, columns=["metodo", "orcamento", "seed", "f1_macro", "acuracia"])
    curva.to_csv(ROOT / "curva_orcamento.csv", index=False)
    pd.DataFrame(faltam, columns=["metodo", "orcamento", "seed"]).to_csv(
        RESULTADOS / "parte2_pendencias.csv", index=False)
    if curva.empty:
        return faltam
    resumo = curva.groupby(["metodo", "orcamento"]).agg(
        n_seeds=("seed", "count"), f1_media=("f1_macro", "mean"), f1_desvio=("f1_macro", "std"),
        acuracia_media=("acuracia", "mean"), acuracia_desvio=("acuracia", "std")).reset_index()
    resumo.to_csv(RESULTADOS / "parte2_resumo.csv", index=False)

    prof = pd.read_csv(RESULTADOS / "professor_test_public.csv", dtype={"image_id": str})
    if prof.image_id.duplicated().any():
        raise ValueError("IDs repetidos nas predicoes do professor")
    prof = prof.set_index("image_id").sort_index()
    if not prof.index.equals(gabarito.index) or not prof.pred.isin(CLASSES).all():
        raise ValueError("Professor e aluno devem usar o mesmo test_public")
    f1_prof, acc_prof = metricas(gabarito, prof.pred)
    pd.DataFrame([dict(modelo="professor", f1_macro=f1_prof, acuracia=acc_prof)]).to_csv(
        RESULTADOS / "parte2_referencia_professor.csv", index=False)
    status = "" if not faltam else f" (PARCIAL: {len(curva)}/10 execucoes)"
    fig, ax = plt.subplots(figsize=(8, 5))
    for metodo in ["piso", "aluno"]:
        s = resumo[resumo.metodo == metodo].sort_values("orcamento")
        ax.errorbar(s.orcamento * len(CLASSES), s.f1_media, yerr=s.f1_desvio.fillna(0),
                    marker="o", capsize=4, label="Piso ImageNet" if metodo == "piso" else "Aluno, tau=0,9")
    ax.axhline(TETO, color="black", linestyle="--", label=f"Teto do monitor ({TETO:.3f})")
    ax.axhline(f1_prof, color="gray", linestyle=":", label=f"Professor zero-shot ({f1_prof:.3f})")
    ax.set_xscale("log")
    ax.set_xticks([21, 105, 420], ["21 (1/classe)", "105 (5/classe)", "420 (20/classe)"])
    ax.set(xlabel="Numero de rotulos humanos (escala log)", ylabel="F1 macro no test_public",
           title="Curva de orcamento" + status, ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figs" / "curva_orcamento.png", dpi=160)
    plt.close(fig)

    ganhos = []
    for seed in [0, 1, 2]:
        if not all((m, 5, seed) in predicoes for m in ["piso", "aluno"]):
            continue
        piso, aluno = [predicoes[m, 5, seed] for m in ["piso", "aluno"]]
        fp, fa = [f1_score(d.label, d.pred, labels=CLASSES, average=None, zero_division=0)
                  for d in [piso, aluno]]
        for classe, p, a in zip(CLASSES, fp, fa):
            ganhos.append(dict(seed=seed, classe=classe, n_test=int((piso.label == classe).sum()),
                               piso=p, aluno=a, ganho=a - p))
    por_classe = pd.DataFrame(ganhos)
    por_classe.to_csv(RESULTADOS / "parte2_por_classe_seeds.csv", index=False)
    if not por_classe.empty:
        comp = por_classe.groupby("classe").agg(n_test=("n_test", "first"), n_seeds=("seed", "count"),
            piso_media=("piso", "mean"), aluno_media=("aluno", "mean"),
            ganho_media=("ganho", "mean"), ganho_desvio=("ganho", "std")).sort_values("ganho_media")
        comp.to_csv(RESULTADOS / "parte2_por_classe.csv")
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(comp.index, comp.ganho_media, xerr=comp.ganho_desvio.fillna(0),
                color=["#218559" if g > 0 else "#ba493e" for g in comp.ganho_media], capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        parcial = "" if (comp.n_seeds == 3).all() else " (PARCIAL)"
        ax.set(xlabel="F1 aluno - F1 piso (media e desvio entre seeds)",
               title="Ganho por classe, 5 rotulos/classe" + parcial)
        fig.tight_layout()
        fig.savefig(ROOT / "figs" / "parte2_ganho_por_classe.png", dpi=160)
        plt.close(fig)
    print(resumo.to_string(index=False), flush=True)
    print(f"Execucoes concluidas: {len(curva)}/10; pendentes: {len(faltam)}", flush=True)
    return faltam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--consolidar", action="store_true", help="gera tabelas e figuras sem treinar")
    args = ap.parse_args()
    faltam = consolidar()
    if args.consolidar:
        return bool(faltam)
    if faltam:
        for nome in ["rotulos_1", "rotulos_5", "rotulos_20", "val", "test_public"]:
            if not (Path(RAIZ) / "labels" / f"{nome}.csv").exists():
                ap.error("Dados ausentes: descompacte pa3_dados.zip ou defina PA3_DADOS")
    print(f"Dispositivo: {DEV}; tau fixo: {TAU}", flush=True)
    for item in faltam:
        metodo, orcamento, seed = item["metodo"], item["orcamento"], item["seed"]
        cmd = [sys.executable, str(ROOT / ("parte0.py" if metodo == "piso" else "aluno.py")),
               "--orcamento", str(orcamento), "--seed", str(seed)]
        cmd += ["--so_b"] if metodo == "piso" else ["--taus", str(TAU)]
        print("Executando:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        consolidar()
    return bool(consolidar())


if __name__ == "__main__":
    sys.exit(main())
