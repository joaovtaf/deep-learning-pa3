"""Seleciona o checkpoint no val e reproduz a galeria de Grad-CAM da Parte 3."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from comum import AQUI, RAIZ, CLASSES, DEV, TAM, MEDIA, DESVIO, TESTE, Imagens, le, metricas, resnet18

ROOT = Path(AQUI)
R = ROOT / "resultados"
FIGS = ROOT / "figs" / "parte3"
META = R / "parte3_modelo.json"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def preparar():
    """Reproduz pesos ausentes e escolhe apenas entre checkpoints do orcamento 5."""
    (ROOT / "pesos").mkdir(exist_ok=True)
    linhas = []
    for seed in [0, 1, 2]:
        tag = f"r5_s{seed}"
        if not (ROOT / "pesos" / f"aluno_{tag}_t0.9.pt").exists():
            tag += "_reproduzido"
        peso = ROOT / "pesos" / f"aluno_{tag}_t0.9.pt"
        if not peso.exists():
            from aluno import roda
            ids, y = le("rotulos_5")
            idv, yv = le("val")
            idt, yt = le("test_public")
            prof = pd.read_csv(R / "professor_train.csv")
            prof = prof[~prof.image_id.isin(ids)]
            va = Imagens("val", idv, yv, TESTE)
            te = Imagens("test_public", idt, yt, TESTE)
            te.ids = idt
            print(f"Reproduzindo seed {seed}; os arquivos originais serao preservados.", flush=True)
            roda(0.9, argparse.Namespace(seed=seed, epocas=6, amostras=3000), tag, ids, y, prof, va, te)
        v = pd.read_csv(R / f"aluno_{tag}_t0.9_val.csv").iloc[0]
        if not peso.exists():
            raise FileNotFoundError(peso)
        linhas.append(dict(seed=seed, tag=tag, f1_val=float(v.f1_val), epoca_val=int(v.epoca_val),
                           pesos=str(peso.relative_to(ROOT))))
    selecao = pd.DataFrame(linhas)
    idx = selecao.f1_val.idxmax()
    selecao["escolhido"] = selecao.index == idx
    selecao.to_csv(R / "parte3_selecao.csv", index=False)
    escolhido = linhas[idx]
    final = ROOT / "modelo_final.pt"
    shutil.copyfile(ROOT / escolhido["pesos"], final)
    teste = pd.read_csv(R / f"aluno_{escolhido['tag']}_t0.9_test.csv", dtype={"image_id": str})
    gab = pd.read_csv(Path(RAIZ) / "labels" / "test_public.csv", dtype={"image_id": str})
    if teste.image_id.duplicated().any() or not teste.set_index("image_id").label.sort_index().equals(
            gab.set_index("image_id").label.sort_index()):
        raise ValueError("Predicoes e gabarito do test_public diferem")
    f1, acc = metricas(teste.label, teste.pred)
    teste.to_csv(R / "parte3_test.csv", index=False)
    meta = dict(**escolhido, orcamento=5, tau=0.9, pesos_final="modelo_final.pt",
                sha256=sha256(final), f1_macro=f1, acuracia=acc, tamanho_entrada=TAM,
                camada_gradcam="layer4", alvo_gradcam="logit da classe prevista",
                seeds_reproduzidas=[s["seed"] for s in linhas if s["tag"].endswith("reproduzido")])
    META.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(selecao.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def gradcam(model, x):
    """Grad-CAM do logit previsto na ultima camada convolucional, sem alterar pesos."""
    ativacoes = []
    hook = model.layer4.register_forward_hook(lambda _m, _i, out: ativacoes.append(out))
    try:
        model.eval()
        with torch.enable_grad():
            logits = model(x)
            classe = int(logits[0].argmax())
            a = ativacoes[0]
            grad, = torch.autograd.grad(logits[0, classe], a)
            pesos = grad.mean(dim=(2, 3), keepdim=True)
            mapa = (a * pesos).sum(dim=1, keepdim=True).relu()
            mapa = F.interpolate(mapa, size=x.shape[-2:], mode="bilinear", align_corners=False)
            mapa = mapa / mapa.amax().clamp_min(1e-12)
            probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
        return classe, probs, mapa[0, 0].detach().cpu().numpy()
    finally:
        hook.remove()


def carregar():
    meta = json.loads(META.read_text())
    peso = ROOT / meta["pesos_final"]
    if sha256(peso) != meta["sha256"]:
        raise ValueError("O checkpoint mudou desde a selecao; execute --preparar novamente")
    model = resnet18(imagenet=False)
    model.load_state_dict(torch.load(peso, map_location="cpu", weights_only=True))
    return model.to(DEV).eval(), meta


def explicar(model, linha):
    path = Path(RAIZ) / "images" / "test_public" / f"{linha.image_id}.jpg"
    with Image.open(path) as img:
        img = img.convert("RGB")
        original_w, original_h = img.size
        x = TESTE(img).unsqueeze(0).to(DEV)
    classe, probs, mapa = gradcam(model, x)
    if CLASSES[classe] != linha.pred or not np.isclose(probs[classe], linha.conf, atol=1e-4):
        raise ValueError(f"Checkpoint e predicoes divergem: {linha.image_id}")
    vis = x[0].detach().cpu() * torch.tensor(DESVIO)[:, None, None] + torch.tensor(MEDIA)[:, None, None]
    vis = vis.permute(1, 2, 0).clamp(0, 1).numpy()
    if not np.isfinite(mapa).all() or mapa.max() == 0:
        raise ValueError(f"Grad-CAM vazio ou invalido: {linha.image_id}")
    return vis, mapa, dict(largura_original=original_w, altura_original=original_h,
                           conf_gradcam=float(probs[classe]),
                           pico_x=int(mapa.argmax() % TAM), pico_y=int(mapa.argmax() // TAM))


def candidatos(model):
    """Exemplos variados por classe; selecao apenas diagnostica, sem ajustar o modelo."""
    d = pd.read_csv(R / "parte3_test.csv", dtype={"image_id": str})
    for tipo, s in [("erros", d[d.label != d.pred]), ("acertos", d[d.label == d.pred])]:
        s = s.sort_values(["conf", "image_id"], ascending=[False, True]).groupby("label", sort=True).head(2)
        s = s.sort_values(["label", "conf"], ascending=[True, False]).reset_index(drop=True)
        s.to_csv(R / f"parte3_candidatos_{tipo}.csv", index=False)
        for pagina, inicio in enumerate(range(0, len(s), 12), 1):
            lote = s.iloc[inicio:inicio + 12]
            fig, axs = plt.subplots(3, 4, figsize=(16, 12))
            for ax in axs.flat:
                ax.axis("off")
            for ax, linha in zip(axs.flat, lote.itertuples()):
                vis, mapa, _ = explicar(model, linha)
                ax.imshow(vis)
                ax.imshow(mapa, cmap="inferno", vmin=0, vmax=1, alpha=0.6 * mapa)
                ax.set_title(f"{linha.image_id}\n{linha.label} -> {linha.pred}\nconf={linha.conf:.3f}", fontsize=10)
            fig.tight_layout()
            path = FIGS / f"candidatos_{tipo}_{pagina}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            print(path.relative_to(ROOT), flush=True)


def paineis(axs, vis, mapa):
    axs[0].imshow(vis)
    axs[1].imshow(mapa, cmap="inferno", vmin=0, vmax=1)
    axs[2].imshow(vis)
    axs[2].imshow(mapa, cmap="inferno", vmin=0, vmax=1, alpha=0.6 * mapa)
    for ax, titulo in zip(axs, [f"Entrada do modelo ({TAM} x {TAM})", "Grad-CAM (0 a 1)", "Sobreposicao"]):
        ax.set_title(titulo, fontsize=11)
        ax.axis("off")


def galeria(model, meta):
    exemplos = pd.read_csv(ROOT / "parte3_exemplos.csv", dtype={"image_id": str})
    if exemplos.image_id.duplicated().any() or exemplos.tipo.value_counts().to_dict() != {"erro": 3, "acerto": 5}:
        raise ValueError("Selecione exatamente 3 erros e 5 acertos distintos")
    d = pd.read_csv(R / "parte3_test.csv", dtype={"image_id": str})
    d = exemplos.merge(d, on="image_id", validate="one_to_one", how="left")
    if d.isna().any().any() or not (d.tipo.eq("acerto") == d.label.eq(d.pred)).all():
        raise ValueError("Tipo, ID ou diagnostico invalido nos exemplos")
    dados = []
    for linha in d.itertuples():
        vis, mapa, medidas = explicar(model, linha)
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
        paineis(axs, vis, mapa)
        fig.suptitle(f"{linha.tipo.upper()} | {linha.image_id}\nVerdade: {linha.label} | Previsao: {linha.pred} | Conf.: {linha.conf:.2%}", fontsize=12)
        fig.text(0.5, 0.02, textwrap.fill(linha.diagnostico, 140), ha="center", va="bottom", fontsize=10)
        fig.tight_layout(rect=(0, 0.14, 1, 0.88))
        nome = f"{linha.tipo}_{linha.image_id}.png"
        fig.savefig(FIGS / nome, dpi=160)
        plt.close(fig)
        dados.append(dict(image_id=linha.image_id, tipo=linha.tipo, label=linha.label, pred=linha.pred,
                          conf=linha.conf, foco=linha.foco, diagnostico=linha.diagnostico, figura=f"figs/parte3/{nome}",
                          **medidas, modelo_sha256=meta["sha256"]))
    for tipo, nome in [("erro", "erros"), ("acerto", "acertos")]:
        indices = [i for i, item in enumerate(dados) if item["tipo"] == tipo]
        cartoes = []
        for i in indices:
            with Image.open(ROOT / dados[i]["figura"]) as card:
                cartoes.append(np.asarray(card.convert("RGB")))
        Image.fromarray(np.concatenate(cartoes, axis=0)).save(FIGS / f"galeria_{nome}.png")
    pd.DataFrame(dados).to_csv(R / "parte3_galeria.csv", index=False)
    print("Galeria completa: 3 erros, 5 acertos, mapas e diagnosticos.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preparar", action="store_true", help="recupera pesos e seleciona o modelo no val")
    ap.add_argument("--candidatos", action="store_true", help="gera contatos para inspecao dos exemplos")
    args = ap.parse_args()
    R.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    if args.preparar:
        preparar()
        return
    model, meta = carregar()
    if args.candidatos:
        candidatos(model)
    else:
        galeria(model, meta)


if __name__ == "__main__":
    main()
