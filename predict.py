# Classifica uma pasta de imagens com o modelo entregue. CPU e sem internet.
# python predict.py --imagens PASTA --saida preds.csv [--rotulos labels/test_public.csv]
import argparse
import os

import pandas as pd
import torch
from PIL import Image

from comum import AQUI, CLASSES, TESTE, metricas, n_params, resnet18

ap = argparse.ArgumentParser()
ap.add_argument("--imagens", "-imagens", required=True)
ap.add_argument("--saida", "-saida", required=True)
ap.add_argument("--rotulos", "-rotulos", help="CSV image_id,label para imprimir F1 macro e acuracia")
args = ap.parse_args()

torch.set_num_threads(os.cpu_count())
model = resnet18(imagenet=False)  # imagenet=False pra nao baixar nada
model.load_state_dict(torch.load(os.path.join(AQUI, "modelo_final.pt"), map_location="cpu", weights_only=True))
model.eval()
print("parametros:", n_params(model))

arqs = sorted(f for f in os.listdir(args.imagens) if f.lower().endswith((".jpg", ".jpeg", ".png")))
preds = []
with torch.no_grad():
    for i in range(0, len(arqs), 32):
        x = torch.stack([TESTE(Image.open(os.path.join(args.imagens, f)).convert("RGB")) for f in arqs[i:i + 32]])
        preds += model(x).argmax(1).tolist()

saida = pd.DataFrame(dict(image_id=[os.path.splitext(f)[0] for f in arqs], label=[CLASSES[k] for k in preds]))
saida.to_csv(args.saida, index=False)
print(len(saida), "imagens em", args.saida)

if args.rotulos:
    d = saida.merge(pd.read_csv(args.rotulos), on="image_id", suffixes=("_pred", ""))
    f1, acc = metricas(d.label, d.label_pred)
    print(f"F1 macro {f1:.4f}, acuracia {acc:.4f}")
