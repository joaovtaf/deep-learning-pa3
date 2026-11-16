# BioCLIP 2 zero-shot no train e no test_public. Roda local na CPU (regra da trilha C).
# O val fica de fora de proposito, o professor nao pode ver.
import os
import sys

import open_clip
import pandas as pd
import torch
from PIL import Image

from comum import AQUI, CLASSES, RAIZ, quadrado

MODELO = "hf-hub:imageomics/bioclip-2"
torch.set_num_threads(os.cpu_count())
model, _, prep = open_clip.create_model_and_transforms(MODELO)
tok = open_clip.get_tokenizer(MODELO)
model.eval()

# prompt fixado antes de olhar qualquer resultado, so o nome da classe
prompts = [f"a camera trap photo of a {c.replace('_', ' ')}." for c in CLASSES]
with torch.no_grad():
    txt = model.encode_text(tok(prompts))
    txt /= txt.norm(dim=-1, keepdim=True)

limite = int(sys.argv[1]) if len(sys.argv) > 1 else None  # so pra testar rapido
os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)
for conj in ["train", "test_public"]:
    ids = sorted(f[:-4] for f in os.listdir(os.path.join(RAIZ, "images", conj)) if f.endswith(".jpg"))[:limite]
    probs = []
    for i in range(0, len(ids), 32):
        x = torch.stack([prep(quadrado(Image.open(os.path.join(RAIZ, "images", conj, j + ".jpg")).convert("RGB")))
                         for j in ids[i:i + 32]])
        with torch.no_grad():
            f = model.encode_image(x)
            f /= f.norm(dim=-1, keepdim=True)
            probs.append((model.logit_scale.exp() * f @ txt.T).softmax(1))
        print(conj, i + len(x), "/", len(ids), flush=True)
    p = torch.cat(probs)
    pd.DataFrame(dict(image_id=ids, pred=[CLASSES[k] for k in p.argmax(1)], conf=p.max(1).values.numpy())) \
        .to_csv(os.path.join(AQUI, "resultados", f"professor_{conj}.csv"), index=False)
