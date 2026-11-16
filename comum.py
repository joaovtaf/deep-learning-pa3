import copy
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models, transforms as T

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.environ.get("PA3_DADOS", AQUI)
CLASSES = pd.read_csv(os.path.join(AQUI, "classes.csv")).label.tolist()  # versionado, o predict nao depende dos dados
IDX = {c: i for i, c in enumerate(CLASSES)}
DEV = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TAM = 320
MEDIA, DESVIO = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def quadrado(img):
    # completa com preto em vez de cortar, o bicho as vezes ta na borda
    w, h = img.size
    s = max(w, h)
    novo = Image.new("RGB", (s, s))
    novo.paste(img, ((s - w) // 2, (s - h) // 2))
    return novo


TREINO = T.Compose([T.Lambda(quadrado), T.RandomResizedCrop(TAM, scale=(0.6, 1), ratio=(0.9, 1.1)),
                    T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2), T.ToTensor(), T.Normalize(MEDIA, DESVIO)])
TESTE = T.Compose([T.Lambda(quadrado), T.Resize((TAM, TAM)), T.ToTensor(), T.Normalize(MEDIA, DESVIO)])


def le(nome):
    d = pd.read_csv(os.path.join(RAIZ, "labels", nome + ".csv"))
    return d.image_id.tolist(), d.label.map(IDX).tolist()


class Imagens(torch.utils.data.Dataset):
    def __init__(self, conj, ids, y, tf):
        self.arqs = [os.path.join(RAIZ, "images", conj, i + ".jpg") for i in ids]
        self.y = list(y)
        self.tf = tf

    def __len__(self):
        return len(self.arqs)

    def __getitem__(self, i):
        return self.tf(Image.open(self.arqs[i]).convert("RGB")), self.y[i]


def loader(ds, bs, shuffle=False, sampler=None):
    workers = int(os.environ.get("PA3_WORKERS", min(8, os.cpu_count() or 1)))
    return torch.utils.data.DataLoader(ds, bs, shuffle=shuffle, sampler=sampler, num_workers=workers,
                                       pin_memory=DEV == "cuda", persistent_workers=workers > 0)


def semente(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def resnet18(imagenet=True):
    m = models.resnet18(weights="IMAGENET1K_V1" if imagenet else None)
    m.fc = nn.Linear(512, len(CLASSES))
    return m


def cnn_pequena():
    camadas, c = [], 3
    for s in [32, 64, 128, 256]:
        camadas += [nn.Conv2d(c, s, 3, padding=1), nn.BatchNorm2d(s), nn.ReLU(), nn.MaxPool2d(2)]
        c = s
    return nn.Sequential(*camadas, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, len(CLASSES)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def metricas(y, p):
    labels = CLASSES if isinstance(np.asarray(y)[0], str) else range(len(CLASSES))
    return f1_score(y, p, average="macro", labels=labels, zero_division=0), accuracy_score(y, p)


@torch.no_grad()
def prediz(model, ds, bs=64):
    model.eval()
    out = []
    for x, _ in loader(ds, bs):
        with torch.autocast("cuda", enabled=DEV == "cuda"):
            out.append(model(x.to(DEV)).float().softmax(1).cpu())
    return torch.cat(out).numpy()


def treina(model, ds_treino, ds_val, epocas, lr, sampler=None, bs=32):
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    dl = loader(ds_treino, bs, shuffle=sampler is None, sampler=sampler)
    yv = np.array(ds_val.y)
    hist, melhor, estado = [], -1, None
    for ep in range(epocas):
        model.train()
        soma, acertos, n = 0.0, 0, 0
        for passo, (x, y) in enumerate(dl, 1):
            x, y = x.to(DEV), y.to(DEV)
            with torch.autocast("cuda", enabled=DEV == "cuda"):
                out = model(x)
                perda = F.cross_entropy(out, y)
            opt.zero_grad()
            scaler.scale(perda).backward()
            scaler.step(opt)
            scaler.update()
            soma += perda.item() * len(y)
            acertos += (out.argmax(1) == y).sum().item()
            n += len(y)
            if passo % 30 == 0:
                print(f"epoca {ep + 1}/{epocas}, lote {passo}/{len(dl)}", flush=True)
        sched.step()
        pv = prediz(model, ds_val)
        f1v, accv = metricas(yv, pv.argmax(1))
        hist.append(dict(epoca=ep + 1, perda_treino=soma / n, acc_treino=acertos / n,
                         perda_val=float(-np.log(pv[np.arange(len(yv)), yv] + 1e-8).mean()), f1_val=f1v, acc_val=accv))
        print(hist[-1], flush=True)
        if f1v > melhor:
            melhor, estado = f1v, copy.deepcopy(model.state_dict())
    model.load_state_dict(estado)
    return pd.DataFrame(hist)
