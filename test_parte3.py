"""Verifica Grad-CAM com um modelo cuja explicacao pode ser calculada a mao."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import parte3
from parte3 import gradcam


class ModeloTeste(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Conv2d(1, 2, 1, bias=False)
        self.head = nn.Linear(2, 2)
        with torch.no_grad():
            self.layer4.weight.copy_(torch.tensor([1., -1.]).reshape(2, 1, 1, 1))
            self.head.weight.copy_(torch.eye(2))
            self.head.bias.zero_()

    def forward(self, x):
        return self.head(self.layer4(x).mean((2, 3)))


class GradCAMTest(unittest.TestCase):
    def test_selecao_usa_val_e_preserva_resultados_originais(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = root / "resultados"
            r.mkdir()
            (root / "pesos").mkdir()
            (root / "labels").mkdir()
            c0, c1 = parte3.CLASSES[:2]
            pd.DataFrame({"image_id": ["id"], "label": [c0]}).to_csv(root / "labels/test_public.csv", index=False)
            for seed, tag, val in [(0, "r5_s0_reproduzido", .8), (1, "r5_s1", .6), (2, "r5_s2", .9)]:
                (root / "pesos" / f"aluno_{tag}_t0.9.pt").write_bytes(f"peso{seed}".encode())
                pd.DataFrame([dict(f1_val=val, epoca_val=3)]).to_csv(r / f"aluno_{tag}_t0.9_val.csv", index=False)
                # A seed 2 perde no teste, mas deve ser escolhida pelo val.
                pd.DataFrame([dict(image_id="id", label=c0, pred=c1 if seed == 2 else c0, conf=.9)]).to_csv(
                    r / f"aluno_{tag}_t0.9_test.csv", index=False)
            antigo = r / "aluno_r5_s0_t0.9_val.csv"
            antigo.write_text("f1_val,epoca_val\n0.99,6\n")
            with patch.multiple(parte3, ROOT=root, R=r, META=r / "modelo.json", RAIZ=str(root)), patch("builtins.print"):
                parte3.preparar()
            meta = json.loads((r / "modelo.json").read_text())
            self.assertEqual(meta["seed"], 2)
            self.assertEqual(meta["f1_macro"], 0)
            self.assertEqual(meta["seeds_reproduzidas"], [0])
            self.assertEqual((root / "pesos/modelo_final.pt").read_bytes(), b"peso2")
            self.assertEqual(antigo.read_text(), "f1_val,epoca_val\n0.99,6\n")

    def test_mapa_conhecido_sem_alterar_modelo(self):
        model = ModeloTeste()
        estado = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.tensor([[[[0., 1.], [2., 4.]]]])
        classe, probs, mapa = gradcam(model, x)
        self.assertEqual(classe, 0)
        self.assertGreater(probs[0], probs[1])
        np.testing.assert_allclose(mapa, [[0, 0.25], [0.5, 1]])
        self.assertEqual(len(model.layer4._forward_hooks), 0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, estado[k])
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_gradiente_zero_nao_produz_nan(self):
        model = ModeloTeste()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias[0] = 10
        _, _, mapa = gradcam(model, torch.ones(1, 1, 2, 2))
        np.testing.assert_array_equal(mapa, np.zeros((2, 2)))
        self.assertTrue(np.isfinite(mapa).all())
        self.assertEqual(len(model.layer4._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
