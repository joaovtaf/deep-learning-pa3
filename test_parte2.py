"""Confere estatisticas, alinhamento dos testes e execucoes ausentes."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import parte2


class Parte2Test(unittest.TestCase):
    def test_reaproveita_legado_e_aceita_nova_execucao(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(parte2, "RESULTADOS", root):
                novo = root / "parte0_b_resnet18_imagenet_r5_s0_test.csv"
                legado = root / "parte0_b_resnet18_imagenet_test.csv"
                self.assertEqual(parte2.arquivo("piso", 5, 0), novo)
                legado.touch()
                self.assertEqual(parte2.arquivo("piso", 5, 0), legado)
                novo.touch()
                self.assertEqual(parte2.arquivo("piso", 5, 0), novo)

    def test_consolidacao_e_pendencias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resultados = root / "resultados"
            resultados.mkdir()
            dados = pd.DataFrame({"image_id": [f"id{i}" for i in range(21)],
                                  "label": parte2.CLASSES, "pred": parte2.CLASSES})
            with patch.multiple(parte2, ROOT=root, RESULTADOS=resultados):
                for metodo in ["piso", "aluno"]:
                    for orcamento, seed in parte2.EXECUCOES:
                        d = dados.copy()
                        if metodo == "aluno" and orcamento == 5 and seed == 1:
                            d["pred"] = np.roll(parte2.CLASSES, 1)
                        # A ordem das linhas nao pode alterar as metricas.
                        d.iloc[::-1].to_csv(parte2.arquivo(metodo, orcamento, seed), index=False)
                dados[["image_id", "pred"]].to_csv(resultados / "professor_test_public.csv", index=False)
                self.assertEqual(parte2.consolidar(), [])
                curva = pd.read_csv(root / "curva_orcamento.csv")
                self.assertEqual(len(curva), 10)
                resumo = pd.read_csv(resultados / "parte2_resumo.csv")
                aluno5 = resumo[(resumo.metodo == "aluno") & (resumo.orcamento == 5)].iloc[0]
                self.assertEqual(aluno5.n_seeds, 3)
                self.assertAlmostEqual(aluno5.f1_media, 2 / 3)
                self.assertAlmostEqual(aluno5.f1_desvio, np.sqrt(1 / 3))
                classes = pd.read_csv(resultados / "parte2_por_classe.csv")
                np.testing.assert_allclose(classes.ganho_media, -1 / 3)
                np.testing.assert_allclose(classes.ganho_desvio, np.sqrt(1 / 3))
                self.assertTrue((root / "figs" / "curva_orcamento.png").exists())
                parte2.arquivo("aluno", 20, 0).unlink()
                self.assertEqual(parte2.consolidar(), [dict(metodo="aluno", orcamento=20, seed=0)])
                self.assertEqual(len(pd.read_csv(root / "curva_orcamento.csv")), 9)

    def test_rejeita_testes_diferentes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resultados = root / "resultados"
            resultados.mkdir()
            with patch.multiple(parte2, ROOT=root, RESULTADOS=resultados):
                for seed in [0, 1]:
                    pd.DataFrame({"image_id": [f"id{seed}"], "label": [parte2.CLASSES[0]],
                                  "pred": [parte2.CLASSES[0]]}).to_csv(
                                      parte2.arquivo("piso", 5, seed), index=False)
                with self.assertRaisesRegex(ValueError, "IDs ou rotulos diferentes"):
                    parte2.consolidar()


if __name__ == "__main__":
    unittest.main()
