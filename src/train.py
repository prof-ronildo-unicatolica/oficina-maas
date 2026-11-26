"""Etapa 1: treina o primeiro modelo (v1) e o promove a campeão.

Uso: python -m src.train
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, SEED, TARGET
from src.data import preparar_lotes
from src.model import construir_pipeline, ler_registro, metricas, promover, salvar_versao, separar_features


def main() -> None:
    """Prepara os dados (se preciso), treina o v1, mede na validação e o promove."""
    if not (DATA_DIR / "referencia.csv").exists():
        preparar_lotes()
    if ler_registro()["campeao"]:
        print("Já existe um campeão. Rode 'make reset' para recomeçar do zero.")
        return

    df = pd.read_csv(DATA_DIR / "referencia.csv")
    X, num, cat = separar_features(df)
    y = df[TARGET]

    # 75% para treinar, 25% para validar. "stratify" mantém a mesma taxa de maus nos dois.
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    pipe = construir_pipeline(num, cat).fit(X_tr, y_tr)
    m = metricas(y_va, pipe.predict_proba(X_va)[:, 1])  # medido em dados que o modelo não viu
    print(f"Validação: AUC={m['auc']:.3f}  KS={m['ks']:.3f}  Gini={m['gini']:.3f}")

    versao = salvar_versao(pipe, X_tr, m, len(X_tr), origem="treino inicial")
    promover(versao)
    print(f"Modelo {versao} salvo e promovido a campeão.")


if __name__ == "__main__":
    main()
