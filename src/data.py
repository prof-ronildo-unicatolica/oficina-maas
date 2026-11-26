"""Carrega o German Credit e prepara os lotes da oficina.

Arquivos gerados em data/:
- referencia.csv      -> histórico usado para treinar o modelo v1
- producao_normal.csv -> clientes "do mês seguinte", parecidos com o histórico
- producao_crise.csv  -> clientes durante uma "crise": dados e comportamento mudam
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from src.config import DATA_DIR, SEED, TARGET

# Cópia local do dataset (já vem no repositório para funcionar sem internet).
BRUTO = DATA_DIR / "credit_g.csv"


def baixar_dados() -> pd.DataFrame:
    """Devolve o German Credit como DataFrame.

    Usa a cópia local se ela existir; senão baixa do OpenML, converte a
    coluna "class" (good/bad) em "inadimplente" (0/1) e salva a cópia local.
    """
    if BRUTO.exists():
        return pd.read_csv(BRUTO)
    print("Baixando German Credit (OpenML 'credit-g')...")
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    df[TARGET] = (df["class"] == "bad").astype(int)  # bad -> 1, good -> 0
    df = df.drop(columns=["class"])
    df.to_csv(BRUTO, index=False)
    return pd.read_csv(BRUTO)


def aplicar_crise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simula uma crise econômica em uma cópia do lote.

    Data drift (os dados de entrada mudam): pedidos maiores e mais longos,
    clientes mais jovens.
    Concept drift (a relação entre dados e resultado muda): crédito para
    consumo (rádio/TV, móveis) passa a ter inadimplência muito maior, algo
    que o modelo antigo nunca viu.
    """
    c = df.copy()
    # --- data drift ---
    c["credit_amount"] = (c["credit_amount"] * 1.8).round()  # pedidos 80% maiores
    c["duration"] = (c["duration"] * 1.5).round()  # prazos 50% mais longos
    c["age"] = (c["age"] - 8).clip(lower=19)  # clientes 8 anos mais jovens (mínimo 19)
    # --- concept drift ---
    consumo = c["purpose"].isin(["radio/tv", "furniture/equipment"])
    vira_mau = consumo & (rng.random(len(c)) < 0.75)  # 75% do consumo passa a não pagar
    c.loc[vira_mau, TARGET] = 1
    return c


def preparar_lotes() -> None:
    """Embaralha o dataset, cria ids de cliente e grava os três lotes em data/."""
    df = baixar_dados().sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.insert(0, "id_cliente", [f"C{i:04d}" for i in range(len(df))])  # C0000, C0001, ...
    rng = np.random.default_rng(SEED)

    # 1000 clientes no total: 600 de histórico, 200 normais e 200 de crise.
    referencia = df.iloc[:600]
    normal = df.iloc[600:800]
    crise = aplicar_crise(df.iloc[800:], rng)

    referencia.to_csv(DATA_DIR / "referencia.csv", index=False)
    normal.to_csv(DATA_DIR / "producao_normal.csv", index=False)
    crise.to_csv(DATA_DIR / "producao_crise.csv", index=False)
    print(f"Lotes prontos: referência={len(referencia)}, normal={len(normal)}, crise={len(crise)}")
    print(
        f"Taxa de inadimplência: referência={referencia[TARGET].mean():.0%}, "
        f"normal={normal[TARGET].mean():.0%}, crise={crise[TARGET].mean():.0%}"
    )


if __name__ == "__main__":
    preparar_lotes()
