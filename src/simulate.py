"""Etapa 3: simula o tráfego de produção chamando a API.

Uso: python -m src.simulate --lote normal|crise

Os clientes do lote são enviados à API em pacotes, como faria um sistema
de concessão de crédito. O resultado real (pagou ou não) é gravado em
logs/rotulos.csv, como se chegasse semanas depois pela área de cobrança.
"""

import argparse

import pandas as pd
import requests

from src.config import API_URL, DATA_DIR, ROTULOS, TARGET


def main() -> None:
    """Lê o lote escolhido, envia à API e grava os resultados reais."""
    ap = argparse.ArgumentParser(description="Envia um lote de clientes para a API.")
    ap.add_argument("--lote", choices=["normal", "crise"], required=True)
    ap.add_argument("--tamanho-envio", type=int, default=50, help="clientes por requisição")
    args = ap.parse_args()

    df = pd.read_csv(DATA_DIR / f"producao_{args.lote}.csv")
    rotulos = df[["id_cliente", TARGET]].assign(lote=args.lote)  # guardado à parte
    payload = df.drop(columns=[TARGET]).to_dict(orient="records")  # a API NÃO vê o resultado

    recusas = 0
    for i in range(0, len(payload), args.tamanho_envio):
        pacote = payload[i : i + args.tamanho_envio]
        r = requests.post(f"{API_URL}/predict", json=pacote, timeout=30)
        r.raise_for_status()  # para tudo se a API responder erro
        recusas += sum(x["decisao"] == "recusar" for x in r.json())

    # O resultado real chega "depois": acrescenta ao arquivo (cabeçalho só na primeira vez).
    rotulos.to_csv(ROTULOS, mode="a", header=not ROTULOS.exists(), index=False)
    print(
        f"Lote '{args.lote}': {len(df)} clientes enviados, {recusas} recusados "
        f"({recusas / len(df):.0%}). Rótulos gravados em {ROTULOS.name}."
    )


if __name__ == "__main__":
    main()
