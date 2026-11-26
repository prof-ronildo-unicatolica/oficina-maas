"""Etapa 4: monitora o campeão comparando a produção com a linha de base.

Uso: python -m src.monitor [--ultimos 200]

Duas perguntas:
1) Estabilidade: os dados que chegam se parecem com os do treino? (PSI)
2) Desempenho: o modelo continua acertando? (KS e AUC, quando o resultado real chega)
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PREDICOES, PSI_ALERTA, PSI_ATENCAO, QUEDA_KS_ALERTA, REPORTS_DIR, ROTULOS, TARGET
from src.model import ler_registro, metadata, metricas


def psi(ref: pd.Series, atual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: quanto a distribuição atual se afastou da referência.

    Divide os valores em faixas (decis da referência, para variáveis numéricas;
    cada categoria, para as categóricas), compara a proporção de cada faixa
    nos dois conjuntos e soma (atual - ref) * ln(atual / ref).
    0 = distribuições iguais; quanto maior, mais diferente.
    """
    eps = 1e-4  # evita divisão por zero e log(0) em faixas vazias
    if pd.api.types.is_numeric_dtype(ref):
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        cortes[0], cortes[-1] = -np.inf, np.inf  # pontas abertas pegam valores fora do intervalo do treino
        p_ref = pd.cut(ref, cortes).value_counts(normalize=True, sort=False)
        p_atu = pd.cut(atual, cortes).value_counts(normalize=True, sort=False)
    else:
        cats = sorted(set(ref.astype(str)) | set(atual.astype(str)))
        p_ref = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        p_atu = atual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    p_ref, p_atu = p_ref.values + eps, p_atu.values + eps
    return float(np.sum((p_atu - p_ref) * np.log(p_atu / p_ref)))


def classificar(valor: float) -> str:
    """Traduz um PSI em status: ok, atenção ou ALERTA."""
    if valor >= PSI_ALERTA:
        return "ALERTA"
    if valor >= PSI_ATENCAO:
        return "atenção"
    return "ok"


def carregar_producao(versao: str, ultimos: int) -> pd.DataFrame:
    """Lê o log da API e devolve as últimas predições feitas por esta versão, uma coluna por variável."""
    linhas = [json.loads(linha) for linha in PREDICOES.read_text().splitlines() if linha.strip()]
    prod = pd.json_normalize(linhas)  # {"features": {"age": 30}} vira a coluna "features.age"
    prod = prod[prod["versao_modelo"] == versao].tail(ultimos)
    prod.columns = [c.removeprefix("features.") for c in prod.columns]
    return prod.rename(columns={"prob_inadimplencia": "score"})


def main() -> None:
    """Calcula PSI e desempenho, decide entre OK e ALERTA e salva um relatório em reports/."""
    ap = argparse.ArgumentParser(description="Monitora o modelo campeão.")
    ap.add_argument("--ultimos", type=int, default=200, help="quantas predições recentes avaliar")
    args = ap.parse_args()

    versao = ler_registro()["campeao"]
    if versao is None:
        print("Nenhum modelo treinado ainda. Rode 'make train' primeiro.")
        return
    if not PREDICOES.exists():
        print("Ainda não há predições. Rode a simulação primeiro.")
        return
    ref = pd.read_csv(MODELS_DIR / versao / "referencia.csv")
    prod = carregar_producao(versao, args.ultimos)
    if prod.empty:
        print(f"Nenhuma predição recente do modelo {versao}.")
        return

    # 1) Estabilidade: PSI de cada variável e do próprio score
    linhas = [{"variavel": c, "psi": round(psi(ref[c], prod[c]), 3)} for c in ref.columns]
    tabela = pd.DataFrame(linhas).sort_values("psi", ascending=False)
    tabela["status"] = tabela["psi"].map(classificar)

    # 2) Desempenho: só dá para medir quando o resultado real já chegou
    base = metadata(versao)["metricas_validacao"]
    desempenho = None
    if ROTULOS.exists():
        rot = pd.read_csv(ROTULOS).drop_duplicates("id_cliente", keep="last")
        com_rotulo = prod.merge(rot[["id_cliente", TARGET]], on="id_cliente")
        if com_rotulo[TARGET].nunique() == 2:  # AUC/KS exigem bons E maus
            desempenho = metricas(com_rotulo[TARGET], com_rotulo["score"])

    # 3) Decisão: junta os motivos de alerta
    psi_score = float(tabela.loc[tabela["variavel"] == "score", "psi"].iloc[0])
    motivos = []
    if psi_score >= PSI_ALERTA:
        motivos.append(f"PSI do score = {psi_score:.2f}")
    instaveis = tabela[(tabela["status"] == "ALERTA") & (tabela["variavel"] != "score")]["variavel"].tolist()
    if instaveis:
        motivos.append("variáveis instáveis: " + ", ".join(instaveis))
    if desempenho and base["ks"] - desempenho["ks"] >= QUEDA_KS_ALERTA:
        motivos.append(f"KS caiu de {base['ks']:.2f} para {desempenho['ks']:.2f}")
    status = "ALERTA: recomenda-se retreino" if motivos else "OK"

    # 4) Saída no terminal
    print(f"\n=== Monitoramento do modelo {versao} ({len(prod)} predições recentes) ===")
    print(tabela.head(8).to_string(index=False))
    print(f"\nDesempenho na validação: KS={base['ks']:.3f}  AUC={base['auc']:.3f}")
    if desempenho:
        print(f"Desempenho em produção: KS={desempenho['ks']:.3f}  AUC={desempenho['auc']:.3f}")
    else:
        print("Desempenho em produção: aguardando resultados reais (rótulos).")
    print(f"\nSTATUS: {status}")
    for m in motivos:
        print(f"  - {m}")

    # 5) Relatório em JSON (o que um painel ou um alerta automático consumiria)
    relatorio = {
        "versao": versao,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "n_predicoes": len(prod),
        "status": status,
        "motivos": motivos,
        "psi": tabela.to_dict(orient="records"),
        "desempenho_validacao": base,
        "desempenho_producao": desempenho,
    }
    nome = REPORTS_DIR / f"monitoramento_{datetime.now():%Y%m%d_%H%M%S}.json"
    nome.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))
    print(f"\nRelatório salvo em reports/{nome.name}")


if __name__ == "__main__":
    main()
