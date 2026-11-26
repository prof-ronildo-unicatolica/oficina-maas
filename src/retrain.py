"""Etapa 5: retreina um desafiante e só o promove se ele vencer o campeão.

Uso: python -m src.retrain

Lógica campeão x desafiante:
1. Junta o histórico com os dados recentes de produção (já com o resultado real).
2. Separa 30% dos dados recentes como teste, que NENHUM dos dois modelos viu.
3. Treina o desafiante e compara os dois nesse teste.
4. Promove o desafiante só se ele ganhar por uma margem mínima de AUC.
"""

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, MARGEM_PROMOCAO, PREDICOES, ROTULOS, SEED, TARGET
from src.model import carregar, construir_pipeline, ler_registro, metricas, promover, salvar_versao, separar_features


def dados_recentes() -> pd.DataFrame:
    """Junta o que a API viu em produção com o resultado real que chegou depois."""
    linhas = [json.loads(linha) for linha in PREDICOES.read_text().splitlines() if linha.strip()]
    prod = pd.json_normalize(linhas).drop_duplicates("id_cliente", keep="last")
    prod.columns = [c.removeprefix("features.") for c in prod.columns]
    rot = pd.read_csv(ROTULOS).drop_duplicates("id_cliente", keep="last")
    return prod.merge(rot[["id_cliente", TARGET]], on="id_cliente")  # só clientes com resultado real


def main() -> None:
    """Treina o desafiante, compara com o campeão e promove se ele vencer."""
    campeao = ler_registro()["campeao"]
    if campeao is None or not PREDICOES.exists() or not ROTULOS.exists():
        print("Faltam dados: rode 'make train' e pelo menos uma simulação antes de retreinar.")
        return

    historico = pd.read_csv(DATA_DIR / "referencia.csv")
    recentes = dados_recentes()[historico.columns.tolist()]  # mesmas colunas, mesma ordem

    # Separa um pedaço dos dados recentes que NENHUM dos dois modelos viu no treino
    rec_treino, rec_teste = train_test_split(recentes, test_size=0.3, stratify=recentes[TARGET], random_state=SEED)
    treino = pd.concat([historico, rec_treino], ignore_index=True)

    X_tr, num, cat = separar_features(treino)
    X_te, _, _ = separar_features(rec_teste)
    y_te = rec_teste[TARGET]

    # Mesmo algoritmo, dados mais novos: o que muda é o que o modelo aprendeu.
    desafiante = construir_pipeline(num, cat).fit(X_tr, treino[TARGET])
    m_campeao = metricas(y_te, carregar(campeao).predict_proba(X_te)[:, 1])
    m_desafiante = metricas(y_te, desafiante.predict_proba(X_te)[:, 1])

    print(f"Teste em {len(rec_teste)} clientes recentes (fora do treino):")
    print(f"  Campeão    {campeao}: AUC={m_campeao['auc']:.3f}  KS={m_campeao['ks']:.3f}")
    print(f"  Desafiante    : AUC={m_desafiante['auc']:.3f}  KS={m_desafiante['ks']:.3f}")

    if m_desafiante["auc"] >= m_campeao["auc"] + MARGEM_PROMOCAO:
        origem = f"retreino; venceu {campeao} ({m_desafiante['auc']:.3f} x {m_campeao['auc']:.3f})"
        versao = salvar_versao(desafiante, X_tr, m_desafiante, len(X_tr), origem=origem)
        promover(versao)
        print(f"Desafiante vence. {versao} promovido a campeão; a API passa a usá-lo na próxima chamada.")
    else:
        print(f"Desafiante não superou o campeão pela margem de {MARGEM_PROMOCAO}. {campeao} continua.")


if __name__ == "__main__":
    main()
