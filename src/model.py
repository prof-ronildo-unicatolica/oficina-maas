"""Pipeline de modelagem, métricas e registro de versões.

Repare que a parte de "modelo" de verdade (construir_pipeline) tem poucas
linhas. O resto é o que faz o modelo poder ser operado: versões, metadados,
linha de base para o monitoramento e quem é o campeão.
"""

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, REGISTRY, TARGET

# Colunas que NÃO entram no modelo: o alvo (seria "cola") e o identificador.
COLUNAS_FORA = {TARGET, "id_cliente"}


# ---------------- modelagem ----------------
def separar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Separa as variáveis de entrada (X) e diz quais são numéricas e quais são categóricas."""
    X = df.drop(columns=[c for c in COLUNAS_FORA if c in df.columns])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return X, num, cat


def construir_pipeline(num: list[str], cat: list[str]) -> Pipeline:
    """Monta o pipeline: pré-processamento + regressão logística.

    - Numéricas: padronizadas (média 0, desvio 1).
    - Categóricas: viram colunas 0/1 (one-hot). Categoria nunca vista é ignorada,
      em vez de derrubar a API.
    Guardar o pré-processamento junto com o modelo garante que a produção
    transforme os dados exatamente como o treino transformou.
    """
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])


def metricas(y, score) -> dict:
    """Calcula as métricas clássicas de crédito a partir do resultado real (y) e do score.

    - AUC: 0,5 = sorteio, 1 = perfeito.
    - KS: maior distância entre as curvas de bons e maus (quanto o modelo separa os dois).
    - Gini: 2 * AUC - 1, a mesma informação da AUC em outra escala.
    """
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "auc": round(float(auc), 4),
        "ks": round(float(np.max(tpr - fpr)), 4),
        "gini": round(float(2 * auc - 1), 4),
    }


# ---------------- registro de modelos ----------------
# O registry.json é um "model registry" mínimo:
# {"campeao": "v2", "versoes": [{metadados do v1}, {metadados do v2}]}


def ler_registro() -> dict:
    """Lê o registry.json. Se ainda não existe, devolve um registro vazio."""
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"campeao": None, "versoes": []}


def salvar_registro(reg: dict) -> None:
    """Grava o registro em disco (JSON legível, com acentos)."""
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False))


def proxima_versao() -> str:
    """Devolve o nome da próxima versão: v1, v2, v3..."""
    return f"v{len(ler_registro()['versoes']) + 1}"


def salvar_versao(pipe: Pipeline, X_ref: pd.DataFrame, metricas_validacao: dict, n_treino: int, origem: str) -> str:
    """Salva uma nova versão em models/vN e a registra (sem promover).

    Cada versão guarda três arquivos:
    - model.joblib:   o pipeline treinado;
    - referencia.csv: a "fotografia" dos dados de treino + score, que é a
                      linha de base do monitoramento;
    - metadata.json:  quando, como e com que desempenho foi criada.
    """
    versao = proxima_versao()
    pasta = MODELS_DIR / versao
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, pasta / "model.joblib")

    # Linha de base: sem ela não há com o que comparar a produção.
    ref = X_ref.copy()
    ref["score"] = pipe.predict_proba(X_ref)[:, 1]
    ref.to_csv(pasta / "referencia.csv", index=False)

    meta = {
        "versao": versao,
        "criado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "origem": origem,
        "n_treino": int(n_treino),
        "metricas_validacao": metricas_validacao,
    }
    (pasta / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    reg = ler_registro()
    reg["versoes"].append(meta)
    salvar_registro(reg)
    return versao


def promover(versao: str) -> None:
    """Marca a versão como campeã. A API passa a usá-la na próxima chamada."""
    reg = ler_registro()
    reg["campeao"] = versao
    salvar_registro(reg)


def carregar(versao: str) -> Pipeline:
    """Carrega do disco o pipeline treinado de uma versão."""
    return joblib.load(MODELS_DIR / versao / "model.joblib")


def metadata(versao: str) -> dict:
    """Lê os metadados (data, origem, métricas de validação) de uma versão."""
    return json.loads((MODELS_DIR / versao / "metadata.json").read_text())
