"""Etapa 2: serve o modelo campeão como API (o "as a Service").

Endpoints:
- GET  /health  -> a API está de pé? Quem é o campeão?
- GET  /model   -> histórico de versões do registro
- POST /predict -> recebe uma lista de clientes e devolve probabilidade,
                   decisão e a versão do modelo que decidiu

Documentação interativa (Swagger): http://localhost:8888/docs
"""

import json
import threading
import uuid
from datetime import UTC, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import PREDICOES
from src.model import carregar, ler_registro

app = FastAPI(
    title="Oficina MaaS: score de crédito",
    description="Modelo de inadimplência treinado no German Credit (OpenML).",
)


class Cliente(BaseModel):
    """Contrato de entrada da API: as 20 variáveis de um pedido de crédito.

    O Pydantic valida cada pedido. Se faltar um campo ou vier texto onde se
    espera número, a API responde 422 explicando o erro, antes de chegar ao modelo.
    """

    id_cliente: str | None = Field(
        default=None, description="Identificador opcional, usado para cruzar com o resultado real depois"
    )
    checking_status: str
    duration: float
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_commitment: float
    personal_status: str
    other_parties: str
    residence_since: float
    property_magnitude: str
    age: float
    other_payment_plans: str
    housing: str
    existing_credits: float
    job: str
    num_dependents: float
    own_telephone: str
    foreign_worker: str

    # Exemplo que aparece preenchido no Swagger (/docs).
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "checking_status": "<0",
                    "duration": 24,
                    "credit_history": "existing paid",
                    "purpose": "radio/tv",
                    "credit_amount": 3000,
                    "savings_status": "<100",
                    "employment": "1<=X<4",
                    "installment_commitment": 4,
                    "personal_status": "male single",
                    "other_parties": "none",
                    "residence_since": 2,
                    "property_magnitude": "car",
                    "age": 30,
                    "other_payment_plans": "none",
                    "housing": "own",
                    "existing_credits": 1,
                    "job": "skilled",
                    "num_dependents": 1,
                    "own_telephone": "none",
                    "foreign_worker": "yes",
                }
            ]
        }
    }


# ---------- carregamento do campeão (recarrega sozinho quando ele muda) ----------
# O modelo fica em memória para não ser lido do disco a cada chamada.
# O lock evita que duas requisições simultâneas carreguem/escrevam ao mesmo tempo.
_lock = threading.Lock()
_estado = {"versao": None, "modelo": None}


def modelo_atual():
    """Devolve (versão, modelo) do campeão atual.

    A cada chamada, consulta o registry.json. Se o campeão mudou (por exemplo,
    depois de um retreino), carrega o novo modelo, sem reiniciar a API.
    """
    campeao = ler_registro()["campeao"]
    if campeao is None:
        raise HTTPException(503, "Nenhum modelo treinado ainda. Rode a etapa de treino.")
    with _lock:
        if _estado["versao"] != campeao:
            _estado["modelo"] = carregar(campeao)
            _estado["versao"] = campeao
        return _estado["versao"], _estado["modelo"]


def registrar(linhas: list[dict]) -> None:
    """Acrescenta as predições ao log (uma linha JSON por cliente).

    Esse log é a base da auditoria ("por que este cliente foi recusado?")
    e do monitoramento.
    """
    with _lock, PREDICOES.open("a") as f:
        for linha in linhas:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")


@app.get("/health")
def health():
    """Verificação de saúde: usada pelo healthcheck do Docker e por quem consome a API."""
    return {"status": "ok", "campeao": ler_registro()["campeao"]}


@app.get("/model")
def model_info():
    """Mostra o campeão e os metadados de todas as versões já treinadas."""
    reg = ler_registro()
    return {"campeao": reg["campeao"], "versoes": reg["versoes"]}


@app.post("/predict")
def predict(clientes: list[Cliente]):
    """Pontua uma lista de clientes com o modelo campeão.

    Para cada cliente devolve a probabilidade de inadimplência, a decisão
    (recusar se a probabilidade for >= 0,5) e a versão do modelo usada.
    """
    versao, modelo = modelo_atual()
    df = pd.DataFrame([c.model_dump() for c in clientes])

    # Cliente sem id ganha um id anônimo, para a linha do log ser rastreável.
    ids = [i or f"anon-{uuid.uuid4().hex[:8]}" for i in df.pop("id_cliente")]
    scores = modelo.predict_proba(df)[:, 1]  # coluna 1 = probabilidade de ser inadimplente
    agora = datetime.now(UTC).isoformat(timespec="seconds")

    respostas, log = [], []
    for id_, s, feats in zip(ids, scores, df.to_dict(orient="records"), strict=True):
        r = {
            "id_cliente": id_,
            "prob_inadimplencia": round(float(s), 4),
            "decisao": "recusar" if s >= 0.5 else "aprovar",
            "versao_modelo": versao,
        }
        respostas.append(r)
        log.append({"ts": agora, **r, "features": feats})  # o log guarda também a entrada
    registrar(log)
    return respostas
