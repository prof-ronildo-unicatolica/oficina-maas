"""Configurações centrais da oficina.

Tudo o que é "ajustável" fica aqui: pastas, arquivos, limites de monitoramento
e o endereço da API. Os outros módulos importam daqui, então basta mudar
um valor neste arquivo para mudar o comportamento do projeto inteiro.
"""

import os
from pathlib import Path

# Pasta raiz do projeto (um nível acima de src/).
ROOT = Path(__file__).resolve().parents[1]

# Pastas compartilhadas entre a API e os jobs (no Docker, viram volumes).
DATA_DIR = ROOT / "data"  # dados brutos e lotes de produção
MODELS_DIR = ROOT / "models"  # uma subpasta por versão de modelo (v1, v2, ...)
LOGS_DIR = ROOT / "logs"  # predições feitas e resultados reais
REPORTS_DIR = ROOT / "reports"  # relatórios de monitoramento em JSON

# Cria as pastas se ainda não existirem, para nenhum módulo quebrar na primeira execução.
for d in (DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

REGISTRY = MODELS_DIR / "registry.json"  # quem é o campeão e o histórico de versões
PREDICOES = LOGS_DIR / "predicoes.jsonl"  # tudo o que a API respondeu (uma linha JSON por cliente)
ROTULOS = LOGS_DIR / "rotulos.csv"  # o "resultado real" (pagou ou não), que chega depois

TARGET = "inadimplente"  # coluna alvo: 1 = mau pagador, 0 = bom pagador
SEED = 42  # semente fixa: todo mundo da turma vê os mesmos números

# Limites de monitoramento (regra de bolso do mercado para PSI):
#   PSI < 0,10 -> estável | 0,10 a 0,25 -> atenção | >= 0,25 -> alerta
PSI_ATENCAO = 0.10
PSI_ALERTA = 0.25
QUEDA_KS_ALERTA = 0.10  # alerta se o KS cair 0,10 ou mais em relação à validação
MARGEM_PROMOCAO = 0.01  # desafiante precisa ganhar por pelo menos 0,01 de AUC

# Endereço da API. Dentro do Docker, os jobs recebem API_URL=http://api:8888
# pelo docker-compose.yml; rodando direto na máquina, vale o padrão abaixo.
API_URL = os.getenv("API_URL", "http://localhost:8888")
