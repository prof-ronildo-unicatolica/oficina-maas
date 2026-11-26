# Oficina: Model as a Service na prática

Um mini **Model as a Service** de ponta a ponta: treinamos um modelo de crédito, publicamos como API, monitoramos a produção, detectamos uma crise e retreinamos com a lógica **campeão × desafiante**.

> Você não precisa saber Python para acompanhar. Todos os comandos estão prontos abaixo. Quem programa pode abrir os arquivos em `src/`: cada função tem um comentário explicando o que faz.

## Requisitos

- **Docker** com Compose v2. Confira com `docker compose version`.
- **`make`** (opcional). Cada atalho tem o comando `docker compose` equivalente ao lado.

## Como rodar

```bash
# 0. Baixar o projeto
git clone https://github.com/prof-ronildo-unicatolica/oficina-maas.git
cd oficina-maas

# 1. Construir a imagem (5 a 10 min na primeira vez)
make build        # ou: docker compose build

# 2. Preparar os dados e treinar o modelo v1
make train        # ou: docker compose run --rm jobs python -m src.train

# 3. Subir a API
make up           # ou: docker compose up -d api
```

A API fica em **http://localhost:8888**, e a documentação interativa (Swagger) em **http://localhost:8888/docs**.

```bash
curl http://localhost:8888/health
# {"status":"ok","campeao":"v1"}
```

## O roteiro da oficina

| Passo | Atalho | Comando equivalente | O que acontece |
|---|---|---|---|
| 1 | `make train` | `docker compose run --rm jobs python -m src.train` | Treina o v1 e o promove a campeão |
| 2 | `make up` | `docker compose up -d api` | Sobe a API na porta 8888 |
| 3 | `make normal` | `docker compose run --rm jobs python -m src.simulate --lote normal` | Envia 200 clientes "normais" |
| 4 | `make monitor` | `docker compose run --rm jobs python -m src.monitor` | Mede o drift → **STATUS: OK** |
| 5 | `make crise` | `docker compose run --rm jobs python -m src.simulate --lote crise` | Envia 200 clientes durante uma crise |
| 6 | `make monitor` | `docker compose run --rm jobs python -m src.monitor` | Mede de novo → **STATUS: ALERTA** |
| 7 | `make retrain` | `docker compose run --rm jobs python -m src.retrain` | Desafiante vence, e o **v2** vira campeão |

Para acompanhar os logs da API: `make logs` (ou `docker compose logs -f api`).

## Parar e limpar

```bash
make down         # ou: docker compose down   → para os containers
make reset        # apaga lotes, modelos, logs e relatórios e para tudo (recomeça do zero)
```

## Arquitetura

Uma única imagem Docker (`oficina-maas`) é usada de dois jeitos: o serviço **`api`** fica sempre no ar, e o container **`jobs`** roda as tarefas avulsas. Os dois compartilham as pastas do projeto por volumes.

```mermaid
flowchart LR
    subgraph docker["Docker Compose · imagem oficina-maas"]
        API["api<br/>FastAPI :8888<br/>sempre usa o CAMPEÃO"]
        subgraph jobs["jobs (execuções avulsas)"]
            TRAIN["train.py"]
            SIM["simulate.py"]
            MON["monitor.py"]
            RET["retrain.py"]
        end
    end

    subgraph vol["Volumes compartilhados"]
        DATA[("data/<br/>lotes CSV")]
        MODELS[("models/<br/>v1, v2…<br/>registry.json")]
        LOGS[("logs/<br/>predicoes.jsonl<br/>rotulos.csv")]
        REP[("reports/<br/>monitoramento_*.json")]
    end

    USER(["Você / Swagger<br/>localhost:8888"]) -->|HTTP| API
    TRAIN -->|lê| DATA
    TRAIN -->|salva v1 e promove| MODELS
    SIM -->|lê lote| DATA
    SIM -->|POST /predict| API
    SIM -->|resultado real| LOGS
    API -->|lê campeão| MODELS
    API -->|grava cada predição| LOGS
    MON -->|lê| LOGS
    MON -->|linha de base| MODELS
    MON -->|OK / ALERTA| REP
    RET -->|lê| LOGS
    RET -->|salva vN e promove se vencer| MODELS
```

## Fluxo da aplicação

```mermaid
flowchart TD
    A["make train<br/>v1 vira campeão"] --> B["make up<br/>API servindo o campeão"]
    B --> C["make normal<br/>clientes parecidos com o treino"]
    C --> D{"make monitor<br/>PSI ≥ 0,25 ou KS caiu ≥ 0,10?"}
    D -->|não| OK["STATUS: OK<br/>segue operando"]
    OK --> E["make crise<br/>pedidos maiores, clientes mais jovens,<br/>consumo passa a não pagar"]
    E --> F{"make monitor<br/>PSI ≥ 0,25 ou KS caiu ≥ 0,10?"}
    F -->|sim| AL["STATUS: ALERTA<br/>recomenda-se retreino"]
    AL --> G["make retrain<br/>treina desafiante com dados recentes"]
    G --> H{"desafiante ganha por<br/>≥ 0,01 de AUC?"}
    H -->|sim| P["promove v2 a campeão"]
    H -->|não| K["campeão continua"]
    P --> B
    K --> B
```

## Diagrama de classes

O código é organizado em módulos com funções (não há classes de verdade, exceto `Cliente`). O diagrama mostra cada módulo como uma "classe", com suas funções e as dependências entre eles.

```mermaid
classDiagram
    direction LR

    class config {
        <<módulo>>
        DATA_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR
        REGISTRY : models/registry.json
        PREDICOES : logs/predicoes.jsonl
        ROTULOS : logs/rotulos.csv
        PSI_ATENCAO = 0.10
        PSI_ALERTA = 0.25
        QUEDA_KS_ALERTA = 0.10
        MARGEM_PROMOCAO = 0.01
        API_URL
    }

    class data {
        <<módulo>>
        baixar_dados() DataFrame
        aplicar_crise(df, rng) DataFrame
        preparar_lotes()
    }

    class model {
        <<módulo>>
        separar_features(df) X, num, cat
        construir_pipeline(num, cat) Pipeline
        metricas(y, score) dict
        ler_registro() dict
        salvar_registro(reg)
        proxima_versao() str
        salvar_versao(pipe, X_ref, metricas, n, origem) str
        promover(versao)
        carregar(versao) Pipeline
        metadata(versao) dict
    }

    class api {
        <<módulo FastAPI>>
        modelo_atual() versao, modelo
        registrar(linhas)
        GET /health
        GET /model
        POST /predict(clientes)
    }

    class Cliente {
        <<Pydantic BaseModel>>
        id_cliente : str (opcional)
        checking_status : str
        duration : float
        credit_amount : float
        age : float
        purpose : str
        + outras 14 variáveis
    }

    class train {
        <<job>>
        main()
    }
    class simulate {
        <<job>>
        main()
    }
    class monitor {
        <<job>>
        psi(ref, atual, bins) float
        classificar(valor) str
        carregar_producao(versao, ultimos) DataFrame
        main()
    }
    class retrain {
        <<job>>
        dados_recentes() DataFrame
        main()
    }

    api *-- Cliente : valida entrada
    api ..> model : carregar, ler_registro
    train ..> data : preparar_lotes
    train ..> model : treina, salva, promove
    monitor ..> model : metadata, metricas
    retrain ..> model : treina, compara, promove
    simulate ..> api : HTTP POST /predict
    data ..> config
    model ..> config
    api ..> config
    monitor ..> config
    retrain ..> config
    simulate ..> config
```

## Diagrama de sequência

O ciclo completo de vida do modelo, do treino ao retreino.

```mermaid
sequenceDiagram
    autonumber
    actor U as Você (terminal)
    participant J as jobs
    participant A as API :8888
    participant M as models/<br/>(registry.json)
    participant L as logs/

    Note over U,L: 1. Treino
    U->>J: make train
    J->>J: preparar_lotes() e treina v1
    J->>M: salvar_versao(v1) e promover(v1)

    Note over U,L: 2. Servir
    U->>A: make up
    U->>A: POST /predict [cliente]
    A->>M: ler_registro() → campeão = v1
    A->>A: carrega v1 (só se mudou)
    A->>L: registrar(predição + versão)
    A-->>U: prob, decisão, versao_modelo=v1

    Note over U,L: 3. Produção e monitoramento
    U->>J: make normal / make crise
    loop pacotes de 50 clientes
        J->>A: POST /predict
        A->>L: grava predições
        A-->>J: decisões
    end
    J->>L: grava resultado real (rotulos.csv)
    U->>J: make monitor
    J->>M: lê linha de base (referencia.csv)
    J->>L: lê predições e rótulos
    J-->>U: STATUS: OK ou ALERTA (+ reports/*.json)

    Note over U,L: 4. Retreino campeão × desafiante
    U->>J: make retrain
    J->>L: dados_recentes()
    J->>J: treina desafiante, compara AUC em teste inédito
    alt desafiante vence por ≥ 0,01
        J->>M: salvar_versao(v2) e promover(v2)
    else não vence
        J-->>U: campeão continua
    end

    Note over U,L: 5. A API troca de modelo sozinha
    U->>A: POST /predict
    A->>M: ler_registro() → campeão = v2
    A->>A: recarrega v2, sem reiniciar
    A-->>U: versao_modelo=v2
```

## Estrutura do projeto

```text
oficina-maas/
├── src/
│   ├── config.py     # pastas, arquivos e limites (PSI, KS, margem)
│   ├── data.py       # carrega o dataset e cria os lotes normal e crise
│   ├── model.py      # pipeline, métricas e registro de versões
│   ├── train.py      # etapa 1: treina o v1
│   ├── api.py        # etapa 2: API FastAPI
│   ├── simulate.py   # etapa 3: simula clientes chamando a API
│   ├── monitor.py    # etapa 4: PSI, KS e AUC → OK / ALERTA
│   └── retrain.py    # etapa 5: campeão × desafiante
├── data/             # credit_g.csv (vem no projeto) + lotes gerados
├── models/           # v1/, v2/... e registry.json (gerado)
├── logs/             # predicoes.jsonl e rotulos.csv (gerado)
├── reports/          # relatórios de monitoramento (gerado)
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml    # configuração do Ruff
```

## Qualidade de código (Ruff)

O projeto usa o [Ruff](https://docs.astral.sh/ruff/) como linter e formatador. Ele roda em um container, sem instalar nada:

```bash
make lint         # verifica estilo, imports e erros comuns
make format       # corrige o que der automaticamente e formata
```

## Trocar a porta

Se a 8888 estiver ocupada:

```bash
cp .env.example .env    # e mude para API_PORT=8889
# ou, só desta vez:
API_PORT=8889 make up
```

Os jobs continuam funcionando, porque falam com a API pela rede interna do Docker.

## Dataset

German Credit Data (Statlog), Prof. Hans Hofmann, Universidade de Hamburgo, 1994. Fonte: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), licença CC BY 4.0. Já vem em `data/credit_g.csv`, para funcionar sem internet.
