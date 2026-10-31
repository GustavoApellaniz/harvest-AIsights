# AgriQuotes Assistant

Assistente de perguntas e respostas sobre **cotações de commodities agrícolas** (mercado físico na França e
futuros Euronext, CBOT e Argus). Usa **RAG (Retrieval-Augmented Generation)**: busca as linhas mais
relevantes da tabela de cotações e pede a um LLM (Google Gemini) que responda **somente** com elas, devolvendo
junto as linhas usadas (arquivo e número da linha).

> Projeto de portfólio em estágio inicial. Leia a seção [Limitações](#limitações-conhecidas).
> **Não é recomendação de investimento.** Os dados são de um único dia e podem estar defasados.

## Os dados

Arquivo `data/agritel_quotes.csv`, fornecido pelo autor do projeto (o nome do arquivo indica a Agritel como origem).
É um **retrato de uma única data de referência: 30/10/2026**, sem série histórica.

| Tipo | Mercado | Linhas | Observação |
|---|---|---|---|
| `physical` | Physical | 8 | Preços de mercado físico (ex.: milho entregue em Bordéus, cevada em Rouen), em EUR/tonne |
| `futures` | Euronext | 23 | EUR/tonne |
| `futures` | CBOT | 79 | US cents/bushel, US cents/pound e USD/short ton |
| `futures` | Argus | 21 | EUR/tonne e USD/tonne |

Produtos: trigo, trigo duro, milho, cevada, soja, farelo de soja, óleo de soja, colza, óleo de colza, girassol e ervilhas.
Colunas principais: `final_price`, `variation`, `high`, `low`, `volume`, `open_interest`, `settlement_price`,
preços/volumes de compra e venda (`bid`/`ask`), `taxes`, `surcharges` e `spot_net_price`.

### Como a tabela é transformada

Cada linha do CSV vira **um texto autocontido** (1 linha = 1 trecho indexado), por exemplo:

```text
Cotação de mercado físico, produto Corn (milho), mercado Physical, data de referência 2026-10-30.
Descrição: Milho entregue em Bordéus - Base em julho de 2026. Unidade: EUR/tonne. Preço final: 272.0. ...
```

Decisões e **interpretações** (leia com atenção):

- **Contrato (inferência):** no CSV o campo `contract` aparece como `26 de dezembro`, `27 de março` etc.
  Os números crescem de 26 a 29 junto com os meses de vencimento, então o projeto os interpreta como **ano do contrato**
  (`dezembro de 2026`, `março de 2027`), e não como dia do mês. O valor original é mantido no texto. Esta é uma
  inferência baseada no padrão dos dados, **não** uma informação confirmada pela fonte.
- Datas `DD/MM/AAAA` são convertidas para `AAAA-MM-DD`. O valor `-` é tratado como vazio.
- O significado e a unidade de `variation` (absoluta ou percentual) **não estão documentados** no arquivo; o texto usa o rótulo "Variação" sem interpretar.
- Nomes de produto em inglês ganham a tradução em português (dicionário em `app/services/tabular.py`) para ajudar buscas em português.

## O que é (e o que não é) Machine Learning aqui

| Componente | Tipo | Observação |
|---|---|---|
| Geração da resposta | LLM consumido **via API** (Google Gemini, SDK `google-genai`) | Nenhum modelo próprio é treinado |
| Embeddings das linhas e das perguntas | **Inferência** com modelo pré-treinado (`sentence-transformers`, padrão `intfloat/multilingual-e5-small`) | Sem treinamento nem fine-tuning |
| Busca | Similaridade de cosseno em banco vetorial (ChromaDB) | Recupera as `TOP_K` linhas mais próximas |

Este projeto **não treina nem faz fine-tuning** de modelos.

## Funcionalidades

- `POST /chat`: pergunta → resposta + linhas recuperadas (arquivo, número da linha, distância, texto).
- Histórico por sessão, em memória (últimos N turnos).
- Indexação por linha de comando (`python -m app.ingest`), idempotente.
- Também indexa PDF, TXT e Markdown (carregadores genéricos; o foco e os testes de dados reais são o CSV).
- Validação de entrada, erros em JSON e logs básicos.
- Mitigação de prompt injection: o texto das linhas entra no prompt escapado, em um bloco declarado como dados não confiáveis.
- Avaliação da recuperação (hit@k e MRR) com gerador de perguntas sintéticas.
- Testes automatizados com `pytest`.

## Arquitetura

```
Cliente ──POST /chat──► routes/chat.py ──► services/rag.py
                                              │ 1. vectorstore.search(pergunta)  ──► ChromaDB (+ embeddings)
                                              │ 2. history.get_messages(sessão)
                                              │ 3. prompts.build_user_message(pergunta, linhas)
                                              │ 4. llm.generate(system, mensagens) ──► Gemini API
                                              └ 5. history.add_turn(...)
```

```
app/
├── main.py              # app FastAPI e handler de erros
├── dependencies.py      # monta os serviços uma única vez
├── ingest.py            # CLI de indexação
├── core/                # config.py (variáveis de ambiente), exceptions.py
├── models/schemas.py    # contratos de entrada/saída (Pydantic)
├── routes/              # chat.py, health.py
└── services/
    ├── ingestion.py     # leitura de arquivos, chunking, indexação
    ├── tabular.py       # CSV -> um texto por linha
    ├── embeddings.py    # wrapper do sentence-transformers
    ├── vectorstore.py   # ChromaDB
    ├── history.py       # histórico em memória
    ├── prompts.py       # system prompt e montagem da mensagem
    ├── llm.py           # cliente Gemini + tratamento de falhas
    └── rag.py           # orquestração
eval/                    # evaluate.py, make_questions.py, questions.jsonl
tests/                   # pytest
data/agritel_quotes.csv  # conjunto de dados
```

## Requisitos

- Python 3.12 (versão em que os testes automatizados foram executados; outras versões 3.10+ provavelmente funcionam, mas não foram testadas).
- Uma chave da Gemini API: https://aistudio.google.com/apikey (verifique no site as condições de uso e de cobrança).
- Espaço em disco para o PyTorch (dependência do `sentence-transformers`). Em geral, o modelo de embeddings é baixado do Hugging Face na primeira execução.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env               # depois edite e preencha GEMINI_API_KEY
```

### Variáveis de ambiente

Ver `.env.example`. As principais:

| Variável | Descrição | Padrão |
|---|---|---|
| `GEMINI_API_KEY` | Chave da API (obrigatória para `/chat`) | vazio |
| `LLM_MODEL` | Modelo Gemini. Confira os disponíveis na [documentação](https://ai.google.dev/gemini-api/docs/models) | `gemini-2.5-flash` |
| `LLM_MAX_TOKENS` | Limite de saída. Modelos com "thinking" contam o raciocínio interno aqui; se a resposta vier vazia ou cortada, aumente | `2048` |
| `EMBEDDING_MODEL` | Modelo do `sentence-transformers` | `intfloat/multilingual-e5-small` |
| `TOP_K` | Quantas linhas recuperar | `8` |
| `MAX_HISTORY_TURNS` | Turnos guardados por sessão | `5` |

Para listar os modelos disponíveis para a sua chave:

```python
from google import genai
for m in genai.Client(api_key="SUA_CHAVE").models.list():
    print(m.name)
```

## Execução local

```bash
python -m app.ingest               # indexa data/ (inclui o CSV)
uvicorn app.main:app --reload
```

Documentação interativa (Swagger): http://127.0.0.1:8000/docs

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status e se a chave do LLM está configurada |
| POST | `/chat` | Faz uma pergunta |
| DELETE | `/chat/{session_id}` | Apaga o histórico de uma sessão (204) |

Códigos de erro: `422` entrada inválida · `502` falha no provedor de LLM · `503` chave ausente ou índice vazio.

### Exemplo

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qual o preço final do milho entregue em Bordéus?", "session_id": "demo"}'
```

Formato da resposta. Os campos de `sources` abaixo vêm do CSV; o texto de `answer` é **ilustrativo**
(não foi obtido de uma chamada real ao LLM):

```json
{
  "session_id": "demo",
  "answer": "Texto gerado pelo LLM, citando as linhas como [1].",
  "sources": [
    {
      "ref": 1,
      "source": "agritel_quotes.csv",
      "page": null,
      "row": 3,
      "chunk_id": "agritel_quotes.csv::r3::c0",
      "distance": 0.0,
      "excerpt": "Cotação de mercado físico, produto Corn (milho), mercado Physical, data de referência 2026-10-30. Descrição: Milho entregue em Bordéus - Base em julho de 2026. Unidade: EUR/tonne. Preço final: 272.0. ..."
    }
  ]
}
```

(O valor de `distance` acima é um placeholder; ele depende do modelo de embeddings.)
`sources` lista as linhas **enviadas ao LLM**, não necessariamente todas as citadas na resposta.

## Testes

```bash
pytest
```

Os testes usam um embedder e um LLM **falsos** (determinísticos) e um ChromaDB temporário. Cobrem chunking,
leitura de CSV/PDF/TXT/MD, a transformação das linhas do CSV (incluindo o arquivo real em `data/`), indexação idempotente,
busca, histórico, montagem do prompt (incluindo escape de tags maliciosas), mapeamento de erros do Gemini
(com respostas simuladas), endpoints e a métrica de avaliação. **Não há teste automatizado que chame o modelo de
embeddings real nem a API real do Gemini.**

## Avaliando a recuperação

```bash
python -m eval.make_questions --csv data/agritel_quotes.csv --out eval/questions.jsonl --n 40
python -m eval.evaluate --questions eval/questions.jsonl --k 8
```

Mede **só a recuperação** (a linha certa está nos top-k?), não a qualidade do texto gerado.
As perguntas geradas são **sintéticas**, montadas com os próprios campos da linha (produto, mercado, contrato); por isso
tendem a **superestimar** o desempenho real. Escreva também perguntas à mão (formato em `eval/evaluate.py`).
Nenhum resultado de avaliação com o modelo de embeddings real é publicado aqui ainda.

## Limitações conhecidas

- **O fluxo com o modelo de embeddings real e com a API real do Gemini ainda não foi validado de ponta a ponta.**
- **Perguntas de agregação ou ranking** sobre a tabela inteira ("qual commodity mais subiu?", "média dos preços do trigo") **não são confiáveis**:
  o LLM só vê as `TOP_K` linhas recuperadas, não a tabela completa.
- Dados de **uma única data**: não há histórico, tendências nem comparação entre dias.
- A interpretação do `contract` como ano é uma inferência (ver seção Os dados).
- **Sem autenticação nem limite de requisições**: expor esta API publicamente permitiria que terceiros gastem a sua cota/chave.
- Histórico **em memória**: perdido ao reiniciar; não funciona com múltiplos processos/workers.
- A busca usa só a pergunta atual; perguntas de seguimento ("e o de março?") não são reescritas com o histórico.
- Sem reranking e sem limiar de relevância: sempre são enviadas as `TOP_K` linhas mais próximas, mesmo que irrelevantes.
- PDFs escaneados não são lidos (sem OCR). Trocar `EMBEDDING_MODEL` exige apagar `chroma_db/` e reindexar.
- O escape do texto reduz, mas não elimina, o risco de prompt injection. Não há teste contra um LLM real.
- O modelo padrão (`gemini-2.5-flash`) pode ser descontinuado; confira a [lista de modelos](https://ai.google.dev/gemini-api/docs/models).

## Melhorias futuras

Chamadas de função (function calling) com consultas em pandas para perguntas de agregação · histórico de datas (série temporal) ·
reescrita da pergunta com o histórico · reranking · limiar de distância calibrado com a avaliação · GitHub Actions rodando `pytest` ·
Dockerfile · histórico em SQLite · streaming · autenticação e rate limiting · teste de prompt injection contra o LLM real.
