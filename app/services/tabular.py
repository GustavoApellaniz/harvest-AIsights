"""Leitura de CSV: cada linha vira um pequeno texto autocontido (1 linha = 1 documento).

O renderizador conhece as colunas do arquivo de cotações (`agritel_quotes.csv`), mas qualquer
coluna desconhecida é mantida como "nome: valor"; assim CSVs de outro formato ainda são indexados.
"""

import csv
import re
from pathlib import Path

from app.services.ingestion import Page

RECORD_TYPE_PT = {"physical": "mercado físico", "futures": "futuros"}

# Tradução apenas para ajudar a busca por perguntas em português; o nome original é mantido.
PRODUCT_PT = {
    "Wheat": "trigo",
    "Durum wheat": "trigo duro",
    "Corn": "milho",
    "Barley": "cevada",
    "Soybean": "soja",
    "Soybean meal": "farelo de soja",
    "Soybean oil": "óleo de soja",
    "Rapeseed": "colza",
    "Rapeseed oil": "óleo de colza",
    "Sunflower": "girassol",
    "Peas": "ervilhas",
}

LABELS = {
    "description": "Descrição",
    "unit": "Unidade",
    "final_price": "Preço final",
    "variation": "Variação",
    "high": "Máxima",
    "low": "Mínima",
    "volume": "Volume",
    "open_interest": "Interesse em aberto",
    "settlement_price": "Preço de liquidação",
    "bid_volume": "Volume de compra (bid)",
    "bid_price": "Preço de compra (bid)",
    "ask_price": "Preço de venda (ask)",
    "ask_volume": "Volume de venda (ask)",
    "last_trade_time": "Hora do último negócio",
    "taxes": "Taxas",
    "surcharges": "Sobretaxas",
    "spot_net_price": "Preço spot líquido",
}
HEADLINE_COLUMNS = {"record_type", "product", "market", "reference_date", "contract"}
INTEGER_COLUMNS = {"volume", "open_interest", "bid_volume", "ask_volume"}

MONTHS_PT = {
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
}  # fmt: skip


def normalize_contract(raw: str) -> str:
    """'26 de dezembro' -> 'dezembro de 2026 (no CSV: "26 de dezembro")'.

    INFERÊNCIA: no arquivo fornecido o número antes do mês cresce de 26 a 29 junto com os
    vencimentos (dez, mar, mai, set...), o que indica ano do contrato (2026...2029), e não o dia do mês.
    O valor original é sempre mantido no texto.
    """
    match = re.fullmatch(r"(\d{2}) de (\w+)", raw.strip(), flags=re.IGNORECASE)
    if match and match.group(2).lower() in MONTHS_PT:
        return f'{match.group(2).lower()} de 20{match.group(1)} (no CSV: "{raw.strip()}")'
    return raw.strip()


def _clean(column: str, value: str | None) -> str:
    value = (value or "").strip()
    if value in {"", "-"}:
        return ""
    if column == "reference_date" and (m := re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value)):
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    if column in INTEGER_COLUMNS and re.fullmatch(r"-?\d+\.0+", value):
        return value.split(".")[0]
    return value


def render_row(row: dict[str, str | None]) -> str:
    values = {k.strip(): _clean(k.strip(), v) for k, v in row.items() if k}
    values = {k: v for k, v in values.items() if v}
    if not values:
        return ""

    headline = []
    record_type = values.get("record_type")
    if record_type:
        headline.append(f"Cotação de {RECORD_TYPE_PT.get(record_type, record_type)}")
    product = values.get("product")
    if product:
        pt = PRODUCT_PT.get(product)
        headline.append(f"produto {product} ({pt})" if pt else f"produto {product}")
    if values.get("market"):
        headline.append(f"mercado {values['market']}")
    if values.get("reference_date"):
        headline.append(f"data de referência {values['reference_date']}")
    if values.get("contract"):
        headline.append(f"contrato {normalize_contract(values['contract'])}")

    fields = [
        f"{LABELS.get(k, k)}: {v}" for k, v in values.items() if k not in HEADLINE_COLUMNS
    ]
    return (", ".join(headline) + ". " if headline else "") + ". ".join(fields) + ("." if fields else "")


def _detect_delimiter(header_line: str) -> str:
    return max(";,\t", key=header_line.count)


def load_csv(path: Path, base_dir: Path) -> list[Page]:
    """Uma `Page` por linha de dados. `row` é o número da linha no arquivo (cabeçalho = linha 1)."""
    source = path.relative_to(base_dir).as_posix()
    with path.open(encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter=_detect_delimiter(header))
        pages = []
        for line_no, row in enumerate(reader, start=2):
            text = render_row(row)
            if text:
                pages.append(Page(source=source, page=None, row=line_no, text=text))
    return pages
