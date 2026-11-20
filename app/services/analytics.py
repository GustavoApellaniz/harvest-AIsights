"""Agregações exatas sobre o CSV de cotações (sem LLM e sem pandas)."""

import csv
from collections import defaultdict
from pathlib import Path

from app.core.exceptions import DataFileNotFoundError

QUOTES_FILENAME = "agritel_quotes.csv"


def _to_float(value: str | None) -> float | None:
    """Converte o preço; vazio, '-' ou texto inválido viram None (linha ignorada)."""
    if value is None:
        return None
    try:
        return float(value.strip().replace(",", "."))
    except ValueError:
        return None


def summarize_quotes(
    data_dir: Path, product: str | None = None, market: str | None = None
) -> list[dict]:
    """Estatísticas de `final_price` por produto + mercado + unidade.

    Unidades nunca são misturadas: cada combinação distinta gera seu próprio grupo.
    Os filtros `product` e `market` ignoram maiúsculas/minúsculas e comparam o valor exato.
    """
    path = data_dir / QUOTES_FILENAME
    if not path.is_file():
        raise DataFileNotFoundError(f"Arquivo de cotações não encontrado: {path}")

    prices: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if product and (row.get("product") or "").strip().lower() != product.strip().lower():
                continue
            if market and (row.get("market") or "").strip().lower() != market.strip().lower():
                continue
            price = _to_float(row.get("final_price"))
            if price is None:
                continue
            key = ((row.get("product") or "").strip(), (row.get("market") or "").strip(), (row.get("unit") or "").strip())
            prices[key].append(price)

    return [
        {
            "product": p,
            "market": m,
            "unit": u,
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 4),
        }
        for (p, m, u), values in sorted(prices.items())
    ]
