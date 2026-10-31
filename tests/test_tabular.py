from pathlib import Path

import pytest

from app.services.ingestion import build_chunks, load_document
from app.services.tabular import load_csv, normalize_contract, render_row

HEADER = "record_type;product;market;reference_date;contract;description;unit;final_price;variation;high;low;volume;last_trade_time"
FUTURES = "futures;Corn;Euronext;30/10/2026;27 de março;;EUR/tonne;240.5;-1.25;241.0;239.0;12345.0;-"
PHYSICAL = "physical;Barley;Physical;2026-10-30;;Cevada para ração entregue em Rouen;EUR/tonne;207.5;1.0;;;;"


def test_normalize_contract_infers_year_and_keeps_original():
    out = normalize_contract("27 de março")
    assert out.startswith("março de 2027") and '"27 de março"' in out
    assert normalize_contract("texto livre") == "texto livre"


def test_render_futures_row():
    row = dict(zip(HEADER.split(";"), FUTURES.split(";")))
    text = render_row(row)
    assert "futuros" in text and "Corn (milho)" in text and "mercado Euronext" in text
    assert "2026-10-30" in text  # data normalizada para ISO
    assert "março de 2027" in text
    assert "Volume: 12345." in text  # sem '.0'
    assert "Preço final: 240.5" in text and "Variação: -1.25" in text
    assert "Hora do último negócio" not in text  # '-' é tratado como vazio


def test_render_physical_row_skips_empty_fields():
    row = dict(zip(HEADER.split(";"), PHYSICAL.split(";")))
    text = render_row(row)
    assert "mercado físico" in text and "Barley (cevada)" in text
    assert "Descrição: Cevada para ração entregue em Rouen" in text
    assert "Máxima" not in text and "contrato" not in text


def test_unknown_columns_are_kept():
    assert "cor: azul" in render_row({"cor": "azul"})


def test_load_csv_handles_bom_crlf_and_line_numbers(tmp_path):
    f = tmp_path / "q.csv"
    f.write_bytes(("\ufeff" + HEADER + "\r\n" + FUTURES + "\r\n" + PHYSICAL + "\r\n").encode("utf-8"))
    pages = load_document(f, tmp_path)
    assert [p.row for p in pages] == [2, 3]
    assert pages[0].source == "q.csv"
    chunks = build_chunks(pages, 800, 100)
    assert len(chunks) == 2 and chunks[0].id == "q.csv::r2::c0" and chunks[0].row == 2


REAL = Path(__file__).resolve().parent.parent / "data" / "agritel_quotes.csv"


@pytest.mark.skipif(not REAL.exists(), reason="dataset não está em data/")
def test_real_dataset_loads_every_row_as_single_chunk():
    pages = load_csv(REAL, REAL.parent)
    assert len(pages) == 131
    chunks = build_chunks(pages, 800, 120)
    assert len(chunks) == 131 and len({c.id for c in chunks}) == 131
    assert all("Cotação de" in c.text for c in chunks)
