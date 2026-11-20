import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.services.analytics import summarize_quotes

CSV = (
    "record_type;product;market;unit;final_price\n"
    "x;Corn;CBOT;US cents/bushel;400\n"
    "x;Corn;CBOT;US cents/bushel;500\n"
    "x;Corn;Euronext;EUR/tonne;200\n"
    "x;Wheat;Argus;EUR/tonne;300\n"
    "x;Wheat;Argus;USD/tonne;330\n"
    "x;Wheat;Argus;USD/tonne;-\n"
)


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "agritel_quotes.csv").write_text(CSV, encoding="utf-8-sig")
    return tmp_path


def test_groups_by_product_market_unit_without_mixing_units(data_dir):
    result = summarize_quotes(data_dir)
    wheat = [r for r in result if r["product"] == "Wheat"]
    assert {r["unit"] for r in wheat} == {"EUR/tonne", "USD/tonne"}
    assert all(r["count"] == 1 for r in wheat)  # linha com "-" ignorada
    corn = next(r for r in result if r["market"] == "CBOT")
    assert (corn["count"], corn["min"], corn["max"], corn["mean"]) == (2, 400.0, 500.0, 450.0)


def test_filters_are_case_insensitive(data_dir):
    result = summarize_quotes(data_dir, product="corn", market="euronext")
    assert [(r["product"], r["market"]) for r in result] == [("Corn", "Euronext")]


def test_endpoint_returns_summary_and_404_when_missing(data_dir, tmp_path):
    app.dependency_overrides[get_settings] = lambda: Settings(data_dir=data_dir)
    client = TestClient(app)
    try:
        r = client.get("/quotes/summary", params={"product": "Corn"})
        assert r.status_code == 200 and len(r.json()) == 2

        app.dependency_overrides[get_settings] = lambda: Settings(data_dir=tmp_path / "vazio")
        r = client.get("/quotes/summary")
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()
