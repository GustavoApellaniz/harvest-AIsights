from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.services.analytics import summarize_quotes

router = APIRouter(prefix="/quotes", tags=["quotes"])


@router.get("/summary")
def quotes_summary(
    product: str | None = Query(default=None, description="Filtra por produto (ex.: Corn)"),
    market: str | None = Query(default=None, description="Filtra por mercado (ex.: CBOT)"),
    settings: Settings = Depends(get_settings),
) -> list[dict]:
    """Contagem, mínimo, máximo e média de `final_price` por produto, mercado e unidade."""
    return summarize_quotes(settings.data_dir, product, market)
