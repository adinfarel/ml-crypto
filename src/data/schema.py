from typing import List
from pydantic import BaseModel

class BinanceKlineSchema(BaseModel):
    COLUMNS: List[str] = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades_count",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
        "ignore",
    ]
    
    NUMERIC_COLUMNS: List[str] = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base_vol",
        "taker_buy_quote_vol",
    ]
    
SCHEMA = BinanceKlineSchema()