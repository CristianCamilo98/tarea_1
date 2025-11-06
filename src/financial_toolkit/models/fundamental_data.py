from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict


@dataclass
class FundamentalData:
    """
    Standardized fundamental data structure for a single company
    at a given reporting date or period.

    All numeric fields are intended as *per-share* or *absolute* values
    in the reporting currency (e.g. USD).
    """
    symbol: str
    company_name: str
    report_date: datetime
    period: str
    currency: str = "USD"
    source: str = "unknown"

    # Market info (snapshot-type)
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    beta: Optional[float] = None

    # Income statement
    revenuettm: Optional[float] = None # trailing twelve months revenue
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    evtoebitda: Optional[float] = None

    # Balance sheet
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_equivalents: Optional[float] = None

    # Cash flow
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # Per-share metrics
    eps_basic: Optional[float] = None
    eps_diluted_ttm: Optional[float] = None
    book_value_per_share: Optional[float] = None
    dividends_per_share: Optional[float] = None

    # Valuation ratios
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio_ttm: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None

    def __post_init__(self):
        # You can add basic validations here if you want:
        if self.report_date.tzinfo is None:
            # you might want everything in UTC like in PriceData
            # or allow naive dates & normalize later
            pass
        # Calculation of derived fields
        self.net_income = self.revenuettm * self.profit_margin if self.revenuettm and self.profit_margin else None

    def to_dict(self) -> Dict:
        """Convert FundamentalData to dictionary."""
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "report_date": self.report_date.isoformat(),
            "period": self.period,
            "currency": self.currency,
            "source": self.source,
            "market_cap": self.market_cap,
            "shares_outstanding": self.shares_outstanding,
            "beta": self.beta,
            "revenuettm": self.revenuettm,
            "ebit": self.ebit,
            "ebitda": self.ebitda,
            "evtoebitda": self.evtoebitda,
            "total_assets": self.total_assets,
            "total_liabilities": self.total_liabilities,
            "total_equity": self.total_equity,
            "total_debt": self.total_debt,
            "cash_and_equivalents": self.cash_and_equivalents,
            "operating_cash_flow": self.operating_cash_flow,
            "free_cash_flow": self.free_cash_flow,
            "eps_basic": self.eps_basic,
            "eps_diluted_ttm": self.eps_diluted_ttm,
            "book_value_per_share": self.book_value_per_share,
            "dividends_per_share": self.dividends_per_share,
            "pe_ratio": self.pe_ratio,
            "pb_ratio": self.pb_ratio,
            "ps_ratio_ttm": self.ps_ratio_ttm,
            "dividend_yield": self.dividend_yield,
            "roe": self.roe,
            "roa": self.roa,
            "profit_margin": self.profit_margin,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'FundamentalData':
        """Create FundamentalData from dictionary."""
        return cls(**data)

