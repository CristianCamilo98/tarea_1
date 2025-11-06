from dataclasses import dataclass
from typing import Dict
from pandas import pd

@dataclass
class MacroSeriesData:
    """
    Standardized macroeconomic / fundamental time series
    """
    series_id: str
    description: str
    data: pd.Series
    units: str
    frequency: str
    source: str = "unknown"

    def to_dict(self) -> Dict:
        return {
            "series_id": self.series_id,
            "description": self.description,
            "data": self.data.to_dict(),
            "units": self.units,
            "frequency": self.frequency,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MacroSeriesData':
        series = pd.Series(data["data"])
        series.index = pd.to_datetime(series.index)
        return cls(
            series_id=data["series_id"],
            description=data["description"],
            data=series,
            units=data["units"],
            frequency=data["frequency"],
            source=data.get("source", "unknown"),
        )
