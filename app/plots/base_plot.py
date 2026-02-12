from abc import ABC, abstractmethod
from typing import Any

from plotly.graph_objects import Figure
from plotly.utils import PlotlyJSONEncoder

import json



class BasePlot(ABC):

    @abstractmethod
    def plot(self, *args: Any, **kwargs: Any) -> str:
        ...

    @staticmethod
    def _get_json(fig: Figure) -> str:
        fig_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        return fig_json
