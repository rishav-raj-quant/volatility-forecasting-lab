"""Model registry for instantiating machine learning volatility models."""

from typing import Dict, Any, List
from volatility_lab.machine_learning.linear import get_linear_model
from volatility_lab.machine_learning.tree_models import get_random_forest_model, get_extra_trees_model
from volatility_lab.machine_learning.boosting import (
    get_gradient_boosting_model,
    get_hist_gradient_boosting_model,
    get_xgboost_model,
    get_lightgbm_model,
)
from volatility_lab.machine_learning.svr import get_svr_model


class MLModelRegistry:
    """Central registry and factory for machine learning forecasting models."""

    @staticmethod
    def list_models() -> List[str]:
        return [
            "ridge",
            "lasso",
            "elastic_net",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "hist_gradient_boosting",
            "xgboost",
            "lightgbm",
            "svr",
        ]

    @staticmethod
    def get(model_name: str, **kwargs: Any):
        name = model_name.lower().replace("-", "_")
        if name == "ridge":
            return get_linear_model("ridge", **kwargs)
        elif name == "lasso":
            return get_linear_model("lasso", **kwargs)
        elif name in ("elastic_net", "elasticnet"):
            return get_linear_model("elastic_net", **kwargs)
        elif name == "random_forest":
            return get_random_forest_model(**kwargs)
        elif name == "extra_trees":
            return get_extra_trees_model(**kwargs)
        elif name == "gradient_boosting":
            return get_gradient_boosting_model(**kwargs)
        elif name == "hist_gradient_boosting":
            return get_hist_gradient_boosting_model(**kwargs)
        elif name == "xgboost":
            return get_xgboost_model(**kwargs)
        elif name == "lightgbm":
            return get_lightgbm_model(**kwargs)
        elif name == "svr":
            return get_svr_model(**kwargs)
        else:
            raise ValueError(f"Unknown ML model '{model_name}'. Available: {MLModelRegistry.list_models()}")
