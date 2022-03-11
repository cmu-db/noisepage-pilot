from __future__ import annotations

from behavior.plans import DIFFERENCING_SCHEMA

METHODS = [
    "lr",
    "huber",
    "svr",
    "kr",
    "rf",
    "gbm",
    "mlp",
    "mt_lasso",
    "lasso",
    "dt",
    "mt_elastic",
    "elastic",
]

# Any feature that ends with any keyword in BLOCKED_FEATURES is dropped
# from the input schema. These features are intentionally dropped since
# we don't want feature selection/model to try and learn from these.
BLOCKED_FEATURES: list[str] = DIFFERENCING_SCHEMA + [
    "plan_type",  # Plan Type
    "cpu_id",  # CPU ID
    "relid",  # Scan index into range able
    "indexid",  # Index OID
]
