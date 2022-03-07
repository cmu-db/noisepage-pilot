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

# These are the set of features that are blocked as being input
# features to the models. It is also worth noting that if any feature
# ends with any suffixes in BLOCKED_FEATURES, that feature is also
# removed from the schema.
#
# These features are intentionally blocked since we don't want
# feature selection or the model to try and learn based on these.
BLOCKED_FEATURES: list[str] = DIFFERENCING_SCHEMA + [
    "plan_type",  # Plan Type
    "cpu_id",  # CPU ID
    "relid",  # Scan index into range able
    "indexid",  # Index OID
]
