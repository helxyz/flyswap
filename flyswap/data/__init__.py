"""Dataset ingestion: one schema, three connectomes, plus synthetic fixtures."""

from flyswap.data.schema import (  # noqa: F401
    DATASETS,
    Connectome,
    DatasetSpec,
    config_digest,
    file_digest,
    normalise_nt,
)
