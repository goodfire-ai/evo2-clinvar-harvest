"""Dataset file paths.

Local data: ``data/{module}/`` for cached metadata and downloaded source files.
Activation datasets live in shared artifact storage (passed as CLI args to scripts).
"""

from __future__ import annotations

from pathlib import Path

# --- Root directories ---

OUTPUTS = Path("data")
DOWNLOADS = Path("data/_downloads")

# --- Local cached metadata ---

CLINVAR = OUTPUTS / "clinvar"
GENCODE = OUTPUTS / "gencode"

# --- Source files (built by setup scripts) ---

CLINVAR_VARIANTS = CLINVAR / "variants.feather"
GENCODE_GENES = GENCODE / "genes.feather"
GENCODE_CHROMOSOMES = GENCODE / "chromosomes.feather"
GENCODE_EXONS = GENCODE / "exons.safetensors"
GENCODE_CDS = GENCODE / "cds.safetensors"


# --- Storage layout helpers ---


def activations_dir(module: str, preset: str, model: str) -> Path:
    """Activation chunks: data/{module}/{preset}/activations/{model}/"""
    return OUTPUTS / module / preset / "activations" / model


def embeddings_dir(module: str, preset: str, model: str, pooler: str) -> Path:
    """Embeddings: data/{module}/{preset}/embeddings/{model}/{pooler}/"""
    return OUTPUTS / module / preset / "embeddings" / model / pooler
