"""Build GENCODE genes and chromosome datasets from GTF + FASTA."""

from pathlib import Path

import gffutils
import polars as pl
from loguru import logger
from pyfaidx import Fasta
from tqdm import tqdm

from utils import CHROMOSOMES, ensure_parent, reverse_complement



def parse_gtf(gtf_path: Path, biotype: str = "protein_coding") -> pl.DataFrame:
    """Parse GTF to DataFrame (creates sqlite cache on first run)."""
    db_path = gtf_path.with_suffix(".gtf.db")
    db = (
        gffutils.FeatureDB(str(db_path))
        if db_path.exists()
        else gffutils.create_db(
            str(gtf_path), str(db_path), force=True, keep_order=True,
            merge_strategy="merge", disable_infer_transcripts=True, disable_infer_genes=True,
        )
    )

    def attr(g, key: str, default: str = "") -> str:
        return g.attributes.get(key, [default])[0]

    genes = [g for g in db.features_of_type("gene") if attr(g, "gene_type", attr(g, "gene_biotype")) == biotype]
    return pl.DataFrame([
        {
            "gene_id": attr(g, "gene_id"), "gene_name": attr(g, "gene_name"),
            "chrom": g.chrom, "start": g.start - 1, "end": g.end, "strand": g.strand,
            "length": g.end - g.start + 1, "hgnc_id": attr(g, "hgnc_id"),
            "level": int(attr(g, "level", "0")),
        }
        for g in tqdm(genes, desc="Parsing GTF")
    ])


def extract_sequences(genes: pl.DataFrame, fasta_path: Path) -> dict[str, str]:
    """Extract gene sequences from reference FASTA."""
    genome = Fasta(str(fasta_path))
    available = set(genome.keys())

    def resolve(c: str) -> str | None:
        if c in available:
            return c
        alt = c.replace("chr", "") if "chr" in c else f"chr{c}"
        return alt if alt in available else None

    chrom_map = {c: resolve(c) for c in genes["chrom"].unique().to_list()}
    sequences, n_unmapped = {}, 0
    for row in tqdm(genes.iter_rows(named=True), total=len(genes), desc="Extracting sequences"):
        chrom = chrom_map.get(row["chrom"])
        if chrom is None:
            n_unmapped += 1
            continue
        seq = str(genome[chrom][row["start"] : row["end"]])
        if row["strand"] == "-":
            seq = reverse_complement(seq)
        sequences[row["gene_id"]] = seq
    logger.info(f"Extracted {len(sequences):,} sequences ({n_unmapped} unmapped)")
    return sequences


def build_genes(gtf_path: Path, fasta_path: Path, output_path: Path) -> None:
    """Build genes dataset with sequences."""
    genes = parse_gtf(gtf_path)
    logger.info(f"Parsed {len(genes):,} genes")
    genes = genes.with_columns(
        pl.col("gene_id").str.replace(r"\.\d+$", ""),
        pl.col("chrom").str.replace("^chr", ""),
    )
    seqs = extract_sequences(genes, fasta_path)
    genes = genes.with_columns(sequence=pl.col("gene_id").replace_strict(seqs, default=None))
    n_before = len(genes)
    genes = genes.drop_nulls(subset=["sequence"])
    logger.info(f"{len(genes):,} genes with sequences (dropped {n_before - len(genes):,})")
    ensure_parent(output_path)
    genes.write_ipc(output_path)


def build_chromosomes(fasta_path: Path, output_path: Path) -> None:
    """Extract full chromosome sequences from FASTA to feather."""
    genome = Fasta(str(fasta_path))
    records = [
        {"chrom": name.replace("chr", ""), "sequence": str(genome[name][:]).upper()}
        for name in genome.keys() if name.replace("chr", "") in CHROMOSOMES
    ]
    df = pl.DataFrame(records)
    logger.info(f"Extracted {len(df)} chromosome sequences")
    ensure_parent(output_path)
    df.write_ipc(output_path)
