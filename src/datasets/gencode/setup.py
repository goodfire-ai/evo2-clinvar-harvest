"""Setup GENCODE genes dataset."""

from pathlib import Path

import gffutils
import polars as pl
from loguru import logger
from pyfaidx import Fasta
from tqdm import tqdm

from .. import paths
from ..utils import CHROMOSOMES, ensure_parent, reverse_complement, run_setup, strip_version


def parse_gtf(gtf_path: Path, biotype: str = "protein_coding") -> pl.DataFrame:
    """Parse GTF to DataFrame (creates sqlite cache on first run)."""
    db_path = gtf_path.with_suffix(".gtf.db")

    db = (
        gffutils.FeatureDB(str(db_path))
        if db_path.exists()
        else gffutils.create_db(
            str(gtf_path), str(db_path), force=True, keep_order=True,
            merge_strategy="merge", disable_infer_transcripts=True, disable_infer_genes=True
        )
    )

    def attr(g, key: str, default: str = "") -> str:
        return g.attributes.get(key, [default])[0]

    genes = [g for g in db.features_of_type("gene") if attr(g, "gene_type", attr(g, "gene_biotype")) == biotype]
    records = [
        {
            "gene_id": attr(g, "gene_id"),
            "gene_name": attr(g, "gene_name"),
            "chrom": g.chrom,
            "start": g.start - 1,
            "end": g.end,
            "strand": g.strand,
            "length": g.end - g.start + 1,  # GTF is 1-based inclusive
            "hgnc_id": attr(g, "hgnc_id"),
            "level": int(attr(g, "level", "0")),
        }
        for g in tqdm(genes, desc="Parsing GTF")
    ]

    return pl.DataFrame(records)


def extract_sequences(genes: pl.DataFrame, fasta_path: Path) -> dict[str, str]:
    """Extract sequences from FASTA for genes."""
    genome = Fasta(str(fasta_path))
    available = set(genome.keys())

    def resolve_chrom(c: str) -> str | None:
        if c in available:
            return c
        alt = c.replace("chr", "") if "chr" in c else f"chr{c}"
        return alt if alt in available else None

    chrom_map = {c: resolve_chrom(c) for c in genes["chrom"].unique().to_list()}

    # Row iteration required: genome is pyfaidx object, not vectorizable
    sequences = {}
    n_unmapped = 0
    for row in tqdm(genes.iter_rows(named=True), total=len(genes), desc="Extracting sequences"):
        chrom = chrom_map.get(row["chrom"])
        if chrom is None:
            n_unmapped += 1
            continue
        seq = str(genome[chrom][row["start"] : row["end"]])
        if row["strand"] == "-":
            seq = reverse_complement(seq)
        sequences[row["gene_id"]] = seq

    logger.info(f"Extracted {len(sequences):,} sequences ({n_unmapped} genes with unmapped chromosomes skipped)")
    return sequences


def build_genes(gtf_path: Path, fasta_path: Path, output_path: Path) -> None:
    """Build genes dataset with sequences.

    Normalizes to canonical format during build:
    - gene_id: stripped of Ensembl version suffix (ENSG00000139618.19 -> ENSG00000139618)
    - chrom: stripped of 'chr' prefix (chr17 -> 17)
    """
    genes = parse_gtf(gtf_path)
    logger.info(f"Parsed {len(genes):,} genes")

    genes = genes.with_columns(
        pl.col("gene_id").str.replace(r"\.\d+$", ""),
        pl.col("chrom").str.replace("^chr", ""),
    )

    seqs = extract_sequences(genes, fasta_path)
    genes = genes.with_columns(
        sequence=pl.col("gene_id").replace_strict(seqs, default=None)
    )
    n_before = len(genes)
    genes = genes.drop_nulls(subset=["sequence"])
    logger.info(f"Extracted {len(genes):,} sequences (dropped {n_before - len(genes):,} genes without FASTA match)")

    ensure_parent(output_path)
    genes.write_ipc(output_path)
    logger.info(f"Saved {len(genes):,} items to {output_path}")


def build_chromosomes(fasta_path: Path, output_path: Path) -> None:
    """Extract full chromosome sequences from FASTA to feather.

    Normalizes chromosome names to match GENCODE convention (chr1 -> 1).
    Only includes primary assembly chromosomes (1-22, X, Y, M).
    """
    genome = Fasta(str(fasta_path))
    records = []
    for chrom_name in genome.keys():
        normalized = chrom_name.replace("chr", "")
        if normalized not in CHROMOSOMES:
            continue
        records.append({"chrom": normalized, "sequence": str(genome[chrom_name][:]).upper()})
    df = pl.DataFrame(records)
    logger.info(f"Extracted {len(df)} chromosome sequences")

    ensure_parent(output_path)
    df.write_ipc(output_path)
    logger.info(f"Saved to {output_path}")


def main() -> None:
    """Download GENCODE GTF/FASTA and build genes dataset."""
    base_url = "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49"
    gtf_url = f"{base_url}/gencode.v49.basic.annotation.gtf.gz"
    fasta_url = f"{base_url}/GRCh38.primary_assembly.genome.fa.gz"

    gtf_path = paths.DOWNLOADS / "gencode.v49.basic.annotation.gtf"
    fasta_path = paths.DOWNLOADS / "GRCh38.primary_assembly.genome.fa"

    run_setup(
        "gencode",
        "Download GENCODE GTF/FASTA and build genes dataset",
        downloads=(
            (gtf_url, gtf_path, True),
            (fasta_url, fasta_path, True),
        ),
        sources={
            "genes": (paths.GENCODE_GENES,
                      lambda: build_genes(gtf_path, fasta_path, paths.GENCODE_GENES)),
            "chromosomes": (paths.GENCODE_CHROMOSOMES,
                            lambda: build_chromosomes(fasta_path, paths.GENCODE_CHROMOSOMES)),
        },
    )


if __name__ == "__main__":
    main()
