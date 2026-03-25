"""Setup ClinVar variants dataset."""

from pathlib import Path

import polars as pl
from cyvcf2 import VCF
from loguru import logger
from tqdm import tqdm

from src.datasets.paths import OUTPUTS

from .. import gencode, paths
from ..utils import STAR_MAPPING, ensure_parent, reverse_complement, run_setup, strand_aware_seq_pos
from .main import PRESETS, metadata


def _classify_significance(sig: str) -> str:
    """Map CLNSIG string to a canonical label.

    Distinguishes pathogenic from likely_pathogenic and benign from likely_benign.
    Compound annotations like "pathogenic/likely_pathogenic" map to "pathogenic"
    (the stronger claim). Modifiers like "|drug_response" are ignored.

    Conflicting and uncertain are checked first since their CLNSIG strings
    contain "pathogenic" as a substring (e.g., "conflicting_classifications_of_pathogenicity").

    Labels: pathogenic, likely_pathogenic, benign, likely_benign, vus, conflicting, other.
    """
    if "conflicting" in sig:
        return "conflicting"
    if "uncertain" in sig:
        return "vus"
    if "pathogenic" in sig and "benign" not in sig:
        if "likely_pathogenic" in sig and "pathogenic/likely_pathogenic" not in sig:
            return "likely_pathogenic"
        return "pathogenic"
    if "benign" in sig and "pathogenic" not in sig:
        if "likely_benign" in sig and "benign/likely_benign" not in sig:
            return "likely_benign"
        return "benign"
    return "other"


def parse_vcf(vcf_path: Path, output_path: Path) -> None:
    """Parse all ClinVar variants from VCF to feather format.

    Keeps ALL variants (pathogenic, benign, VUS, conflicting, other) with a
    label column derived from CLNSIG. Includes both SNVs and indels. Only
    skips: no ALT (multi-allelic), no gene info, MNV (complex substitutions).

    Filters variants outside gene boundaries and validates reference alleles
    against GENCODE sequences.
    """
    variants = []
    vcf = VCF(str(vcf_path))

    n_total = 0
    n_skipped_complex = 0
    n_no_gene = 0

    for variant in tqdm(vcf, desc="Parsing VCF"):
        n_total += 1

        if not variant.ALT:
            n_skipped_complex += 1
            continue

        ref, alt = variant.REF, variant.ALT[0]
        ref_len, alt_len = len(ref), len(alt)

        is_snv = ref_len == 1 and alt_len == 1
        is_indel = ref_len != alt_len

        if is_snv:
            variant_type = "snv"
        elif is_indel:
            variant_type = "insertion" if alt_len > ref_len else "deletion"
        else:
            n_skipped_complex += 1
            continue

        info = variant.INFO

        gene_info = info.get("GENEINFO", "")
        if not gene_info or ":" not in gene_info:
            n_no_gene += 1
            continue
        gene = gene_info.split(":")[0]

        sig = info.get("CLNSIG", "").lower()
        label = _classify_significance(sig)

        mc_field = info.get("MC", "")
        consequence = mc_field.split(",")[0].split("|")[-1] if "|" in mc_field else None

        stars = STAR_MAPPING.get(info.get("CLNREVSTAT", ""), 0)

        disease_raw = info.get("CLNDN", None)
        disease_name = None
        if disease_raw:
            disease_name = disease_raw.split("|")[0].replace("_", " ")
            if disease_name.lower() in ("not provided", "not specified"):
                disease_name = None

        allele_id_raw = info.get("ALLELEID", None)
        allele_id = int(allele_id_raw) if allele_id_raw is not None else None
        rs_raw = info.get("RS", None)
        rs_id = f"rs{rs_raw}" if rs_raw else None

        chrom = variant.CHROM
        chrom = f"chr{'M' if chrom == 'MT' else chrom}"

        variants.append({
            "chrom": chrom,
            "pos": variant.POS - 1,  # VCF 1-based -> 0-based
            "ref": ref,
            "alt": alt,
            "variant_type": variant_type,
            "gene_name": gene,
            "label": label,
            "clinical_significance": sig,
            "stars": stars,
            "consequence": consequence,
            "allele_id": allele_id,
            "disease_name": disease_name,
            "rs_id": rs_id,
        })

    df = pl.DataFrame(variants)
    label_counts = df.group_by("label").len().sort("len", descending=True)
    logger.info(
        f"Parsed {n_total:,} variants: kept {len(df):,}, "
        f"filtered {n_skipped_complex:,} complex/skipped, {n_no_gene:,} no gene\n"
        f"  Labels: {dict(zip(label_counts['label'].to_list(), label_counts['len'].to_list(), strict=True))}"
    )

    genes = gencode.metadata()

    n_vcf = len(df)
    df = df.join(genes.select("gene_name", "start", "end", "strand", "sequence"), on="gene_name", how="inner")
    n_joined = len(df)

    ref_len = pl.col("ref").str.len_bytes()
    df = df.filter(
        (pl.col("pos") >= pl.col("start")) &
        (pl.col("pos") + ref_len <= pl.col("end"))
    )
    n_in_bounds = len(df)

    df = df.with_columns(strand_aware_seq_pos().alias("seq_pos"))

    df = df.with_columns(
        pl.col("sequence").str.slice(pl.col("seq_pos"), ref_len).str.to_uppercase().alias("extracted_ref"),
        pl.when(pl.col("strand") == "-")
        .then(pl.col("ref").map_elements(reverse_complement, return_dtype=pl.String))
        .otherwise(pl.col("ref"))
        .str.to_uppercase()
        .alias("ref_corrected"),
    )
    n_before_ref = len(df)
    df = df.filter(pl.col("extracted_ref") == pl.col("ref_corrected"))
    n_ref_matched = len(df)

    df = df.drop("start", "end", "strand", "sequence", "seq_pos", "extracted_ref", "ref_corrected")

    logger.info(
        f"Filtered {n_vcf - n_joined:,} variants (gene not in GENCODE), "
        f"{n_joined - n_in_bounds:,} (outside boundaries), "
        f"{n_before_ref - n_ref_matched:,} (reference mismatch)"
    )

    df = df.with_columns(pl.col("stars").cast(pl.Int8))

    ensure_parent(output_path)
    df.write_ipc(output_path)
    logger.info(f"Saved {len(df):,} items to {output_path}")


def main() -> None:
    """Download and parse ClinVar VCF, optionally build cached presets."""
    vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    vcf_path = paths.DOWNLOADS / "clinvar.vcf.gz"
    tbi_path = Path(f"{vcf_path}.tbi")

    run_setup(
        "clinvar",
        "Download and build ClinVar datasets",
        downloads=(
            (vcf_url, vcf_path, False),
            (f"{vcf_url}.tbi", tbi_path, False),
        ),
        sources={"variants": (paths.CLINVAR_VARIANTS,
                               lambda: parse_vcf(vcf_path, paths.CLINVAR_VARIANTS))},
        presets={name: (OUTPUTS / "clinvar" / name / "metadata.feather",
                        lambda n=name: metadata(n))
                 for name, builder in PRESETS.items() if builder is not None},
    )


if __name__ == "__main__":
    main()
