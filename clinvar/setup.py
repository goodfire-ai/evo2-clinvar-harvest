"""Parse ClinVar VCF into feather format."""

from pathlib import Path

import polars as pl
from cyvcf2 import VCF
from loguru import logger
from tqdm import tqdm

import gencode
from utils import STAR_MAPPING, ensure_parent, reverse_complement, strand_aware_seq_pos


def _classify_significance(sig: str) -> str:
    """Map CLNSIG string to a canonical label."""
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
    """Parse all ClinVar variants from VCF to feather format."""
    variants = []
    vcf = VCF(str(vcf_path))
    n_total, n_skipped, n_no_gene = 0, 0, 0

    for variant in tqdm(vcf, desc="Parsing VCF"):
        n_total += 1
        if not variant.ALT:
            n_skipped += 1
            continue

        ref, alt = variant.REF, variant.ALT[0]
        is_snv = len(ref) == 1 and len(alt) == 1
        is_indel = len(ref) != len(alt)
        if is_snv:
            variant_type = "snv"
        elif is_indel:
            variant_type = "insertion" if len(alt) > len(ref) else "deletion"
        else:
            n_skipped += 1
            continue

        info = variant.INFO
        gene_info = info.get("GENEINFO", "")
        if not gene_info or ":" not in gene_info:
            n_no_gene += 1
            continue

        sig = info.get("CLNSIG", "").lower()
        mc_field = info.get("MC", "")
        disease_raw = info.get("CLNDN", None)
        disease_name = None
        if disease_raw:
            disease_name = disease_raw.split("|")[0].replace("_", " ")
            if disease_name.lower() in ("not provided", "not specified"):
                disease_name = None

        allele_id_raw = info.get("ALLELEID", None)
        rs_raw = info.get("RS", None)
        chrom = variant.CHROM
        chrom = f"chr{'M' if chrom == 'MT' else chrom}"

        variants.append({
            "chrom": chrom, "pos": variant.POS - 1, "ref": ref, "alt": alt,
            "variant_type": variant_type, "gene_name": gene_info.split(":")[0],
            "label": _classify_significance(sig), "clinical_significance": sig,
            "stars": STAR_MAPPING.get(info.get("CLNREVSTAT", ""), 0),
            "consequence": mc_field.split(",")[0].split("|")[-1] if "|" in mc_field else None,
            "allele_id": int(allele_id_raw) if allele_id_raw is not None else None,
            "disease_name": disease_name,
            "rs_id": f"rs{rs_raw}" if rs_raw else None,
        })

    df = pl.DataFrame(variants)
    logger.info(f"Parsed {n_total:,} variants: kept {len(df):,}, skipped {n_skipped:,} complex, {n_no_gene:,} no gene")

    # Filter against GENCODE gene boundaries and validate reference alleles
    genes = gencode.metadata()
    n_vcf = len(df)
    df = df.join(genes.select("gene_name", "start", "end", "strand", "sequence"), on="gene_name", how="inner")

    ref_len = pl.col("ref").str.len_bytes()
    df = df.filter((pl.col("pos") >= pl.col("start")) & (pl.col("pos") + ref_len <= pl.col("end")))
    df = df.with_columns(strand_aware_seq_pos().alias("seq_pos"))
    df = df.with_columns(
        pl.col("sequence").str.slice(pl.col("seq_pos"), ref_len).str.to_uppercase().alias("extracted_ref"),
        pl.when(pl.col("strand") == "-")
        .then(pl.col("ref").map_elements(reverse_complement, return_dtype=pl.String))
        .otherwise(pl.col("ref")).str.to_uppercase().alias("ref_corrected"),
    )
    df = df.filter(pl.col("extracted_ref") == pl.col("ref_corrected"))
    df = df.drop("start", "end", "strand", "sequence", "seq_pos", "extracted_ref", "ref_corrected")
    df = df.with_columns(pl.col("stars").cast(pl.Int8))

    logger.info(f"After filtering: {len(df):,} / {n_vcf:,} variants")
    ensure_parent(output_path)
    df.write_ipc(output_path)
