#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 /path/to/directory /path/to/tail.reg /path/to/mask.fits"
    exit 1
fi

indir=$1
region_file=$2
mask_fits=$3

if [[ ! -d "$indir" ]]; then
    echo "Error: '$indir' is not a directory"
    exit 1
fi

if [[ ! -f "$region_file" ]]; then
    echo "Error: region file '$region_file' does not exist"
    exit 1
fi

if [[ ! -f "$mask_fits" ]]; then
    echo "Error: mask FITS '$mask_fits' does not exist"
    exit 1
fi

shopt -s nullglob

files=(
    "$indir"/s[0-9]_bin4.fits
    "$indir"/s[0-9][0-9]_bin4.fits
    "$indir"/s[0-9][0-9][0-9]_bin4.fits
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matching s0_bin4.fits through s99_bin4.fits found in $indir"
    exit 1
fi

echo "=== Building mu images ==="
for file in "${files[@]}"; do
    base=$(basename "$file" .fits)
    out_mu="$indir/${base}_mu.fits"

    if [[ -f "$out_mu" ]]; then
        echo "Skipping existing: $out_mu"
        continue
    fi

    echo "Processing: $file -> $out_mu"
    mu_from_legacy.py "$file" "$out_mu" \
        --smooth gauss \
        --gauss-sigma 1.0 \
        --kernel-mode mean \
        --overwrite
done

echo
echo "=== Running surface-brightness Monte Carlo stats ==="

surface_brightness_mc_stats.py "${indir}/s*_bin4_mu.fits" "$region_file" "$mask_fits" \
    --measurer "$HOME/tidal-tail-software/tail_surface_brightness.py" \
    --summary-csv "$indir/sb_mc_measurements.csv" \
    --stats-csv "$indir/sb_mc_stats.csv" \
    --flux-hdu SKY_SUB_NMGY \
    --mask-hdu 0 \
    --nrand 1000 \
    --subpixels 8

echo
echo "Done."
echo "Per-realization SB table: $indir/sb_mc_measurements.csv"
echo "Aggregate SB stats:       $indir/sb_mc_stats.csv"
