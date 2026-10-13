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
    "$indir"/s_bin4_dereddened_[0-9][0-9][0-9][0-9].fits
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matching s_bin4_dereddened_0000.fits through s_bin4_observed_9999.fits found in $indir"
    exit 1
fi

echo
echo "=== Running surface-brightness Monte Carlo stats ==="

surface_brightness_mc_stats.py "${indir}/s_bin4_dereddened*.fits" "$region_file" "$mask_fits" \
    --measurer "$HOME/tidal-tail-software/tail_surface_brightness.py" \
    --summary-csv "$indir/sb_mc_measurements.csv" \
    --stats-csv "$indir/sb_mc_stats.csv" \
    --flux-hdu SCIENCE_NMGY \
    --mask-hdu 0 \
    --nrand 1000 \
    --subpixels 8

echo
echo "Done."
echo "Per-realization SB table: $indir/sb_mc_measurements.csv"
echo "Aggregate SB stats:       $indir/sb_mc_stats.csv"
