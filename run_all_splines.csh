#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

indir=$1

if [[ ! -d "$indir" ]]; then
    echo "Error: '$indir' is not a directory"
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
echo "=== Running tail-length Monte Carlo stats ==="

tail_length_mc_stats.py "${indir}/s*_bin4_mu.fits" \
    --tracer "$HOME/tidal-tail-software/trace_sat_tail_spline.py" \
    --summary-csv "$indir/tail_mc_lengths.csv" \
    --stats-csv "$indir/tail_mc_length_stats.csv" \
    --products-dir "$indir/tail_mc_products" \
    --hdu MU_OBS \
    --seed-points-csv ~/LEDA175897_reference/seed_points.csv \
    --mu-max 28.5 \
    --smooth-sigma 2.0 \
    --step 25 \
    --direction-relax 0.55 \
    --intensity-power 0.5 \
    --cap-percentile 98.5 \
    --sigma-parallel-factor 0.6 \
    --sigma-perp-factor 0.45 \
    --spline-s 25 \
    --star-mask-fits ~/LEDA175897_reference/final_brick_mask.fits \
    --beam-width 6 \
    --lookahead 4 \
    --n-long-bins 8 \
    --min-bin-frac 0.5 \
    --mean-weight 0.10 \
    --std-penalty 0.35

echo
echo "Done."
echo "Per-realization lengths: $indir/tail_mc_lengths.csv"
echo "Aggregate stats:         $indir/tail_mc_length_stats.csv"
echo "Per-run products:        $indir/tail_mc_products"
