#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 /path/to/directory [--seed-points-csv /path/to/seed_points.csv]"
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

indir=$1
shift
seed_points_csv="$HOME/LEDA175897_legacy/tail_points/seed_points_old.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed-points-csv)
            if [[ $# -lt 2 ]]; then
                echo "Error: --seed-points-csv requires a file path" >&2
                exit 1
            fi
            seed_points_csv=$2
            shift 2
            ;;
        --seed-points-csv=*)
            seed_points_csv=${1#*=}
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown argument '$1'" >&2
            usage
            exit 1
            ;;
    esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$indir" ]]; then
    echo "Error: '$indir' is not a directory"
    exit 1
fi

if [[ ! -f "$seed_points_csv" || ! -r "$seed_points_csv" ]]; then
    echo "Error: seed-points CSV '$seed_points_csv' is not a readable file" >&2
    exit 1
fi

shopt -s nullglob

files=(
    "$indir"/s_bin4_observed_[0-9][0-9][0-9][0-9].fits
)

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matching s_bin4_observed_0000.fits through s_bin4_observed_9999.fits found in $indir"
    exit 1
fi

echo
echo "=== Running tail-length and turning-angle Monte Carlo stats ==="

python3 "$script_dir/tail_length_mc_stats.py" \
    "${indir}/s_bin4_observed_[0-9][0-9][0-9][0-9].fits" \
    --tracer "$HOME/tidal-tail-software/trace_sat_tail_spline.py" \
    --summary-csv "$indir/tail_mc_lengths.csv" \
    --stats-csv "$indir/tail_mc_length_stats.csv" \
    --products-dir "$indir/tail_mc_products" \
    --turning-n-samples 500 \
    --turning-smooth-sigma 2.0 \
    --hdu MU_SCIENCE \
    --seed-points-csv "$seed_points_csv" \
    --mu-max 28.5 \
    --smooth-sigma 2.25 \
    --step 20 \
    --direction-relax 1.8 \
    --intensity-power 0.5 \
    --cap-percentile 98.5 \
    --sigma-parallel-factor 0.3 \
    --sigma-perp-factor 0.3 \
    --spline-s 50 \
    --star-mask-fits ~/LEDA175897_legacy/maskbits_auto_gaia_bin4.fits \
    --beam-width 6 \
    --lookahead 3 \
    --n-long-bins 6 \
    --min-bin-frac 0.5 \
    --mean-weight 0.10 \
    --std-penalty 0.35
echo
echo "Done."
echo "Per-realization lengths and angles: $indir/tail_mc_lengths.csv"
echo "Aggregate length and angle stats:  $indir/tail_mc_length_stats.csv"
echo "Per-run products:                  $indir/tail_mc_products"
