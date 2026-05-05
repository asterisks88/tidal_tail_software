#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"

found=0

for infile in "$DIR"/s*.fits; do
    [ -e "$infile" ] || continue

    name="$(basename "$infile")"
    if [[ "$name" =~ ^s[0-9]+\.fits$ ]]; then
        found=1
        base="${name%.fits}"
        outfile="$DIR/${base}_bin4.fits"

        echo "Rebinning: $infile -> $outfile"
        rebin_fits.py "$infile" "$outfile" -B 4 --sum --overwrite
    fi
done

if [ "$found" -eq 0 ]; then
    echo "No files matching s<number>.fits found in: $DIR" >&2
    exit 1
fi

echo "Done."
