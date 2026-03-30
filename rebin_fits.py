#!/usr/bin/env python3
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def block_reduce_mean(img, B):
    # Trim so shape is divisible by B
    ny, nx = img.shape
    ny2, nx2 = (ny // B) * B, (nx // B) * B
    img = img[:ny2, :nx2]
    # Reshape and average blocks
    img = img.reshape(ny2 // B, B, nx2 // B, B).mean(axis=(1, 3))
    return img


def block_reduce_sum(img, B):
    ny, nx = img.shape
    ny2, nx2 = (ny // B) * B, (nx // B) * B
    img = img[:ny2, :nx2]
    img = img.reshape(ny2 // B, B, nx2 // B, B).sum(axis=(1, 3))
    return img


def update_wcs_for_block(header, B, out_shape):
    """Scale WCS for integer block size B."""
    hdr = header.copy()
    ny, nx = out_shape

    # If CD matrix present, scale it by B; if CDELT present, scale by B.
    if any(k in hdr for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2")):
        for k in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            if k in hdr:
                hdr[k] = hdr[k] * B
    # If PC is used with CDELT, leave PC alone; only scale CDELT below
    if "CDELT1" in hdr and "CDELT2" in hdr:
        hdr["CDELT1"] = hdr["CDELT1"] * B
        hdr["CDELT2"] = hdr["CDELT2"] * B

    if "IMAGEW" in hdr and "IMAGEH" in hdr:
        hdr["IMAGEW"] /= B
        hdr["IMAGEH"] /= B

    # Adjust CRPIX: pixels are now larger, so reference pixel index shrinks by B
    # Rough rule: new_crpix = (old_crpix - 0.5)/B + 0.5
    for ax in (1, 2):
        key = f"CRPIX{ax}"
        if key in hdr:
            hdr[key] = (hdr[key] - 0.5) / B + 0.5

    # Update NAXIS
    hdr["NAXIS1"] = nx
    hdr["NAXIS2"] = ny

    # Remove BSCALE/BZERO on integer data to avoid surprises; keep BUNIT etc.
    for k in ("BSCALE", "BZERO"):
        if k in hdr:
            del hdr[k]

    # Update PIXSCL and A_SEC2
    if "PIXSCL" in hdr:
        hdr["PIXSCL"] *= B

    # Leave SIP/TPV if present (not re-fit). For exact wide-field geometry, reproject later.
    return hdr


def main():
    ap = argparse.ArgumentParser(
        description="Downsample a FITS image by integer factor B and preserve WCS."
    )
    ap.add_argument("input_fits")
    ap.add_argument("output_fits")
    ap.add_argument(
        "-B",
        "--block",
        type=int,
        required=True,
        help="Integer block factor, e.g. 2,4,8",
    )
    ap.add_argument(
        "--sum",
        action="store_true",
        help="Use block-sum instead of mean (default mean).",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with fits.open(args.input_fits) as hdul:
        # Choose first HDU with image data
        ihdu = next(h for h in hdul if h.data is not None)
        data = ihdu.data.astype(float)
        hdr = ihdu.header

    B = int(args.block)
    if B < 1:
        raise ValueError("Block factor B must be >= 1")
    reducer = block_reduce_sum if args.sum else block_reduce_mean
    out = reducer(data, B)

    new_hdr = update_wcs_for_block(hdr, B, out.shape)

    fits.PrimaryHDU(data=out.astype(np.float32), header=new_hdr).writeto(
        args.output_fits, overwrite=args.overwrite
    )
    print(
        f"Saved {args.output_fits}  shape={out.shape}  (block {B}, mode={'sum' if args.sum else 'mean'})"
    )


if __name__ == "__main__":
    main()
