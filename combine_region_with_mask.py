#!/usr/bin/env python3
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions


def hdu_key(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(
        description="Combine a DS9 region file with an existing FITS mask."
    )
    parser.add_argument("mask_fits", help="Existing FITS mask; 0=good, nonzero=masked")
    parser.add_argument("region_file", help="DS9 region file to add to the mask")
    parser.add_argument("output_fits", help="Output combined FITS mask")
    parser.add_argument("--hdu", default="0", help="Mask HDU index/name, default 0")
    parser.add_argument(
        "--ref",
        default=None,
        help="Reference FITS image for WCS/shape. Use this if region is fk5/wcs.",
    )
    parser.add_argument("--ref-hdu", default="0", help="Reference image HDU, default 0")
    parser.add_argument(
        "--value",
        type=int,
        default=1,
        help="Value/bit to assign to the new region pixels, default 1",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write output as binary 0/1 mask instead of preserving mask values",
    )
    parser.add_argument(
        "--bitwise-or",
        action="store_true",
        help="For integer maskbits files, OR --value into existing region pixels",
    )
    parser.add_argument(
        "--mode",
        choices=["center", "exact", "subpixels"],
        default="center",
        help="Region rasterization mode, default center",
    )
    parser.add_argument("--subpixels", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    mask_hdu = hdu_key(args.hdu)
    ref_hdu = hdu_key(args.ref_hdu)

    with fits.open(args.mask_fits) as hdul:
        hdul_out = fits.HDUList([hdu.copy() for hdu in hdul])
        mask_data = hdul_out[mask_hdu].data

        if mask_data is None or mask_data.ndim != 2:
            raise ValueError("Mask HDU must contain a 2D image.")

        shape = mask_data.shape

        if args.ref is not None:
            with fits.open(args.ref) as rhdul:
                ref_data = rhdul[ref_hdu].data
                ref_header = rhdul[ref_hdu].header
                if ref_data.shape != shape:
                    raise ValueError(
                        f"Reference image shape {ref_data.shape} does not match "
                        f"mask shape {shape}."
                    )
                wcs = WCS(ref_header)
        else:
            wcs = WCS(hdul_out[mask_hdu].header)

        regions = Regions.read(args.region_file, format="ds9")

        region_mask = np.zeros(shape, dtype=bool)

        for reg in regions:
            # Sky regions, e.g. fk5, need conversion to pixels.
            try:
                pixreg = reg.to_pixel(wcs)
            except AttributeError:
                pixreg = reg

            if not hasattr(pixreg, "to_mask"):
                print(f"Skipping unsupported region: {reg}")
                continue

            rmask = pixreg.to_mask(mode=args.mode, subpixels=args.subpixels)
            img = rmask.to_image(shape)

            if img is None:
                continue

            this_mask = img > 0

            # DS9 exclude regions, if present, subtract from the accumulated mask.
            include = pixreg.meta.get("include", True)
            if include:
                region_mask |= this_mask
            else:
                region_mask &= ~this_mask

        if args.binary:
            out = ((mask_data != 0) | region_mask).astype(np.uint8)

        elif args.bitwise_or:
            if not np.issubdtype(mask_data.dtype, np.integer):
                raise TypeError("--bitwise-or requires an integer FITS mask.")
            out = mask_data.copy()
            out[region_mask] = np.bitwise_or(out[region_mask], args.value)

        else:
            out = mask_data.copy()
            out[(mask_data == 0) & region_mask] = args.value

        hdul_out[mask_hdu].data = out
        hdul_out[mask_hdu].header.add_history(
            f"Combined with DS9 region file: {args.region_file}"
        )
        hdul_out.writeto(args.output_fits, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
