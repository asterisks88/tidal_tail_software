#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Button, RangeSlider, Slider


def read_fits_2d(path: str, hdu_spec="0") -> np.ndarray:
    """
    Read a FITS HDU by index (e.g. '0', '1') or by EXTNAME (e.g. 'MU_OBS'),
    and coerce it to a 2D numpy array.
    """
    spec = str(hdu_spec)

    with fits.open(path, memmap=True) as hdul:
        hdu = None

        if spec.lstrip("-").isdigit():
            hdu = hdul[int(spec)]
        else:
            try:
                hdu = hdul[spec]
            except Exception:
                target = spec.strip().upper()
                for cand in hdul:
                    if getattr(cand, "name", "").strip().upper() == target:
                        hdu = cand
                        break
                if hdu is None:
                    names = [getattr(x, "name", "") for x in hdul]
                    raise KeyError(
                        f"Extension '{spec}' not found in {path}. Available EXTNAMEs: {names}"
                    )

        data = hdu.data

    if data is None:
        raise ValueError(f"No data in {path} (HDU={hdu_spec})")

    arr = np.asarray(data)
    arr = np.squeeze(arr)

    while arr.ndim > 2:
        arr = arr[0]

    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D image after squeezing; got shape {arr.shape} from {path} (HDU={hdu_spec})"
        )

    return arr.astype(np.float32, copy=False)


def normalize(
    img: np.ndarray,
    p_lo: float,
    p_hi: float,
    stretch: str = "linear",
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalize image to [0,1] using percentile clip and stretch. NaNs are ignored."""
    x = np.asarray(img, dtype=np.float32)

    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)

    vals = x[finite]
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)

    # Fallback if percentiles degenerate
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.min(vals)
        hi = np.max(vals)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x, dtype=np.float32)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)

    stretch = stretch.lower()
    if stretch == "linear":
        pass
    elif stretch == "log":
        a = 1000.0
        y = np.log1p(a * y) / np.log1p(a)
    elif stretch == "asinh":
        a = 10.0
        y = np.arcsinh(a * y) / np.arcsinh(a)
    else:
        raise ValueError(f"Unknown stretch: {stretch} (use linear|log|asinh)")

    if gamma and gamma > 0 and gamma != 1.0:
        y = np.power(y, 1.0 / float(gamma))

    y = y.astype(np.float32, copy=False)
    y[~finite] = np.nan
    return y


def build_overlay_alpha(
    norm_img: np.ndarray,
    alpha: float,
    transparent_below: float | None = None,
) -> np.ndarray:
    """
    Build per-pixel alpha for the overlay.
    - NaNs are always transparent.
    - If transparent_below is set, pixels with normalized value <= threshold
      are also transparent.
    """
    mask = np.isfinite(norm_img)

    if transparent_below is not None:
        mask &= np.nan_to_num(norm_img, nan=-np.inf) > float(transparent_below)

    return float(alpha) * mask.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Overlay two co-registered FITS images with adjustable opacity."
    )
    ap.add_argument("base_fits", help="Base FITS image (shown underneath)")
    ap.add_argument("overlay_fits", help="Overlay FITS image (shown on top with alpha)")
    ap.add_argument(
        "--base-hdu",
        type=str,
        default="0",
        help="HDU index (e.g. 0) or EXTNAME (e.g. MU_OBS) for base (default: 0)",
    )
    ap.add_argument(
        "--overlay-hdu",
        type=str,
        default="0",
        help="HDU index (e.g. 0) or EXTNAME (e.g. MU_OBS) for overlay (default: 0)",
    )

    ap.add_argument(
        "--alpha", type=float, default=0.5, help="Initial overlay opacity (0..1)"
    )

    ap.add_argument(
        "--base-stretch", type=str, default="asinh", choices=["linear", "log", "asinh"]
    )
    ap.add_argument(
        "--overlay-stretch",
        type=str,
        default="asinh",
        choices=["linear", "log", "asinh"],
    )
    ap.add_argument(
        "--base-percentiles",
        type=float,
        nargs=2,
        default=(50.0, 90.0),
        metavar=("PLO", "PHI"),
    )
    ap.add_argument(
        "--overlay-percentiles",
        type=float,
        nargs=2,
        default=(50.0, 98.0),
        metavar=("PLO", "PHI"),
    )

    ap.add_argument(
        "--overlay-transparent-below",
        type=float,
        default=None,
        help=(
            "If set, make overlay pixels with normalized value <= this threshold "
            "transparent (range 0..1). Example: 0.0 or 0.02"
        ),
    )
    ap.add_argument("--base-gamma", type=float, default=1.0)
    ap.add_argument("--overlay-gamma", type=float, default=1.0)

    ap.add_argument("--cmap-base", type=str, default="gray")
    ap.add_argument("--cmap-overlay", type=str, default="magma")

    ap.add_argument("--flipx", action="store_true", help="Flip both images left-right")
    ap.add_argument("--flipy", action="store_true", help="Flip both images up-down")
    ap.add_argument(
        "--rot90", type=int, default=0, help="Rotate both images by k*90 deg (k=0..3)"
    )

    ap.add_argument(
        "--save", type=str, default=None, help="Save a PNG (path) immediately on launch"
    )
    ap.add_argument("--title", type=str, default=None, help="Figure title")

    args = ap.parse_args()

    base = read_fits_2d(args.base_fits, args.base_hdu)
    over = read_fits_2d(args.overlay_fits, args.overlay_hdu)

    if base.shape != over.shape:
        raise ValueError(f"Shape mismatch: base {base.shape} vs overlay {over.shape}")

    k = int(args.rot90) % 4
    if args.flipx:
        base = np.fliplr(base)
        over = np.fliplr(over)
    if args.flipy:
        base = np.flipud(base)
        over = np.flipud(over)
    if k:
        base = np.rot90(base, k=k)
        over = np.rot90(over, k=k)

    base_n = normalize(
        base, *args.base_percentiles, stretch=args.base_stretch, gamma=args.base_gamma
    )
    over_n = normalize(
        over,
        *args.overlay_percentiles,
        stretch=args.overlay_stretch,
        gamma=args.overlay_gamma,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.28)

    import matplotlib as mpl

    cmap_base = mpl.colormaps.get_cmap(args.cmap_base).copy()
    cmap_over = mpl.colormaps.get_cmap(args.cmap_overlay).copy()

    cmap_base.set_bad(color="black", alpha=1.0)
    cmap_over.set_bad(color="black", alpha=1.0)
    cmap_base.set_under("black")
    cmap_over.set_under("black")

    im_base = ax.imshow(
        base_n,
        origin="lower",
        cmap=cmap_base,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    alpha0 = build_overlay_alpha(
        over_n,
        args.alpha,
        transparent_below=args.overlay_transparent_below,
    )

    im_over = ax.imshow(
        over_n,
        origin="lower",
        cmap=cmap_over,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        alpha=alpha0,
    )

    ax.set_axis_off()
    if args.title:
        ax.set_title(args.title)

    # --- Sliders ---
    ax_base_pct = fig.add_axes([0.18, 0.16, 0.58, 0.03])
    s_base_pct = RangeSlider(
        ax_base_pct,
        "Base %",
        0.0,
        100.0,
        valinit=tuple(args.base_percentiles),
        valstep=0.1,
    )

    ax_over_pct = fig.add_axes([0.18, 0.10, 0.58, 0.03])
    s_over_pct = RangeSlider(
        ax_over_pct,
        "Overlay %",
        0.0,
        100.0,
        valinit=tuple(args.overlay_percentiles),
        valstep=0.1,
    )

    ax_alpha = fig.add_axes([0.18, 0.04, 0.58, 0.03])
    s_alpha = Slider(
        ax_alpha, "Overlay α", 0.0, 1.0, valinit=float(args.alpha), valstep=0.01
    )

    def update_base(_val=None):
        p_lo, p_hi = map(float, s_base_pct.val)
        if p_hi <= p_lo:
            return
        new_base = normalize(
            base,
            p_lo,
            p_hi,
            stretch=args.base_stretch,
            gamma=args.base_gamma,
        )
        im_base.set_data(new_base)
        fig.canvas.draw_idle()

    def update_overlay(_val=None):
        p_lo, p_hi = map(float, s_over_pct.val)
        if p_hi <= p_lo:
            return

        new_over = normalize(
            over,
            p_lo,
            p_hi,
            stretch=args.overlay_stretch,
            gamma=args.overlay_gamma,
        )
        im_over.set_data(new_over)

        a = build_overlay_alpha(
            new_over,
            s_alpha.val,
            transparent_below=args.overlay_transparent_below,
        )
        im_over.set_alpha(a)
        fig.canvas.draw_idle()

    def on_alpha(val):
        current_over = np.asarray(im_over.get_array())
        a = build_overlay_alpha(
            current_over,
            val,
            transparent_below=args.overlay_transparent_below,
        )
        im_over.set_alpha(a)
        fig.canvas.draw_idle()

    s_base_pct.on_changed(update_base)
    s_over_pct.on_changed(update_overlay)
    s_alpha.on_changed(on_alpha)

    # Save button
    ax_btn = fig.add_axes([0.80, 0.07, 0.15, 0.07])
    btn = Button(ax_btn, "Save PNG")

    def do_save(_evt=None):
        out = args.save
        if out is None:
            out = "overlay.png"
        out = str(Path(out).expanduser())

        # Crop output to the image axes only (excludes sliders/buttons)
        fig.canvas.draw()  # ensure positions are finalized
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        fig.savefig(
            out,
            dpi=200,
            bbox_inches=bbox,
            pad_inches=0,
        )
        print(f"Saved: {out}")

    btn.on_clicked(do_save)

    if args.save:
        do_save()

    plt.show()


if __name__ == "__main__":
    main()
