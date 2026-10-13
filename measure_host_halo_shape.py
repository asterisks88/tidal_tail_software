#!/usr/bin/env python3
"""Measure the host DM density shape in a Gadget HDF5 snapshot.

Requirements: Python >=3.9, numpy, h5py, matplotlib (for plots).
Install: pip install numpy h5py matplotlib

Examples (replace all illustrative selections/radii with your own):
  python measure_host_halo_shape.py snapshot_058.hdf5 --inspect
  python measure_host_halo_shape.py snapshot_058.hdf5 --dm-types 1 \
      --galaxyid-map model.hdf5.galaxyid.json --host-name LEDA175897 --rmax 300
  python measure_host_halo_shape.py snapshot_058.hdf5 --dm-types 1 \
      --host-field GalaxyID --host-value 0 --rmax 300 --shell-edges 20 50 100 200 300
  python measure_host_halo_shape.py snapshot_058.0.hdf5 --dm-types 1 \
      --host-id-range 1 1000000 --rmax 300 --out host_shape.json
  python measure_host_halo_shape.py snapshot_058.hdf5 --dm-types 1 \
      --all-selected-are-host --center 0 0 0 --rmax 300

Host identity MUST be explicit. Gadget does not standardize galaxy tags.
Particle ordering is not a persistent host identifier. ID ranges are inclusive;
--host-ids accepts a plain-text integer list or a one-dimensional integer .npy.
--galaxyid-map accepts the gadget_movie.py-compatible JSON sidecar, using
GalaxyID_IDStart/GalaxyID_IDEnd (inclusive, already offset ParticleIDs).
The first entry in original JSON order is the default host, matching the
movie script. Override with --host-name or zero-based --host-index.
GalaxyID_IDMin/IDMax are pre-offset IDs and are NOT used for selection.
Only --dm-types particles within the host range enter the shape calculation.
Map paths are relative to the working directory; referenced Infile/outfile
paths are descriptive metadata only and are never opened.
Lagrangian host tags select host-origin DM, including stripped particles; they
do not identify the currently bound halo or remove host-origin substructure.
No binding-energy, velocity, or automatic substructure selection is performed.

Definition: q_DM = c/a, with a >= b >= c; also report b/a and
T=(a^2-b^2)/(a^2-c^2). q_DM equals the usual oblate polar/equatorial
density ratio only when b/a is near unity. It is not potential flattening.
For a prolate halo the sorted c/a is still <=1 (not polar/equatorial >1).

Algorithm: iterate S_ij = sum(m*x_i*x_j)/sum(m) in an ellipsoid,
starting spherical. Diagonalize, take sqrt(eigenvalue ratios), rotate and
reshape the selection, holding the outer MAJOR semi-axis fixed. There is
no additional r^-2 weighting. Ellipsoidal radius is
s^2 = X^2 + (Y/(b/a))^2 + (Z/(c/a))^2.
The global result uses rmin <= s < rmax (an enclosed ellipsoid if rmin=0).
It is an effective second-moment shape, NOT an arithmetic mean of local
isodensity axis ratios. Outer particles have more moment leverage.
Optional independently fitted homoeoidal shells estimate local shapes.
See Zemp et al. 2011, ApJS 197, 30, https://arxiv.org/abs/1107.5582 .
Their shells are preferable when radial variations matter. Independently
fitted shells can overlap if orientations/ratios vary; no shell mean is made.

Center: explicit --center is held fixed; otherwise a shrinking-sphere
mass centroid of selected host DM is used. --center-guess supplies its
initial point; --center-radius restricts its initial sphere. Inspect the
estimated center for disturbed/merging haloes. --center-min controls the
smallest retained core population (default 1000, or all particles if fewer).

All input lengths use raw snapshot coordinate units, with NO automatic
cosmological a/h conversion. Ratios are unit independent. --box-size enables
minimum-image periodic wrapping (off by default for isolated encounters).
Choose rmax below half the box when using wrapping. Velocities are unused.
Split snapshots are auto-discovered from any numbered component or basename;
all expected pieces must exist. Selected arrays are held in memory.

Outputs: console and JSON including selection, center, convergence, particle
counts, effective counts, principal axes (columns in simulation coordinates),
and optional shell results. Failure to converge is flagged, not hidden.
No statistical error estimate is implied by the convergence tolerance.
Shape estimates in shallow/constant-density cores are poorly constrained;
convergence alone does not guarantee accuracy. Check particle resolution,
centering, and radial-range sensitivity before reporting a paper measurement.

Uncertainties: --bootstrap 200 (default) resamples N selected host particles
with replacement, keeping their masses. Each draw re-estimates the center
using the same shrinking-sphere procedure and refits all regions. This jointly
includes particle noise and statistical centering uncertainty; do not add a
separate centering error in quadrature. Shared draws preserve correlations.
--fixed-bootstrap-center holds the nominal center fixed for comparison.
An explicit --center is treated as exact unless --center-sigma SX SY SZ supplies
independent Gaussian coordinate uncertainties (snapshot units). This external
center prior is sampled jointly with particle noise, not in addition to an
automatically estimated center. The result does not capture systematic bias
in the centering algorithm, force resolution, or non-independent particles.
Errors are bootstrap sample standard deviations, not divided by sqrt(B).
Percentiles, paired samples, and center displacement covariance are saved.
Require >=80% successful draws and >=20 successes for a reported error.
Use --bootstrap 0 to disable, --seed for reproducibility. Cost scales with B.
Speed: --workers 4 runs four independent bootstrap realizations concurrently
using shared read-only inputs and private working arrays. Install threadpoolctl
for parallel runs: pip install threadpoolctl. Each active worker needs working
memory proportional to the host particle count (potentially hundreds of MB
per million particles); start with 2-4 workers. Memory bandwidth can limit
speedup. Default remains 1 to avoid unexpectedly increasing memory use.
When threadpoolctl is installed, BLAS uses one thread per bootstrap worker.
--bootstrap 50 is useful for exploratory runs (less stable estimated errors);
--bootstrap 0 skips errors altogether. Full recentering remains the default.
Per-draw seeds make serial and parallel samples identical at a given --seed;
these samples differ from the older script's single random-number stream.
Near equal axes, sorting biases ratios; bootstrap scatter is not a bias
correction or a guarantee of confidence-interval coverage.

With shells, shell_range_shape fits one effective ellipsoid between the first
and last shell edges (independent of --rmin/--rmax). This is the recommended
single effective q_DM, not the mean of local shell ratios. Also report
shell_radial_mean_q = sum(delta_r * q_shell)/sum(delta_r), an explicitly
radial-width-weighted descriptive average. Its uncertainty comes from paired
bootstrap draws, including inter-shell correlations; all shells must converge.
shell_radial_mean_T similarly averages shell T with radial-width weights.
It is not T calculated from averaged axis ratios. If any shell T is undefined,
the full-range mean T is undefined; such bootstrap draws are excluded from
its uncertainty rather than silently averaging over fewer shells.
Neither summary captures radial variation or twisting of principal axes.
Triaxiality T=(1-(b/a)^2)/(1-(c/a)^2) is computed for every fit and bootstrap
draw. T=0 is oblate, T=1 prolate; commonly T<1/3 is oblate-like, T>2/3
prolate-like, with intermediate values triaxial (Vega-Ferrero et al. 2017,
MNRAS 467, 3226). T is undefined for a sphere and unstable near sphericity;
null is used when 1-(c/a)^2 <= 1e-6. Interpret T alongside both axis ratios.

With --shell-edges a PNG is automatically saved beside the JSON. --plot PATH
overrides its path (PNG/PDF/SVG); --no-plot disables it. Plot points are the
original fits, vertical bars are +/- bootstrap standard deviation, horizontal
bars show shell extent. Radius is the arithmetic midpoint of major-axis shell
edges. --radius-unit kpc changes the label only, never the coordinate scale.
"""
import argparse
import json
import re
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np


def snapshot_files(name):
    p = Path(name).expanduser()
    if not p.is_file():
        candidates = [Path(str(p) + s) for s in ('.hdf5', '.0.hdf5', '.h5', '.0.h5')]
        p = next((v for v in candidates if v.is_file()), p)
    if not p.is_file():
        raise ValueError(f'Snapshot not found: {name}')
    with h5py.File(p, 'r') as f:
        h = dict(f['Header'].attrs)
    n = int(h.get('NumFilesPerSnapshot', h.get('NumFiles', 1)))
    if n < 1:
        raise ValueError('Invalid snapshot file count')
    if n == 1:
        return [p], h
    match = re.match(r'^(.*)\.\d+(\.hdf5|\.h5)$', str(p))
    if not match:
        raise ValueError('Split snapshot must use basename.N.hdf5 or basename.N.h5')
    files = [Path(f'{match[1]}.{i}{match[2]}') for i in range(n)]
    missing = [str(v) for v in files if not v.is_file()]
    if missing:
        raise ValueError(f'Missing snapshot pieces: {missing}')
    return files, h


def inspect(files):
    for p in files:
        print(f'\n{p}')
        with h5py.File(p, 'r') as f:
            print('Header:', {k: np.asarray(v).tolist() for k, v in f['Header'].attrs.items()})
            for name in f:
                if not name.startswith('PartType'):
                    continue
                print(f'  {name}')
                for key, d in f[name].items():
                    if not isinstance(d, h5py.Dataset):
                        continue
                    print(f'    {key}: shape={d.shape}, dtype={d.dtype}')
                    if d.ndim == 1 and d.size and (key == 'ParticleIDs' or
                                                       ('id' in key.lower())):
                        values = d[:]
                        print(f'      range: {values.min()} .. {values.max()}')
                        if key != 'ParticleIDs':
                            u = np.unique(values)
                            if len(u) <= 20:
                                print(f'      values: {u.tolist()}')


def read_galaxyid_map(path, host_name=None, host_index=None):
    """Resolve one original-order galaxy entry without losing uint64 IDs."""
    path = Path(path).expanduser()
    mapping = json.loads(path.read_text())
    if not isinstance(mapping, dict):
        raise ValueError('GalaxyID map must be a JSON object')
    starts, ends = mapping.get('GalaxyID_IDStart'), mapping.get('GalaxyID_IDEnd')
    if not isinstance(starts, list) or not isinstance(ends, list) or not starts or len(starts) != len(ends):
        raise ValueError('GalaxyID_IDStart/IDEnd must be nonempty arrays of equal length')
    max_id = int(np.iinfo(np.uint64).max)
    if any(type(v) is not int or not 0 <= v <= max_id for v in starts + ends):
        raise ValueError('GalaxyID range endpoints must be nonnegative uint64 integers')
    if any(end < start for start, end in zip(starts, ends)):
        raise ValueError('GalaxyID range end precedes start')
    ranges = sorted(zip(starts, ends))
    if any(right[0] <= left[1] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('GalaxyID ParticleID ranges overlap')
    names = mapping.get('GalaxyID_Name', [])
    if not isinstance(names, list) or (names and len(names) != len(starts)) or any(not isinstance(v, str) for v in names):
        raise ValueError('GalaxyID_Name must be absent, empty, or one string per range')
    if host_name is not None:
        matches = [i for i, name in enumerate(names) if name == host_name]
        if len(matches) != 1:
            raise ValueError(f'Host name {host_name!r} must match exactly one entry; available names: {names}')
        index = matches[0]
    else:
        index = 0 if host_index is None else host_index
    if not 0 <= index < len(starts):
        raise ValueError(f'Host index must be in 0..{len(starts)-1} (original JSON order)')
    return {'source': str(path.resolve()), 'host_index': index,
            'host_name': names[index] if names else None,
            'particle_id_min': starts[index], 'particle_id_max': ends[index]}


def read_particles(files, args, mapped_range=None):
    id_range = mapped_range if mapped_range is not None else args.host_id_range
    wanted = None
    if args.host_ids:
        p = Path(args.host_ids)
        wanted = np.load(p, allow_pickle=False) if p.suffix == '.npy' else np.loadtxt(p, dtype=np.uint64, ndmin=1)
        if wanted.ndim != 1 or wanted.dtype.kind not in 'iu' or np.any(wanted < 0):
            raise ValueError('Host ID list must be one-dimensional nonnegative integers')
        wanted = wanted.astype(np.uint64)
        if not len(wanted):
            raise ValueError('Host ID list is empty')
    positions, masses = [], []
    counts = {str(t): 0 for t in args.dm_types}
    for path in files:
        with h5py.File(path, 'r') as f:
            table = f['Header'].attrs.get('MassTable', [])
            for t in args.dm_types:
                key = f'PartType{t}'
                if key not in f:
                    continue
                g = f[key]
                n = len(g['Coordinates'])
                if not n:
                    continue
                keep = np.ones(n, dtype=bool)
                if args.host_field:
                    if args.host_field not in g:
                        raise ValueError(f'{path}: {key}/{args.host_field} missing')
                    field = g[args.host_field][:]
                    if field.shape != (n,):
                        raise ValueError('Host field must be a scalar per particle')
                    keep = field == args.host_value
                elif id_range is not None or wanted is not None:
                    ids = g['ParticleIDs'][:]
                    if ids.shape != (n,) or ids.dtype.kind not in 'iu':
                        raise ValueError('ParticleIDs must be one-dimensional integers')
                    if id_range is not None:
                        lo, hi = id_range
                        keep = (ids >= lo) & (ids <= hi)
                    else:
                        keep = np.isin(ids, wanted)
                x = np.asarray(g['Coordinates'][:], dtype=float)[keep]
                if x.shape != (int(keep.sum()), 3):
                    raise ValueError('Coordinates must have shape (N,3)')
                if 'Masses' in g:
                    m = np.asarray(g['Masses'][:], dtype=float)[keep]
                elif t < len(table) and table[t] > 0:
                    m = np.full(len(x), float(table[t]))
                else:
                    raise ValueError(f'{key}: neither Masses nor positive MassTable entry')
                if m.shape != (len(x),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(m)) or np.any(m <= 0):
                    raise ValueError('Invalid selected coordinates/masses; masses must be positive')
                positions.append(x)
                masses.append(m)
                counts[str(t)] += len(x)
    if not positions or not sum(counts.values()):
        raise ValueError('No selected host DM particles; check selection')
    return np.concatenate(positions), np.concatenate(masses), counts


def delta(x, c, box):
    d = x - c
    return d if box is None else d - box * np.rint(d / box)


def find_center(x, m, args):
    if args.center is not None:
        return np.array(args.center), {'method': 'fixed'}
    if args.center_guess is not None:
        c = np.array(args.center_guess)
    else:
        c = np.average(x, axis=0, weights=m)
    d = delta(x, c, args.box_size)
    radius = args.center_radius if args.center_radius else float(np.max(np.linalg.norm(d, axis=1))) * 1.000001
    mask = np.linalg.norm(d, axis=1) <= radius
    if mask.sum() < args.min_particles:
        raise ValueError('Too few particles in initial centering sphere')
    target = min(args.center_min, int(mask.sum()))
    steps = 0
    for steps in range(200):
        c = c + np.average(delta(x[mask], c, args.box_size), axis=0, weights=m[mask])
        radius_next = radius * 0.9
        next_mask = np.linalg.norm(delta(x, c, args.box_size), axis=1) <= radius_next
        if next_mask.sum() < target:
            break
        radius, mask = radius_next, next_mask
    if args.box_size:
        c %= args.box_size
    return c, {'method': 'shrinking_sphere', 'core_count': int(mask.sum()),
               'last_radius': radius, 'iterations': steps + 1}


def fit_shape(x, m, inner, outer, min_particles=1000, tol=1e-3, max_iter=200):
    # Since a is fixed and b,c <= a, every iterated ellipsoid is contained
    # within the initial outer sphere. This prefilter is therefore safe.
    inside = np.sum(x*x, axis=1) < outer*outer
    x, m = x[inside], m[inside]
    axes = np.eye(3)
    ratios = np.ones(3)
    metric = np.eye(3)
    converged = False
    selected = np.zeros(len(x), dtype=bool)
    residual = None
    for iteration in range(1, max_iter+1):
        coord = x @ axes
        s2 = np.sum((coord / ratios)**2, axis=1)
        selected = (s2 >= inner**2) & (s2 < outer**2)
        if selected.sum() < min_particles:
            raise ValueError(f'Only {selected.sum()} particles in [{inner:g}, {outer:g}); need {min_particles}')
        y, w = x[selected], m[selected]
        tensor = (y.T @ (y*w[:, None])) / w.sum()
        eigenvalues, new_axes = np.linalg.eigh(tensor)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, new_axes = eigenvalues[order], new_axes[:, order]
        if eigenvalues[-1] <= 0:
            raise ValueError('Degenerate shape tensor')
        new_ratios = np.sqrt(eigenvalues/eigenvalues[0])
        new_metric = (new_axes * (1/new_ratios**2)) @ new_axes.T
        # Metric convergence also tests orientation without eigenvector-sign
        # problems or undefined orientations in spherical/axisymmetric cases.
        residual = float(np.linalg.norm(new_metric-metric)/np.linalg.norm(metric))
        ratio_change = float(np.max(np.abs(new_ratios/ratios-1)))
        axes, ratios, metric = new_axes, new_ratios, new_metric
        if residual < tol and ratio_change < tol:
            converged = True
            break
    # Counts refer to the last tensor's actual contributing set. At convergence
    # its selection and reported ellipsoid agree to the specified tolerance.
    b, c = map(float, ratios[1:])
    mass = float(m[selected].sum())
    neff = float(mass**2/np.sum(m[selected]**2))
    return {'rmin_major': inner, 'rmax_major': outer, 'q_DM': c, 'b_over_a': b,
            'triaxiality': (1-b*b)/(1-c*c) if 1-c*c > 1e-6 else None,
            'n_particles': int(selected.sum()), 'effective_n': neff, 'mass_snapshot_units': mass,
            'axes_columns_major_intermediate_minor': axes.tolist(),
            'converged': converged, 'iterations': iteration, 'metric_residual': residual}


def summarize_bootstrap(samples, keys, nominal, n_boot, draw_indices):
    valid = len(samples) >= max(20, int(np.ceil(.8*n_boot)))
    result = {'requested': n_boot, 'successful': len(samples),
              'failed': n_boot-len(samples), 'valid': valid,
              'sample_columns': keys, 'samples': samples, 'draw_indices': draw_indices}
    if not valid:
        return result
    values = np.array([[np.nan if v is None else v for v in row] for row in samples])
    for j, key in enumerate(keys):
        v = values[:, j]; v = v[np.isfinite(v)]
        if len(v) < max(20, int(np.ceil(.8*n_boot))) or nominal[key] is None:
            result[key] = None
            continue
        pct = np.percentile(v, [16, 50, 84])
        result[key] = {'std': float(v.std(ddof=1)), 'p16': float(pct[0]),
                       'median': float(pct[1]), 'p84': float(pct[2]),
                       'bias_mean_minus_fit': float(v.mean()-nominal[key]),
                       'successful': len(v)}
    if len(keys) >= 2:
        result['axis_ratio_covariance'] = np.cov(values[:, :2], rowvar=False).tolist()
    return result


def bootstrap_draw(seed, x, m, center, shapes, args, fit_kwargs):
    # Workers only read shared inputs; all particle resampling arrays are local.
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), size=len(x))
    bx, bm = x[ix], m[ix]
    try:
        if args.center_sigma is not None:
            bc = center + rng.normal(size=3)*np.array(args.center_sigma)
        elif args.fixed_bootstrap_center or args.center is not None:
            bc = center
        else:
            bc, _ = find_center(bx, bm, args)
        shift = delta(np.array([bc]), center, args.box_size)[0].tolist()
        relative = delta(bx, bc, args.box_size)
    except (ValueError, np.linalg.LinAlgError):
        return None, {}
    fitted = {}
    for shape in shapes:
        if not shape.get('converged'):
            continue
        bounds = (shape['rmin_major'], shape['rmax_major'])
        if bounds not in fitted:
            try:
                trial = fit_shape(relative, bm, *bounds, **fit_kwargs)
                fitted[bounds] = trial if trial['converged'] else None
            except (ValueError, np.linalg.LinAlgError):
                fitted[bounds] = None
    return shift, fitted


def joint_bootstrap(x, m, center, shapes, shells, args, fit_kwargs):
    keys = ['q_DM', 'b_over_a', 'triaxiality']
    rows = [[] for _ in shapes]; indices = [[] for _ in shapes]
    shifts = []; center_indices = []; mean_rows = []; mean_indices = []
    widths = np.array([v['rmax_major']-v['rmin_major'] for v in shells])
    seeds = np.random.SeedSequence(args.seed).spawn(args.bootstrap)
    workers = min(args.workers, args.bootstrap)
    # BLAS may otherwise start many threads for each small (N,3) operation,
    # oversubscribing the CPU when several realizations run concurrently.
    try:
        from threadpoolctl import threadpool_limits
        blas_context = threadpool_limits(limits=1, user_api='blas')
    except ImportError:
        if workers > 1:
            raise ValueError('Parallel bootstrap requires threadpoolctl: pip install threadpoolctl')
        blas_context = nullcontext()
    start = time.monotonic()
    def run(seed):
        return bootstrap_draw(seed, x, m, center, shapes, args, fit_kwargs)
    with blas_context, (ThreadPoolExecutor(max_workers=workers) if workers > 1 else nullcontext()) as executor:
        results = executor.map(run, seeds) if executor is not None else map(run, seeds)
        # Ordered collection plus one seed per draw makes worker count irrelevant
        # to the bootstrap samples and result ordering.
        for draw, (shift, fitted) in enumerate(results):
            if shift is not None:
                shifts.append(shift); center_indices.append(draw)
            for j, shape in enumerate(shapes):
                trial = fitted.get((shape['rmin_major'], shape['rmax_major']))
                if trial is not None:
                    rows[j].append([trial[k] for k in keys]); indices[j].append(draw)
            if shells:
                trials = [fitted.get((v['rmin_major'], v['rmax_major'])) for v in shells]
                if all(t is not None for t in trials):
                    mean_t = (float(np.average([t['triaxiality'] for t in trials], weights=widths))
                              if all(t['triaxiality'] is not None for t in trials) else None)
                    mean_rows.append([float(np.average([t['q_DM'] for t in trials], weights=widths)), mean_t])
                    mean_indices.append(draw)
            if (draw+1)%10 == 0 or draw+1 == args.bootstrap:
                elapsed = time.monotonic()-start
                eta = elapsed/(draw+1)*(args.bootstrap-draw-1)
                print(f'Joint bootstrap: {draw+1}/{args.bootstrap}; workers={workers}; '
                      f'elapsed={elapsed:.0f}s; estimated remaining={eta:.0f}s', flush=True)
    for j, shape in enumerate(shapes):
        if shape.get('converged'):
            shape['bootstrap'] = summarize_bootstrap(rows[j], keys, shape, args.bootstrap, indices[j])
    centering = {'mode': 'external_gaussian' if args.center_sigma is not None else
                 ('fixed' if args.fixed_bootstrap_center or args.center is not None else 'reestimated'),
                 'successful': len(shifts), 'draw_indices': center_indices,
                 'displacements': shifts}
    if len(shifts) >= 2:
        centering['std_xyz'] = np.std(shifts, axis=0, ddof=1).tolist()
        centering['covariance_xyz'] = np.cov(shifts, rowvar=False).tolist()
    return centering, mean_rows, mean_indices


def plot_profile(shells, path, unit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    no_errors = False
    for key, label, marker, color in [('q_DM', r'$c/a$', 'o', '#2166ac'),
                                      ('b_over_a', r'$b/a$', 's', '#d95f02')]:
        # NaNs leave gaps at failed shells instead of bridging them with lines.
        radii = [(s['rmin_major']+s['rmax_major'])/2 for s in shells]
        vals = [s[key] if s.get('converged') and key in s else np.nan for s in shells]
        ax.plot(radii, vals, color=color, alpha=.5, linewidth=1)
        labeled = False
        for r, val, s in zip(radii, vals, shells):
            if not np.isfinite(val):
                continue
            boot = s.get('bootstrap', {})
            valid = boot.get('valid', False)
            no_errors |= not valid
            ax.errorbar(r, val, xerr=(s['rmax_major']-s['rmin_major'])/2,
                        yerr=boot[key]['std'] if valid else None,
                        fmt=marker, color=color, capsize=3,
                        markerfacecolor=color if valid else 'white',
                        label=label if not labeled else None)
            labeled = True
    ax.set_xlabel(f'Major-axis shell radius ({unit})')
    ax.set_ylabel('Density axis ratio')
    ax.set_title('Host dark-matter halo shape')
    if shells:
        low, high = shells[0]['rmin_major'], shells[-1]['rmax_major']
        pad = .03*(high-low)
        ax.set_xlim(max(0, low-pad), high+pad)
    ax.axhline(1, color='.5', ls=':', lw=1)
    ax.grid(alpha=.2)
    # Do not truncate statistical bars at the physical boundaries 0 or 1.
    if any(s.get('converged') for s in shells):
        ax.legend(loc='best')
    note = 'Vertical bars: bootstrap 1 sigma; horizontal bars: shell extent'
    if no_errors:
        note += '\nOpen symbols: uncertainty unavailable or disabled'
    if any(not s.get('converged') for s in shells):
        note += '\nFailed/nonconverged shell fits omitted'
    fig.supxlabel(note, fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('snapshot')
    p.add_argument('--inspect', action='store_true', help='List datasets/header without fitting')
    p.add_argument('--dm-types', nargs='+', type=int, help='Explicit DM particle types, e.g. 1')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--galaxyid-map', '--galaxy-id-map', help='GalaxyID JSON sidecar with inclusive ParticleID ranges')
    selection.add_argument('--host-field', help='Dataset within PartType groups, e.g. GalaxyID or gid')
    selection.add_argument('--host-id-range', nargs=2, type=int, metavar=('MIN', 'MAX'))
    selection.add_argument('--host-ids', help='Text or integer .npy host particle ID list')
    selection.add_argument('--all-selected-are-host', action='store_true', help='Assert chosen types contain only host DM')
    p.add_argument('--host-value', type=int, help='Integer host tag paired with --host-field')
    host = p.add_mutually_exclusive_group()
    host.add_argument('--host-name', help='Exact GalaxyID_Name entry; requires --galaxyid-map')
    host.add_argument('--host-index', type=int, help='Zero-based original JSON entry; default 0; requires --galaxyid-map')
    p.add_argument('--rmin', type=float, default=0)
    p.add_argument('--rmax', type=float, help='Outer major semi-axis in snapshot length units')
    p.add_argument('--shell-edges', type=float, nargs='+', help='Optional increasing shell edges in same units')
    p.add_argument('--center', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    p.add_argument('--center-guess', type=float, nargs=3)
    p.add_argument('--center-radius', type=float)
    p.add_argument('--center-min', type=int, default=1000)
    p.add_argument('--box-size', type=float, help='Explicit periodic box length; default no wrapping')
    p.add_argument('--min-particles', type=int, default=1000)
    p.add_argument('--tol', type=float, default=1e-3)
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--out', default='host_halo_shape.json')
    p.add_argument('--bootstrap', type=int, default=200, metavar='B', help='Bootstrap refits per region: 0 or >=20 (default 200)')
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--workers', type=int, default=1, help='Concurrent bootstrap realizations (default 1); try 2 or 4; more RAM needed')
    p.add_argument('--fixed-bootstrap-center', action='store_true', help='Disable re-estimation of the center in bootstrap draws')
    p.add_argument('--center-sigma', type=float, nargs=3, metavar=('SX','SY','SZ'), help='Independent Gaussian uncertainties on explicit --center')
    plots = p.add_mutually_exclusive_group()
    plots.add_argument('--plot', help='Output PNG/PDF/SVG; default JSON stem plus .png when shells requested')
    plots.add_argument('--no-plot', action='store_true')
    p.add_argument('--radius-unit', default='snapshot length units', help='Plot label only; e.g. kpc')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    files, header = snapshot_files(a.snapshot)
    if a.inspect:
        inspect(files)
        return 0
    if a.bootstrap != 0 and a.bootstrap < 20:
        p.error('--bootstrap must be 0 or >=20')
    if a.center_sigma is not None:
        if a.center is None or a.fixed_bootstrap_center:
            p.error('--center-sigma requires --center and cannot accompany --fixed-bootstrap-center')
        if not np.all(np.isfinite(a.center_sigma)) or min(a.center_sigma) < 0:
            p.error('--center-sigma must contain finite nonnegative values')
    if a.workers < 1:
        p.error('--workers must be >=1')
    if a.workers > 1 and a.bootstrap:
        try:
            import threadpoolctl
        except ImportError:
            p.error('Parallel bootstrap requires threadpoolctl: pip install threadpoolctl')
    if a.seed < 0:
        p.error('--seed must be nonnegative')
    if a.plot and not a.shell_edges:
        p.error('--plot requires --shell-edges')
    plot_path = None
    if a.shell_edges and not a.no_plot:
        plot_path = Path(a.plot) if a.plot else Path(a.out).with_suffix('.png')
        if plot_path.suffix.lower() not in ('.png', '.pdf', '.svg'):
            p.error('--plot must end in .png, .pdf, or .svg')
        if plot_path.resolve() == Path(a.out).resolve():
            p.error('Plot and JSON output paths must differ')
        try:
            import matplotlib
        except ImportError:
            p.error('Plotting requires matplotlib: pip install matplotlib (or use --no-plot)')
    if not a.dm_types or min(a.dm_types) < 0 or len(set(a.dm_types)) != len(a.dm_types):
        p.error('Specify distinct nonnegative --dm-types')
    if not (a.host_field or a.host_id_range or a.host_ids or a.all_selected_are_host or a.galaxyid_map):
        p.error('Explicit host membership selection is required; see --help')
    if (a.host_name is not None or a.host_index is not None) and not a.galaxyid_map:
        p.error('--host-name/--host-index require --galaxyid-map')
    if (a.host_field is None) != (a.host_value is None):
        p.error('--host-field and --host-value must be used together')
    if a.host_id_range and not (0 <= a.host_id_range[0] <= a.host_id_range[1] <= np.iinfo(np.uint64).max):
        p.error('Invalid inclusive ID range')
    if a.rmax is None or not (0 <= a.rmin < a.rmax < float('inf')):
        p.error('Require finite 0 <= rmin < rmax')
    if not (0 < a.tol < 1) or a.max_iter < 1 or a.min_particles < 10 or a.center_min < a.min_particles:
        p.error('Require 0<tol<1, max-iter>=1, center-min>=min-particles>=10')
    for key in ('box_size', 'center_radius'):
        val = getattr(a, key)
        if val is not None and not (0 < val < float('inf')):
            p.error(f'{key} must be finite and positive')
    for key in ('center', 'center_guess'):
        val = getattr(a, key)
        if val is not None and not np.all(np.isfinite(val)):
            p.error(f'{key} must be finite')
    if a.center is not None and (a.center_guess is not None or a.center_radius is not None):
        p.error('Fixed --center cannot be combined with centering controls')
    if a.box_size and a.center is None and a.center_guess is None:
        p.error('Periodic auto-centering requires --center-guess near the host')
    if a.shell_edges is not None:
        e = np.array(a.shell_edges)
        if len(e) < 2 or not np.all(np.isfinite(e)) or e[0] < 0 or np.any(np.diff(e) <= 0):
            p.error('Shell edges must be finite, nonnegative, and strictly increasing')
    if a.box_size and max([a.rmax] + (a.shell_edges or [])) >= a.box_size/2:
        p.error('Fitting radii must be below half the periodic box')
    map_info = None
    mapped_range = None
    if a.galaxyid_map:
        map_info = read_galaxyid_map(a.galaxyid_map, a.host_name, a.host_index)
        mapped_range = (map_info['particle_id_min'], map_info['particle_id_max'])
        print(f"Map host: {map_info['host_name'] or map_info['host_index']}; "
              f"inclusive ParticleIDs {mapped_range[0]}..{mapped_range[1]}")
    x, m, counts = read_particles(files, a, mapped_range)
    if len(x) < a.min_particles:
        raise ValueError('Insufficient selected host particles')
    center, center_info = find_center(x, m, a)
    centered = delta(x, center, a.box_size)
    kwargs = dict(min_particles=a.min_particles, tol=a.tol, max_iter=a.max_iter)
    global_shape = fit_shape(centered, m, a.rmin, a.rmax, **kwargs)
    shells = []
    if a.shell_edges:
        for lo, hi in zip(a.shell_edges[:-1], a.shell_edges[1:]):
            try:
                shells.append(fit_shape(centered, m, lo, hi, **kwargs))
            except ValueError as exc:
                shells.append({'rmin_major': lo, 'rmax_major': hi, 'error': str(exc), 'converged': False})
    shell_range_shape = None
    radial_mean = None
    radial_mean_t = None
    if shells:
        lo, hi = a.shell_edges[0], a.shell_edges[-1]
        try:
            shell_range_shape = fit_shape(centered, m, lo, hi, **kwargs)
        except ValueError as exc:
            shell_range_shape = {'rmin_major': lo, 'rmax_major': hi, 'converged': False, 'error': str(exc)}
        if all(v['converged'] for v in shells):
            radial_mean = {'q_DM': float(np.average([v['q_DM'] for v in shells],
                           weights=np.diff(a.shell_edges))),
                           'definition': 'radial-width-weighted mean of shell c/a',
                           'rmin': lo, 'rmax': hi}
            t_values = [v['triaxiality'] for v in shells]
            radial_mean_t = {'triaxiality': float(np.average(t_values, weights=np.diff(a.shell_edges)))
                             if all(v is not None for v in t_values) else None,
                             'definition': 'radial-width-weighted mean of shell T',
                             'rmin': lo, 'rmax': hi}
    shapes = [global_shape] + ([shell_range_shape] if shell_range_shape else []) + shells
    warnings = []
    center_bootstrap = None
    if a.bootstrap:
        center_bootstrap, mean_rows, mean_indices = joint_bootstrap(x, m, center, shapes, shells, a, kwargs)
        if radial_mean is not None:
            radial_mean['bootstrap'] = summarize_bootstrap([[r[0]] for r in mean_rows], ['q_DM'], radial_mean, a.bootstrap, mean_indices)
            if not radial_mean['bootstrap']['valid']:
                warnings.append('Too few jointly converged shell draws for mean-q uncertainty.')
        if radial_mean_t is not None and radial_mean_t['triaxiality'] is not None:
            valid_t = [(row[1], draw) for row, draw in zip(mean_rows, mean_indices) if row[1] is not None]
            radial_mean_t['bootstrap'] = summarize_bootstrap([[v] for v, _ in valid_t], ['triaxiality'],
                                                            radial_mean_t, a.bootstrap, [d for _, d in valid_t])
            if not radial_mean_t['bootstrap']['valid']:
                warnings.append('Too few jointly converged shells with defined T for mean-T uncertainty.')
        for shape in shapes:
            if not shape.get('converged'):
                continue
            boot = shape['bootstrap']
            region = f"[{shape['rmin_major']:g}, {shape['rmax_major']:g})"
            if not boot['valid']:
                warnings.append(f'{region}: insufficient converged bootstrap fits; uncertainty withheld.')
            elif boot['successful'] < a.bootstrap:
                warnings.append(f"{region}: {a.bootstrap-boot['successful']} bootstrap fits failed; errors use successes only.")
            if boot['valid'] and any(boot[k]['std'] < 5*a.tol for k in ('q_DM', 'b_over_a')):
                warnings.append(f'{region}: bootstrap scatter is close to solver tolerance; reduce --tol and check stability.')
            if boot['valid'] and 1-shape['q_DM'] < 3*boot['q_DM']['std']:
                warnings.append(f'{region}: close to spherical within sampling scatter; T is poorly conditioned.')
    if radial_mean_t is not None and radial_mean_t['triaxiality'] is None:
        warnings.append('Radial mean T is undefined because at least one shell T is undefined.')
    if shell_range_shape and not shell_range_shape['converged']:
        warnings.append('Full shell-range effective shape did not converge.')
    if not global_shape['converged']:
        warnings.append('Global fit did not converge; do not use as a final measurement.')
    if any(not s['converged'] for s in shells):
        warnings.append('One or more shell fits failed or did not converge.')
    if any(s.get('effective_n', 0) < 10000 for s in [global_shape] + shells):
        warnings.append('Some fits have fewer than 10000 effective particles; assess resolution sensitivity.')
    result = {'definition': 'q_DM=c/a; a>=b>=c; density second-moment shape',
              'method': 'iterative mass-weighted unreduced shape tensor; fixed major semi-axis',
              'length_units': 'raw snapshot coordinates; no a/h conversion',
              'files': [str(v.resolve()) for v in files], 'arguments': vars(a),
              'galaxyid_map_selection': map_info,
              'plot_path': str(plot_path.resolve()) if plot_path else None,
              'snapshot_time': float(header['Time']) if 'Time' in header else None,
              'selected_counts_by_type': counts, 'center': center.tolist(),
              'centering': center_info, 'center_bootstrap': center_bootstrap,
              'global': global_shape, 'shell_range_shape': shell_range_shape,
              'shell_radial_mean_q': radial_mean, 'shell_radial_mean_T': radial_mean_t,
              'shells': shells, 'warnings': warnings}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if plot_path:
        plot_profile(shells, plot_path, a.radius_unit)
        print(f'Saved {plot_path}')
    print('Host center:', center)
    if center_bootstrap and 'std_xyz' in center_bootstrap:
        print('Center bootstrap mode:', center_bootstrap['mode'])
        print('Center sigma (x,y,z), snapshot units:', center_bootstrap['std_xyz'])
    print('Region [major-axis radii]     q_DM=c/a     b/a       N       converged')
    for index, s in enumerate(shapes):
        if index == 1 and shell_range_shape:
            print('Full shell-range effective shape:')
        label = f"[{s['rmin_major']:g}, {s['rmax_major']:g})"
        if 'error' in s:
            print(label, 'FAILED:', s['error'])
        else:
            print(f"{label:28s} {s['q_DM']:.5f}    {s['b_over_a']:.5f}  {s['n_particles']:8d}  {s['converged']}")
            print('  T =', 'undefined' if s['triaxiality'] is None else f"{s['triaxiality']:.5f}")
            boot = s.get('bootstrap', {})
            if boot.get('valid'):
                print(f"  joint-bootstrap sigma: c/a={boot['q_DM']['std']:.5f}; b/a={boot['b_over_a']['std']:.5f}")
                if boot.get('triaxiality'):
                    print(f"  sigma(T)={boot['triaxiality']['std']:.5f}")
    if radial_mean is not None:
        print(f"Radial-width-weighted mean q_DM = {radial_mean['q_DM']:.5f}")
        if radial_mean.get('bootstrap', {}).get('valid'):
            print(f"  sigma(mean q_DM) = {radial_mean['bootstrap']['q_DM']['std']:.5f}")
    if radial_mean_t is not None:
        value = radial_mean_t['triaxiality']
        print('Radial-width-weighted mean T =', 'undefined' if value is None else f'{value:.5f}')
        if radial_mean_t.get('bootstrap', {}).get('valid'):
            print(f"  sigma(mean T) = {radial_mean_t['bootstrap']['triaxiality']['std']:.5f}")
    for warning in warnings:
        print('WARNING:', warning, file=sys.stderr)
    print(f'Saved {out}')
    good_errors = all(s.get('bootstrap', {}).get('valid', False) for s in shapes) if a.bootstrap else True
    good_mean = not a.bootstrap or radial_mean is None or radial_mean['bootstrap']['valid']
    if a.bootstrap and radial_mean_t is not None and radial_mean_t['triaxiality'] is not None:
        good_mean = good_mean and radial_mean_t['bootstrap']['valid']
    return 0 if all(s['converged'] for s in shapes) and good_errors and good_mean else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
