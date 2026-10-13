
LEGACY IMAGE
-----------------------------------------------------------
** star mask from Legacy image **
get_gaia_for_legacy_image.py --image legacy/legacysurvey-0470m607-image-r.fits.fz --image-hdu 1 --maglim 21.0 --padding-arcmin 2.0 --out gaia_dr3_0470m607_g21.fits
auto_mask_from_tractor.py --image legacy/legacysurvey-0470m607-image-r.fits.fz --image-hdu 1 --tractor legacy/tractor-0470m607.fits --tractor-hdu 1 --psfsize legacy/legacysurvey-0470m607-psfsize-r.fits.fz --psfsize-hdu 1 --pixscale-arcsec 0.262 --band r --maglim 27.0 --gaia gaia_dr3_0470m607_g21.fits --gaia-maglim 21.0 --gaia-kfwhm 1.5 --out-maskbits maskbits_auto_gaia.fits --star-kfwhm 1.0 --star-mag-scale linear --star-mag-alpha 0.5 --star-mag-max-factor 5 --gal-kre 2.5
rebin_fits.py maskbits_auto_gaia.fits maskbits_auto_gaia_bin4.fits -B 4 --overwrite

** observed **
mu_from_legacy.py legacy/legacysurvey-0470m607-image-r.fits.fz legacysurvey-0470m607-image-r_bin4_mu.fits --image-kind legacy --mask-fits maskbits_auto_gaia.fits --bkg-mode 2d --box-size 32 --filter-size 3 --bkg-exclude-percentile 50 --smooth gauss --gauss-sigma 1.0 --kernel-mode mean --size-arcsec --bin-factor 4 --overwrite
mu_from_legacy.py legacy/legacysurvey-0470m607-image-g.fits.fz legacysurvey-0470m607-image-g_bin4_mu.fits --image-kind legacy --mask-fits maskbits_auto_gaia.fits --bkg-mode 2d --box-size 32 --filter-size 3 --bkg-exclude-percentile 50 --smooth gauss --gauss-sigma 1.0 --kernel-mode mean --size-arcsec --bin-factor 4 --overwrite

** deredden **
mu_from_legacy.py legacy/legacysurvey-0470m607-image-r.fits.fz legacysurvey-0470m607-image-r_deredden_bin4_mu.fits --image-kind legacy --bkg-mode 2d --box-size 32 --filter-size 3 --bkg-exclude-percentile 50 --mask-fits maskbits_auto_gaia.fits --extinction-mode deredden --tractor-fits legacy/tractor-0470m607.fits --dered-region region/tail_region.reg --dered-band r --bin-factor 4 --overwrite
mu_from_legacy.py legacy/legacysurvey-0470m607-image-g.fits.fz legacysurvey-0470m607-image-g_deredden_bin4_mu.fits --image-kind legacy --bkg-mode 2d --box-size 32 --filter-size 3 --bkg-exclude-percentile 50 --mask-fits maskbits_auto_gaia.fits --extinction-mode deredden --tractor-fits legacy/tractor-0470m607.fits --dered-region region/tail_region.reg --dered-band g --bin-factor 4 --overwrite 

** Residual images for comparison purposes**
make_residual_images.py --image legacy/legacysurvey-0470m607-image-r.fits.fz --image-hdu 1 --invvar legacy/legacysurvey-0470m607-invvar-r.fits.fz --invvar-hdu 1 --bbox 2825,3100,2176,2420 --residual-mode none --bg median --maskbits-mode any --smooth-sigma-pix 4 --smooth-mode masked --smooth-snr-from-flux --output-prefix sat_r
make_residual_images.py --image legacy/legacysurvey-0470m607-image-r.fits.fz --image-hdu 1 --invvar legacy/legacysurvey-0470m607-invvar-r.fits.fz --invvar-hdu 1 --maskbits maskbits_auto_gaia.fits --maskbits-hdu 0 --bbox 3100,3520,900,1940 --residual-mode none --bg median --maskbits-mode any --smooth-sigma-pix 4 --smooth-mode masked --smooth-snr-from-flux --output-prefix tail_r

** photometry **
(Note: tractor-max-sma-light-frac=0.85 for LEDA 0.7 for 2MASS)
legacy_ellipse_profile.py measure legacysurvey-0470m607-image-r_bin4_mu.fits galaxies/LEDA175897_Legacy_param.json  LEDA175897_r_profile.csv --tractor-fits legacy/tractor-0470m607.fits --band r --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.85 --no-recenter --plot-profile LEDA175897_r_profile.png --diagnostic-png LEDA175897_r_apertures.png

legacy_ellipse_profile.py measure legacysurvey-0470m607-image-g_bin4_mu.fits galaxies/LEDA175897_Legacy_param.json  LEDA175897_g_profile.csv --tractor-fits legacy/tractor-0470m607.fits --band g --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.85 --magnitudes-out LEDA175897_r_profile.magnitudes.json --no-recenter --plot-profile LEDA175897_g_profile.png --diagnostic-png LEDA175897_g_apertures.png

legacy_ellipse_profile.py measure legacysurvey-0470m607-image-r_bin4_mu.fits galaxies/MASSJ03072631-6043043_Legacy_param.json MASSJ03072631-6043043_r_profile.csv --tractor-fits legacy/tractor-0470m607.fits --band r --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.70 --no-recenter --plot-profile MASSJ03072631-6043043_r_profile.png --diagnostic-png MASSJ03072631-6043043_r_apertures.png

legacy_ellipse_profile.py measure legacysurvey-0470m607-image-g_bin4_mu.fits galaxies/MASSJ03072631-6043043_Legacy_param.json MASSJ03072631-6043043_g_profile.csv --tractor-fits legacy/tractor-0470m607.fits --band g --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.70 --magnitudes-out MASSJ03072631-6043043_r_profile.magnitudes.json --no-recenter --plot-profile MASSJ03072631-6043043_g_profile.png --diagnostic-png MASSJ03072631-6043043_g_apertures.png

** tail **
27.69 +- 0.05 mag/asec^2
tail_surface_brightness.py legacysurvey-0470m607-image-r_deredden_bin4_mu.fits region/tail_region_bin4_pixel.reg maskbits_auto_gaia_bin4.fits --summary-csv tail_sb_comparison.csv --flux-hdu SCIENCE_NMGY

5.63' +- 0.2, -1.44 rad
trace_sat_tail_spline.py legacysurvey-0470m607-image-r_bin4_mu.fits --hdu MU_OBS --seed-points-csv ~/LEDA175897_legacy/tail_points/seed_points_old.csv --mu-max 28.5   --smooth-sigma 2.25   --step 20   --direction-relax 1.8   --intensity-power 0.5   --cap-percentile 98.5   --sigma-parallel-factor 0.3   --sigma-perp-factor 0.3   --out-prefix real_tail --spline-s 50 --star-mask-fits ~/LEDA175897_legacy/maskbits_auto_gaia_bin4.fits --beam-width 6 --lookahead 3 --n-long-bins 6 --min-bin-frac 0.5 --mean-weight 0.10 --std-penalty 0.35

mass_from_photometry.py galaxies/LEDA175897_Legacy_param.json LEDA175897_r_profile.magnitudes.json --ml bell03_diet --planck18 --json --shmr_model umachine --gal_type host > LEDA175897_Legacy_model.json
mass_from_photometry.py galaxies/MASSJ03072631-6043043_Legacy_param.json MASSJ03072631-6043043_r_profile.magnitudes.json --ml zcr09 --planck18 --json --shmr_model umachine --gal_type sat > MASSJ03072631-6043043_Legacy_model.json

GALAXY INITIAL CONDITIONS
----------------------------------------------------------
** AGAMA Host **
build_leda175897_agama_host.py --config agama_models/host_galic_profile_best_config.json
** GaLIC Sat **
run_galic.bash galic_models/MASSJ03072631-6043043_20260216_legacy_dered_med_dt0pt32_S0_0pt23_disk_lambda075_ml-0pt0_lum+0pt64_halo-2pt275_velstruct3_final-hdf5.param
** Combine **
combine_galaxies_nbody_hdf5.py collision_cfg/LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809.cfg --galic-path galic_models --gadget-path .

SYNTHETIC IDEAL IMAGE
----------------------------------------------------------
run_gadget.bash LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809_rp10pt0_e0pt939_20260809_072517-gadget4.param
cd LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809_rp10pt0_e0pt939_20260809_072517

gadget_movie.py galactic_collision_snapshot_{:03d}.hdf5 -b 231 231 --host-params ../LEDA175897_Legacy_model.json --sat-params ../MASSJ03072631-6043043_Legacy_model.json --brick-file ../legacy/legacysurvey-0470m607-image-r.fits.fz --bulge --disk --galaxyid-map galic_models/LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809_rp10pt0_e0pt939_20260927_150327-galic.hdf5.galaxyid.json --camera-load camera_save_231.json --psf 1.259764 --noise 0.0

-- for testing and aligning the model --
***sat***
make_residual_images.py --image synthetic_brick_ideal.fits --image-hdu 0 --invvar ../legacy/legacysurvey-0470m607-invvar-r.fits.fz --invvar-hdu 1 --bbox 2825,3100,2176,2420 --residual-mode none --bg median --maskbits-mode any --smooth-sigma-pix 4 --smooth-mode masked --smooth-snr-from-flux --output-prefix synthetic_sat_r
overlay_fits.py ../sat_r_resid_snr_smooth.fits synthetic_sat_r_resid_snr_smooth.fits --base-stretch asinh --overlay-stretch asinh --overlay-percentiles 50 9

***tail***
make_residual_images.py --image synthetic_brick_ideal.fits --image-hdu 0 --invvar ../legacy/legacysurvey-0470m607-invvar-r.fits.fz --invvar-hdu 1 --maskbits ../maskbits_auto_gaia.fits --maskbits-hdu 0 --bbox 3100,3520,900,1940 --residual-mode none --bg median --maskbits-mode any --smooth-sigma-pix 4 --smooth-mode masked --smooth-snr-from-flux --output-prefix synthetic_tail_r
overlay_fits.py ../tail_r_resid_snr_smooth.fits synthetic_tail_r_resid_snr_smooth.fits --base-stretch asinh --overlay-stretch asinh --overlay-percentiles 50 98

-- generate synthetic image --
***intrinsic for sb compare**
mu_from_legacy.py synthetic_brick_ideal.fits synthetic_brick_ideal_bin4_mu.fits --image-kind synthetic --bkg-mode none --extinction-mode none --smooth gauss --gauss-sigma 1.0  --kernel-mode mean --bin-factor 4 --overwrite 
***reddened for tail compare**
mu_from_legacy.py synthetic_brick_ideal.fits synthetic_brick_ideal_reddened_bin4_mu.fits --image-kind synthetic --bkg-mode none --extinction-mode redden --tractor-fits ~/LEDA175897_reference/tractor-0470m607.fits --smooth gauss --gauss-sigma 1.0  --kernel-mode mean --dered-region ~/LEDA175897_reference/core4_tail_region.reg --bin-factor 4 --overwrite
  
-- test surface brightness to see if is close enough
# 27.69 +- 0.04
tail_surface_brightness.py synthetic_brick_ideal_bin4_mu.fits ../region/tail_region_bin4_pixel.reg ../maskbits_auto_gaia_bin4.fits --flux-hdu SCIENCE_NMGY

-- test tail length to see if close enough
## 5.63 +- 0.2, 1.29 rad
trace_sat_tail_spline.py synthetic_brick_ideal_reddened_bin4_mu.fits --hdu MU_OBS --seed-points-csv ~/LEDA175897_legacy/tail_points/seed_points_old.csv --mu-max 28.5   --smooth-sigma 2.25   --step 20   --direction-relax 1.8   --intensity-power 0.5   --cap-percentile 98.5   --sigma-parallel-factor 0.3   --sigma-perp-factor 0.3   --out-prefix real_tail --spline-s 50 --star-mask-fits ~/LEDA175897_legacy/maskbits_auto_gaia_bin4.fits --beam-width 6 --lookahead 3 --n-long-bins 6 --min-bin-frac 0.5 --mean-weight 0.10 --std-penalty 0.35
 
-- compare tails --
compare_splines.py real_tail.spline_points_pix.csv LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809_rp10pt0_e0pt939_20260809_072517/real_tail.spline_points_pix.csv --pixscale-arcsec 1.048 --plot

-- test noise free r-band photometry
legacy_ellipse_profile.py measure synthetic_brick_ideal_bin4_mu.fits ../galaxies/LEDA175897_Legacy_param.json LEDA175897_r_profile.csv --input-extinction-corrected --hdu DATA_NMGY --tractor-fits ../legacy/tractor-0470m607.fits --band r --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.85 --no-recenter --plot-profile LEDA175897_r_profile.png --diagnostic-png LEDA175897_r_apertures.png
compare_ellipse_profiles.py ../LEDA175897_r_profile.csv LEDA175897_r_profile.csv --label1 "Observed" --label2 "Synthetic" --layout paper --output host_profile_comparison.png --show

legacy_ellipse_profile.py measure synthetic_brick_ideal_bin4_mu.fits ../galaxies/MASSJ03072631-6043043_Legacy_param.json MASSJ03072631-6043043_r_profile.csv --input-extinction-corrected --hdu DATA_NMGY --tractor-fits ../legacy/tractor-0470m607.fits --band r --geometry-from-tractor --max-sma-from-tractor --tractor-max-sma-method sersic-fraction --tractor-max-sma-light-frac 0.70 --no-recenter --plot-profile MASSJ03072631-6043043_r_profile.png --diagnostic-png MASSJ03072631-6043043_r_apertures.png
compare_ellipse_profiles.py ../MASSJ03072631-6043043_r_profile.csv MASSJ03072631-6043043_r_profile.csv --label1 "Observed" --label2 "Synthetic" --layout paper --output sat_profile_comparison.png --show

-- halo measurements of host --
measure_host_halo_shape.py galactic_collision_snapshot_231 --dm-types 1 --galaxyid-map ../galic_models/LEDA175897_galic_profile_rp10pt0_e0pt939_S0_0pt23_disk_galic_real_lambda075_ml-0pt0_lum+0pt64_tilt62_i45_best_20260809_rp10pt0_e0pt939_20260927_150327-galic.hdf5.galaxyid.json --host-name LEDA175897 --out leda175897_halo_measurements.json --plot leda175897_halo_measurements.png --shell-edges 20 60 120 180 240 300 --workers 32 --radius-unit kpc --rmax 300

SYNTHETIC NOISE REALIZATIONS
----------------------------------------------------------
*** for tail length error estimate ***
make_synthetic_noise_realizations.py synthetic_brick_ideal.fits synthetic_observed_realizations --driver-script ~/tidal-tail-software/mu_from_legacy.py --n-realizations 100 --frame observed --prefix s_bin4 --invvar-fits ../legacy/legacysurvey-0470m607-invvar-r.fits.fz --tractor-fits ../legacy/tractor-0470m607.fits --dered-region ../region/tail_region.reg --dered-band r --base-seed 10000 --box-size 32 --smooth gauss --gauss-sigma 1.0 --kernel-mode mean --size-arcsec --overwrite --extra-args --bin-factor 4
cd synthetic_observed_realizations
run_all_splines.sh .

** for surface brightness estimate error estimate **
make_synthetic_noise_realizations.py synthetic_brick_ideal.fits synthetic_dereddened_realizations --driver-script ~/tidal-tail-software/mu_from_legacy.py --n-realizations 100 --frame dereddened --prefix s_bin4 --invvar-fits ../legacy/legacysurvey-0470m607-invvar-r.fits.fz --tractor-fits ../legacy/tractor-0470m607.fits --dered-region ../region/tail_region.reg --dered-band r --base-seed 10000 --box-size 32 --overwrite --extra-args --bin-factor 4 
cd synthetic_dereddened_realizations
run_all_sb.sh . ../../region/tail_region_bin4_pixel.reg ../../maskbits_auto_gaia_bin4.fits

CREATE IMAGE LABELS
----------------------------------------------------------
convert sb_corridor.jpg \( -background none -fill black -font Arial-Bold -pointsize 120 label:"(a)" -trim +repage -background white -gravity center -extent 150x135 \) -gravity SouthWest -geometry +0+0 -composite sb_corridor_annotated.jpg
convert sb_corridor_synthetic.jpg \( -background none -fill black -font Arial-Bold -pointsize 120 label:"(b)" -trim +repage -background white -gravity center -extent 150x135 \) -gravity SouthWest -geometry +0+0 -composite sb_corridor_synthetic_annotated.jpg





