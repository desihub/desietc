# Changelog

## Introduction

This the log of changes to the [desietc package](https://github.com/desihub/desietc).

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.21] - 2025-03-24
### Fixed
 - Use correct capitalization of 'SCR_E_wall_coude' for telemetry query
 - Set refit=False for processing SkyCam frames
## Added
 - Save temperature used for sky-level correction to json

## [0.1.20] - 2025-03-21
### Added
 - Implemented fast centroid fit for SkyCam frames. See DESI-8945 and accompanying notebook for details.
 - Read Coude room east wall temperature using get_telemetry() for SkyCam temperature corrections.
### Changed
 - Use fast centroid fits to process SkyCam frames in ETC.
 - Rescale skylevel estimate by 0.931 when centroid fitting is used, to maintain mean efftime. This ratio
 is derived from the historical average ratio of the new / old algorithms. See DESI-8945 for details.

## [0.1.19] - 2024-10-16
### Changed
- Make default SkyCam processing identical to pre 0.1.18, i.e. refit by default and do not apply centroid fitting or temperature corrections. Centroid fitting is not ready to deploy yet since it runs too slowly. Temperature corrections will require new plumbing with ICS.
### Fixed
- Set MW transparency to one during the backup program. This was supposed to be already implemented, but assumed the wrong case for the program name.

## [0.1.18] - 2024-06-12
### Added
- A version string, so that `desietc.__version__` is now defined.
- Utility function allowing for translations of the reference spot profiles
- New version of the fit_spot function implementing the fitting of the spots position offset in addition to the sky flux and background. Correcting for errors on the flux measurement due to shift in the spots position.
- A linear temperature correction have been added to the updated setraw function.
### Changed
- The setraw function, computing the reduced flux from the raw data, was updated to account for the spot position fitting step.

## [0.1.17] - 2023-10-20
### Added
- Utility class for reading per-exposure json files written by the online ETC.
### Fixed
- Protect against rare race condition seen on 20231018 where pm_info came before acquisition_image.

## [0.1.16] - 2023-08-11
### Fixed
- The 5% increase in calculated EFFTIME_ETC of the last tag went in the wrong direction. This tag implements a 5% decrease instead.

## [0.1.15] - 2023-07-31
### Changed
- Globally increase calculated effective times by a factor of 1.05. See [#11](https://github.com/desihub/desietc/issues/11) for details.
### Fixed
- Replace np.bool with bool because of numpy [deprecation introduced in numpy 1.20.0](https://numpy.org/devdocs/release/1.20.0-notes.html#using-the-aliases-of-builtin-types-like-np-int-is-deprecated).

## [0.1.14] - 2021-11-22
### Changed
- Use MW_transp=1 when the fiberassign FAPRGRM is "BACKUP". Fixes [#8](https://github.com/desihub/desietc/issues/8).
- Use 2,20-min average airmass correction factors in speeds. Fixes [#9](https://github.com/desihub/desietc/issues/9).
- Increase max dropped SkyCam fibers from 1 to 3. Fixes [#7](https://github.com/desihub/desietc/issues/7).

## [0.1.13] - 2021-10-07
### Added
- Description of when ETC telemetry is updated.
### Changed
- Reset speeds to None when there is no recent sky/gfa data.
- Change GFA zeropoint from 27.06 to 26.92 to increase transparencies by 14% and better match the offline analysis.
- Add aircorrection=X**1.75 factor to all speed calculations.
- Rescale EFFTIME so that it increases by 6.0% to match the offline LRG_EFFTIME_DARK during 20210514-20211005 (after correcting for the 14% increase in transparency).
### Fixed
- save FWHM instead of FFRAC to ACQFWHM.

## [0.1.12] - 2021-05-28
### Added
- get_exposure_summary returns FITS header with exposure-averaged quantities.
### Fixed
- protect against NaNs in sky level measurement, which can occur during daytime testing.

## [0.1.11] - 2021-05-21
### Added
- record per-frame FFRAC*TRANSP for ELG, BGS profiles.
- calculate dark/bright/backup speeds with 2- and 20-min averaging.
- save speeds to json file and telemetry.
### Removed
- signal and background values from the telemetry.

## [0.1.10] - 2021-05-17
### Added
- night_to_midnight utility.
- grab more fiberassign header keywords.
### Changed
- mjd_to_night returns an int instead of str (changes type in json header).
- downgrade "Hardcoding EXPTIME=60s" from warning to info.
- more compact "accum" section in the output JSON.
### Fixed
- protect against negative SkyCam fiber flux leading to skylevel=NaN.
- use np.nanmean to protect against NaNs in accumulator.

## [0.1.9] - 2021-05-16
### Added
- More robust handling of problems during sky or guide frame processing in OnlineETC.

## [0.1.8] - 2021-05-14
### Added
- Checks for valid SkyCam data.
- Masking of bad SkyCam fibers 1,9 on SKYCAM0 and 0 on SKYCAM1.
- Fine tuning of SkyCam fiber calibrations.
### Changed
- Average quantities over 60s instead of 240s.
- Freeze expected number of remaining splits at max_split_time / 2.

## [0.1.7] - 2021-05-11
### Added
- Add guide star RA,DEC to JSON.
### Changed
- Divide EFFTIME by 1.07 to correct for the mean ETC/SPEC ratio.
- Move nstars from JSON guide_stars to exp_info.

## [0.1.6] - 2021-05-10
Version with new GFA processing algorithms used for 20210510 testing.
EFFTIME normalization should be roughly the same as before, but with
smaller scatter.  No changes the sky level analysis or read noise model.
### Added
- GFA pixel to CS5 transforms.
- Implement platescale interpolator.
- 0.15 pixel blur to minimize impact of isolated pixels with large ivar.
- Piece-wise linear correction for GFA,SKY deadtime.
### Changed
- Improved method to calculate synthetic fiber profile centered on a guide star.
- Calculate nominal ELG,BGS fiberloss fractions with numerical convolution.

## [0.1.5] - 2021-05-06
### Added
- Implement mininum exposure time as OnlineETC ctor param with 240s default.

## [0.1.4] - 2021-04-05
### Added
- Record efftime,realtime,signal,background in json expinfo.
- Hardcode 60s SKYCAM exposure time (until we debug the abort).
- Save acq_fwhm, acq_ffrac in json expinfo.
### Changed
- Less verbose WARNING log messages.
- Extend FWHM range of ELG/BGS FFRAC correction.
- Remember last sky measurment for 15 mins (was 5 mins).
- Change ffrac_ref from 0.435 back to 0.56 and add scale=(0.56/0.435)**2 to get_efftime to compensate.
### Fixed
- PNG output when some GFAs have no guide stars.

## [0.1.3] - 2021-03-22
### Added
- Calculate FFRAC for sbprof=ELG,BGS,FLT.
- Periodic telemetry updates when the shutters are open.
- mjd_plot utility to graph something versus MJD with localtime labels.
### Changed
- Request to stop/split is only sent once per exposure.
- ICS calls to reset() no longer do anything.
- Tune ETC parameters to TSNR2 using 20210319-21 data.
### Removed
 - Remove debug logging of GFA array view/copy.
### Fixed
 - Fix spurious warnings about exptime jitter.
 - Fix some db bugs.
 - Fix efftime readnoise correction.

## [0.1.2] - 2021-03-11
### Added
- Implement call_when_about_to_stop/split.
- Add last_updated ISO string to status.
- Read new PM info after the shutter closes, to use for next split.
- Include readnoise contribution to effective exposure time.
### Changed
- Rename fieldrot to rel_rotrate in status (but still not implemented).
- Log additional info to debug GFA raw data copy vs view.
- Log additional info to debug EXPTIME jitter error.
### Fixed
- Output of png after acquisition image analysis.

## [0.1.1] - 2021-03-08
### Added
- Implement running averages of FFRAC, TRANSP.
- Correct TRANSP reported to ICS to X=1.
- Implement 6-digit rounding of all float32 values in json output (reduces file size a lot).
- Record git version of desietc being used (still needs testing ICS env).
- Add realtime/efftime_tot status variables.
- Document telemetry variables.
### Changed
- OnlineETC API changes to keep in sync with ICS.
- Refactor accumulation algorithm into separate module.
### Removed
- Periodic status update logic (ETCApp is already doing this).
- Duplicate updates after stop_etc.
### Fixed
- Fix logic for 3 or more cosmic splits.

## [0.1.0] - 2021-02-28
This is the first tag, used for on-sky testing during 20200228.
