# Changelog

This the log of changes to the [desietc package](https://github.com/desihub/desietc).

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - Unreleased
Version used for the official survey start on 20210514.
## Added
- Checks for valid SkyCam data.
- Masking of bad SkyCam fibers.
- Fine tuning of SkyCam fiber calibrations.

## [0.1.7] - 2021-05-11
## Added
- Add guide star RA,DEC to JSON.
## Changed
- Divide EFFTIME by 1.07 to correct for the mean ETC/SPEC ratio.
- Move nstars from JSON guide_stars to exp_info.

## [0.1.6] - 2021-05-10
Version with new GFA processing algorithms used for 20210510 testing.
EFFTIME normalization should be roughly the same as before, but with
smaller scatter.  No changes the sky level analysis or read noise model.
## Added
 - GFA pixel to CS5 transforms.
 - Implement platescale interpolator.
 - 0.15 pixel blur to minimize impact of isolated pixels with large ivar.
 - Piece-wise linear correction for GFA,SKY deadtime.
## Changed
 - Improved method to calculate synthetic fiber profile centered on a guide star.
 - Calculate nominal ELG,BGS fiberloss fractions with numerical convolution.

## [0.1.5] - 2021-05-06
## Added
 - Implement mininum exposure time as OnlineETC ctor param with 240s default.

## [0.1.4] - 2021-04-05
## Added
 - Record efftime,realtime,signal,background in json expinfo.
 - Hardcode 60s SKYCAM exposure time (until we debug the abort).
 - Save acq_fwhm, acq_ffrac in json expinfo.
## Changed
 - Less verbose WARNING log messages.
 - Extend FWHM range of ELG/BGS FFRAC correction.
 - Remember last sky measurment for 15 mins (was 5 mins).
 - Change ffrac_ref from 0.435 back to 0.56 and add scale=(0.56/0.435)**2 to get_efftime to compensate.
## Fixed
 - PNG output when some GFAs have no guide stars.

## [0.1.3] - 2021-03-22
### Added
 - Calculate FFRAC for sbprof=ELG,BGS,FLT.
 - Periodic telemetry updates when the shutters are open.
 - mjd_plot utility to graph something versus MJD with localtime labels.
## Changed
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
