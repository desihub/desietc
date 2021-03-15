# Changelog

This the log of changes to the [desietc package](https://github.com/desihub/desietc).

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - Unreleased
### Added
 - Calculate FFRAC for sbprof=ELG,BGS,FLT.
 - Periodic telemetry updates when the shutters are open.
### Removed
 - Remove debug logging of GFA array view/copy.
### Fixed
 - Fix spurious warnings about exptime jitter.

## [0.1.2] - 2020-03-11
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

## [0.1.1] - 2020-03-08
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

## [0.1.0] - 2020-02-28
This is the first tag, used for on-sky testing during 20200228.
