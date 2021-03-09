# Changelog

This the log of changes to the [desietc package](https://github.com/desihub/desietc).

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2020-03-08
### Added
 - Implement running averages of FFRAC, TRANSP.
 - Correct TRANSP reported to ICS to X=1.
 - Implement 6-digit rounding of all float32 values in json output (reduces file size a lot).
 - Record git version of desietc being used (still needs testing ICS env).
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
