# ETC FITS Header Keywords

The following quantities are normally written (by ICS) to the FITS header of HDU-1 (extname "SPEC") in `desi-nnnnnnnn.fits.gz` files. They are also written to the online exposures database. All keywords have a maximum length of 8 characters.

- **ETCVERS** [str]: Version string identifying which git commit of the desietc package was used.
- **ETCTEFF** [float]: Effective time of this exposure, in seconds, estimated by the ETC for the source profile specified in ETCPROF.
- **ETCREAL** [float]: Real open-shutter time of this exposure, in seconds, based on timestamps provided by ICS. Should be close to EXPTIME but will not match exactly because precise shutter timing is not available to the ETC.
- **ETCPREV** [float]: Cummulative effective time, in seconds, of any previous exposures of this tile in the current visit.
- **ETCSPLIT** [int]: Split sequence number for this visit of the current tile, starting at 1.
- **ETCPROF** [string]: Source surface brightness profile used for the ETC effective time calculation. Must be one of "PSF", "ELG", "BGS".
- **ETCTRANS** [float]: Average of observed TRANSP over the exposure. Normalized to 1 for nominal conditions. Note that this value is not corrected to zenith extinction.
- **ETCTHRUP** [float]: Average of FFRAC*TRANSP over the exposure with FFRAC calculated for a PSF source profile. Normalized to 1 for nominal conditions.
- **ETCTHRUE** [float]: Average of FFRAC*TRANSP over the exposure with FFRAC calculated for a ELG source profile. Normalized to 1 for nominal conditions.
- **ETCTHRUB** [float]: Average of FFRAC*TRANSP over the exposure with FFRAC calculated for a BGS source profile. Normalized to 1 for nominal conditions.
- **ETCFRACP** [float]: Transparency-weighted average of FFRAC over the exposure calculated for a PSF source profile. Calculated as (ETCTHRUP/ETCTRANS)*0.56198 where the constant is the nominal PSF FFRAC.
- **ETCFRACE** [float]: Transparency-weighted average of FFRAC over the exposure calculated for a ELG source profile. Calculated as (ETCTHRUP/ETCTRANS)*0.41220 where the constant is the nominal PSF FFRAC.
- **ETCFRACB** [float]: Transparency-weighted average of FFRAC over the exposure calculated for a BGS source profile. Calculated as (ETCTHRUP/ETCTRANS)*0.18985 where the constant is the nominal PSF FFRAC.
- **ETCSKY** [float]: Average of the SkyCam relative flux over the exposure. Normalized to 1 for nominal conditions.
- **ACQFWHM** [float]: FWHM of the guide star PSF, in arcseconds, measured in the initial acquisition image. In case of an elliptical PSF, this is roughly the geometric mean of FWHM along the major and minor axes.
