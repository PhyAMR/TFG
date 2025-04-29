# CANDELS EGS Data Overview

## Stellar Mass

| Column Name      | Measurement Description | Further Considerations |
|-----------------|-------------------------|------------------------|
| ID              | Sequential ID number in the F160W-based SExtractor catalog | Unique identifier for each detected object |
| RAdeg          | Right Ascension (J2000) in the F160W mosaic | Position of the object in the sky |
| DECdeg         | Declination (J2000) in the F160W mosaic | Position of the object in the sky |
| Hmag           | Magnitude in the F160W band | Measures brightness in the near-infrared |
| PhotFlag       | Flag in the photometric catalog | Identifies issues or special cases in photometry |
| Class_star     | SExtractor CLASS STAR parameter | Determines whether the object is a star or a galaxy |
| AGNflag        | Match in the Nandra et al. (2015) catalog (1=yes, 0=no) | Indicates presence of an active galactic nucleus |
| zphot          | Photometric redshift measurement | Estimated redshift based on multi-band photometry |
| zspec          | Spectroscopic redshift from the DEEP3 catalog | More accurate redshift measurement using spectral lines |
| q_zspec        | Quality flag for the spectroscopic redshift (1=Good, 2=Fair, 3=Poor) | Indicates reliability of spectroscopic redshift |
| r_zspec        | Source of the spectroscopic redshift catalog | Identifies origin of spectroscopic data |
| zbest          | Best redshift measurement, prioritizing spectroscopic over photometric | Combines best available redshift estimate |
| zAGN           | Photometric redshift for AGN sources | Uses AGN-specific SED templates |
| M_neb_med      | Stellar mass measurement considering nebular emission | Accounts for ionized gas contribution to mass estimates |
| s_neb_med      | Uncertainty in M_neb_med | Measurement error due to model assumptions |
| M_med          | Stellar mass measurement without nebular emission | Mass estimate excluding ionized gas effects |
| s_med          | Uncertainty in M_med | Measurement error due to model assumptions |
| Additional 22 columns describing different stellar mass measurement methods and their uncertainties | Various models and methodologies used for estimation | Different assumptions about star formation history and nebular emissions |

## Photometric Redshift

| Column Name       | Measurement Description | Further Considerations |
|------------------|-------------------------|------------------------|
| ID               | Object ID from CANDELS photometry catalog | Unique identifier for each detected object |
| RA              | Right Ascension (J2000) | Position of the object in the sky |
| DEC             | Declination (J2000) | Position of the object in the sky |
| z_best          | Best estimate of redshift (prioritizing spec-z, grism-z, then phot-z) | Indicates the object's distance and lookback time |
| z_best_type     | Type of best redshift (s=spec, g=grism, p=phot) | Source of the redshift determination |
| z_spec          | Spectroscopic redshift if available | More accurate redshift measurement using spectral lines |
| z_spec_ref      | Reference catalog for spectroscopic redshift | Indicates the dataset used for spec-z |
| z_grism         | 3D-HST grism redshift if available | Intermediate accuracy redshift from grism spectroscopy |
| Additional 49 columns describing various redshift estimation methods | Methods include Bayesian, f-divergence, and participant-based calculations | Each method has different statistical confidence levels |

## Physical Properties

| Column Name      | Measurement Description | Further Considerations |
|-----------------|-------------------------|------------------------|
| Age            | Logarithm of stellar age in years | Measures the time since star formation |
| tau            | Star formation timescale in Gyr | Characterizes the duration of star formation |
| Av             | Dust extinction in magnitudes | Indicates the amount of light absorbed by dust |
| SFR            | Star formation rate in solar masses per year | Measures how actively stars are forming |
| Gas Metallicity | Metallicity of gas, normalized to solar values | Indicates chemical enrichment of the galaxy |
| Extinction Law | Calzetti (1) or Small Magellanic Cloud (2) | Describes how dust absorbs/scatters light |
| Reduced chi^2  | Goodness-of-fit statistic | Evaluates the reliability of model fitting |
| log(Mass)_low  | Lower stellar mass limit at 99% confidence | Represents lower bound of stellar mass estimate |
| log(Mass)_high | Upper stellar mass limit at 99% confidence | Represents upper bound of stellar mass estimate |
| E(B-V)         | Color excess due to dust extinction | Quantifies dust absorption effect |
| log(Lbol/Lsun) | Logarithm of bolometric luminosity | Measures total energy output of the galaxy |
| Additional 111 columns covering detailed physical parameters from different authors and models | Properties like star formation history, dust extinction, and rest-frame magnitudes | Provides comprehensive galaxy characterization |

This document summarizes the data available in the CANDELS EGS stellar mass, photometric redshift, and physical properties catalogs.

