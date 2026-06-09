This project is a software package for estimating the ages of early-type stars (B, A, and early-F), which typically exhibit rapid rotation. Fast rotation reshapes the stellar surface temperature distribution and causes the apparent magnitudes to vary with inclination angle. We adopt PARSEC V2.0 as the theoretical reference model for comparison with observations. Combined with parallax and projected rotational velocity (v sin i), we perform joint sampling over six parameters: stellar mass, logAge, metallicity (MH), rotational rate (ωₖ), inclination, and parallax. Optionally, one or three uncertainty-scaling parameters and an extinction parameter can also be included.

To install BELFAST, run 'pip install -e .' in the root directory of the repository.
Please follow the tutorials to first download necessary files and generate regularized grids.
Please also take care of the needed memory and adjust the grid steps to adjust the sizes of regularized grids. 
