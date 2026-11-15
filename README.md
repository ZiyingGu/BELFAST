# BELFAST

BELFAST is a software package for estimating the ages of early-type stars, including B-, A-, and early-F-type stars, which often exhibit rapid rotation. Rapid rotation changes the stellar evolution and surface temperature distribution, causing the observed magnitudes to depend on the inclination angle. In this package, we adopt PARSEC v2.0 as the theoretical reference model for comparison with observations.

By combining photometric observations, parallax, and projected rotational velocity ($v\sin i$), BELFAST performs joint sampling over six parameters: initial stellar mass ($M_{\rm ini}$), age ($\tau$), metallicity ($\rm [M/H]$), initial rotation rate ($\omega_{\rm ini}$), inclination ($i$), and parallax ($\varpi$). Optionally, an extinction parameter ($A_V$, $E(B-V)$, or $E(G_{\rm BP}-G_{\rm RP})$) and an uncertainty-inflation parameter ($\sigma_{\rm adjust}$) can also be included in the sampling.

## Environment Setup

Create a new conda environment with Python 3.13.5, activate it, and install Jupyter Notebook:

```bash
conda create -n belfast python=3.13.5
conda activate belfast
pip install notebook
```

## PARS Installation

BELFAST uses the **PARS** package (*Paint the Atmospheres of Rotating Stars*) to model the inclination-dependent photometric properties of rotating stars. PARS was developed by [Lipatov & Brandt (2020)](https://iopscience.iop.org/article/10.3847/1538-4357/aba8f5) and is available on [GitHub](https://github.com/mlipatov/paint_atmospheres).

First, clone the PARS repository:

```bash
git clone https://github.com/mlipatov/paint_atmospheres.git
```

PARS was originally developed with older package versions, and some files need to be updated for compatibility. Four modified files are provided in `others/updated_PARS_functions`. Please replace the corresponding files in PARS with the modified versions:

```text
pa/lib/limbdark.py
pa/lib/star.py
pa/usr/08_vega_spectrum_comparison.py
pa/usr/spectra_colormag.py
```

After applying these modifications, install PARS in editable mode from the root directory of the PARS repository:

```bash
pip install -e . --config-settings editable_mode=compat
```

### Reference

Lipatov, M., & Brandt, T. D. (2020). *Synthetic Spectra of Rotating Stars*. The Astrophysical Journal, **901**, 100. [https://doi.org/10.3847/1538-4357/aba8f5](https://doi.org/10.3847/1538-4357/aba8f5)

## BELFAST Installation

To install BELFAST, run the following command in the root directory of this repository:

```bash
pip install -e .
```

## Tutorials and Data Preparation

Please follow the tutorials to download the necessary files and generate the regularized grids before running the sampling procedure.

When generating regularized grids, please pay attention to the required memory usage. The grid size can become large depending on the selected parameter steps. If needed, adjust the grid step sizes to reduce memory consumption.
