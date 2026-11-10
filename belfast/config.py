row_limit = 5
# Maximum number of rows to retrieve when querying catalogs.
# Typically, only one row matches the query criteria.
# Avoid setting row_limit = 1 due to a known issue in astroquery.


# Column names for stellar identifiers that should be read as strings.
id_columns = [
    "HIP_ID",
    "HD_ID",
    "Gaia_DR3_ID",
    "TYC_ID",
    "TwoMass_ID",
    "Other_Name",
]


# Constants used in BELFAST
GM_SUN   = 1.32712440018e20         # unit='m3 / s2'
R_SUN    = 6.957e8                  # unit='m'
PC_TO_CM = 3.0856775814913674e+18   # converting the unit from 'pc' to 'cm'


# A_lambda/A_V (20261109)
# Values are from Table 3 of Wang & Chen (2019; WC19), except for the B_T and V_T bands.
# ZG derived the B_T and V_T relative extinctions using Eq. (9) of WC19.
# Uncertainties are set to 2.5% of the corresponding values, except for H and Ks, for which the larger original uncertainties from WC19 are retained.
A_lambda_over_A_V_dic_default = {
    'G':    0.789,  'G_err':    0.789 * 2.5 / 100,
    'G_BP': 1.002,  'G_BP_err': 1.002 * 2.5 / 100,
    'G_RP': 0.589,  'G_RP_err': 0.589 * 2.5 / 100,
    'B_J':  1.317,  'B_J_err':  1.317 * 2.5 / 100,
    'V_J':  1,      'V_J_err':  0,
    'B_T':  1.376,  'B_T_err':  1.376 * 2.5 / 100,
    'V_T':  1.050,  'V_T_err':  1.050 * 2.5 / 100,
    'J':    0.243,  'J_err':    0.243 * 2.5 / 100,
    'H':    0.131,  'H_err':    0.006,
    'Ks':   0.078,  'Ks_err':   0.004,
}


# A_lambda/E(G_BP-G_RP) (20261109)
# Values are from Table 3 of WC19, except for the B_T and V_T bands.
# ZG derived the B_T and V_T relative extinctions using A_lambda/E(G_BP-G_RP) = (A_lambda/A_V) × (A_V/E(G_BP-G_RP)).
# Uncertainties are propagated from A_lambda/A_V and A_V/E(G_BP-G_RP).
# A_lambda/A_V is taken from the table above, while A_V/E(G_BP-G_RP) = 2.394 ± 0.018 (Table 3 of WC19).
A_lambda_over_EBP_RP_dic_default = {
    'G':    1.889,  'G_err':    0.049,
    'G_BP': 2.399,  'G_BP_err': 0.063,
    'G_RP': 1.41,   'G_RP_err': 0.037,
    'B_J':  3.153,  'B_J_err':  0.082,
    'V_J':  2.394,  'V_J_err':  0.018,
    'B_T':  3.294,  'B_T_err':  0.086,
    'V_T':  2.514,  'V_T_err':  0.066,
    'J':    0.582,  'J_err':    0.015,
    'H':    0.314,  'H_err':    0.015,
    'Ks':   0.187,  'Ks_err':   0.010,
}


# -------------------------------------------------------------------
# Parameters from Table 4 of Zorec & Royer 2012
# slow_frac and fast_frac are true mixture weights
# slow_mu is the slow-rotator Maxwellian mode mu_s (km/s)
# fast_sqrt2alpha is the table value sqrt(2)*alpha for fast rotators
# -------------------------------------------------------------------
_VROT_PRIOR_PARAMS = {
    "1.60-2.00 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=114.0, fast_ell=46.0),
    "1.70-2.10 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=124.0, fast_ell=46.0),
    "1.80-2.20 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=128.0, fast_ell=48.0),
    "1.90-2.30 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=133.0, fast_ell=44.0),
    "2.00-2.40 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=140.0, fast_ell=45.0),
    "2.10-2.50 Msun": dict(slow_frac=0.03, slow_mu=25.0, fast_frac=0.97, fast_sqrt2alpha=154.0, fast_ell=27.0),
    "2.20-2.60 Msun": dict(slow_frac=0.08, slow_mu=37.0, fast_frac=0.92, fast_sqrt2alpha=164.0, fast_ell=18.0),
    "2.30-2.70 Msun": dict(slow_frac=0.12, slow_mu=44.0, fast_frac=0.88, fast_sqrt2alpha=179.0, fast_ell=12.0),
    "2.35-2.85 Msun": dict(slow_frac=0.12, slow_mu=47.0, fast_frac=0.88, fast_sqrt2alpha=210.0, fast_ell=0.0),
    "2.45-2.95 Msun": dict(slow_frac=0.18, slow_mu=52.0, fast_frac=0.82, fast_sqrt2alpha=185.0, fast_ell=14.0),
    "2.55-3.05 Msun": dict(slow_frac=0.20, slow_mu=61.0, fast_frac=0.80, fast_sqrt2alpha=191.0, fast_ell=20.0),
    "2.65-3.15 Msun": dict(slow_frac=0.15, slow_mu=55.0, fast_frac=0.85, fast_sqrt2alpha=183.0, fast_ell=24.0),
    "2.65-3.35 Msun": dict(slow_frac=0.14, slow_mu=52.0, fast_frac=0.86, fast_sqrt2alpha=187.0, fast_ell=29.0),
    "2.75-3.45 Msun": dict(slow_frac=0.13, slow_mu=47.0, fast_frac=0.87, fast_sqrt2alpha=201.0, fast_ell=18.0),
    "2.85-3.55 Msun": dict(slow_frac=0.10, slow_mu=39.0, fast_frac=0.90, fast_sqrt2alpha=202.0, fast_ell=20.0),
    "2.95-3.65 Msun": dict(slow_frac=0.06, slow_mu=32.0, fast_frac=0.94, fast_sqrt2alpha=224.0, fast_ell=0.0),
    "3.05-3.75 Msun": dict(slow_frac=0.05, slow_mu=31.0, fast_frac=0.95, fast_sqrt2alpha=229.0, fast_ell=0.0),
    "3.15-3.85 Msun": dict(slow_frac=0.08, slow_mu=32.0, fast_frac=0.92, fast_sqrt2alpha=224.0, fast_ell=15.0),
}
