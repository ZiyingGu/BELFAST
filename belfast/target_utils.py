import numpy as np


def cal_parsec(parallax):
    """
    Convert parallax (in mas) to distance in parsecs.

    Args:
        parallax (float or array-like): Parallax in milliarcseconds.

    Returns:
        float or array-like: Distance in parsecs.
    """
    distance_pc = 1000 / parallax  # distance in pc
    return distance_pc


# # parsec to parallax
# def cal_parallax(distance_pc):
#     # d is parallax in pc
#     parallax = 1000/distance_pc # distance in mas
#     return parallax


# For collecting data from isochrones
def cal_abs_mag(obs_mags, obs_err_mags, parallax_obs, parallax_obs_err, max_sigma_adjustment_parameter=None):
    """
    Calculate absolute magnitudes and uncertainties from observed magnitudes and parallax.

    Args:
        obs_mags (array-like): Observed apparent magnitudes.
        obs_err_mags (array-like): Uncertainties in observed magnitudes.
        parallax_obs (float): Observed parallax in mas.
        parallax_obs_err (float): Uncertainty in parallax.

    Returns:
        tuple: (abs_mags, abs_err_mags), both flattened arrays.
    """

    if max_sigma_adjustment_parameter is not None:
        if isinstance(max_sigma_adjustment_parameter, (float, int)):
            obs_err_mags = np.sqrt(obs_err_mags**2 + max_sigma_adjustment_parameter**2)
        else:
            raise ValueError('max_sigma_adjustment_parameter must be of type float or int.')
    abs_mags = obs_mags - 5 * np.log10(100 / parallax_obs)
    abs_err_mags = np.sqrt(obs_err_mags ** 2 + (5 * parallax_obs_err / np.log(10) / parallax_obs) ** 2)
    return abs_mags.flatten(), abs_err_mags.flatten()


# For calculating likelihood
def cal_obs_mag(abs_mags, parallax):
    """
    Convert absolute magnitudes to observed apparent magnitudes given parallax.

    Args:
        abs_mags (array-like): Absolute magnitudes.
        parallax (float): Parallax in milliarcseconds.

    Returns:
        array-like: Observed magnitudes.
    """
    obs_mags = abs_mags + 5 * np.log10(100 / parallax)
    return obs_mags


# Flux to observed magnitude
def flux_to_obs_mag(F_star, F_star_err, bands, F_vega_dic=None):
    """
    Convert fluxes (Jy) into observed magnitudes using Vega system.

    Args:
        F_star (array-like): Stellar fluxes in Jy.
        F_star_err (array-like): Flux uncertainties.
        bands (list of str): List of photometric bands.
        F_vega_dic (dict, optional): Dictionary of Vega fluxes in Jy. Defaults provided if None.

    Returns:
        tuple: (obs_mag, obs_err_mag) arrays.
    """
    if F_vega_dic is None:
        # From SVO service (all in Vega System)
        F_vega_dic = {
            # Gaia (EDR3)
            'G': 3228.75,
            'G_BP': 3552.01,
            'G_RP': 2554.95,
            # Tycho-2 revised by Mann & von Braun 2014
            'B_T': 3887.24,
            'V_T': 3692.28,
            # 2MASS
            'J': 1574.48,
            'H': 1011.27,
            'Ks': 654.78
        }

    print('F_star should be in list or array.')
    print('bands should be in list.')
    print('Flux should be in the unit of Jy.')

    results = []
    for band in bands:
        result = band in F_vega_dic.keys()
        results.append(result)

    if not np.all(results):
        print('Available bands: ', list(F_vega_dic.keys()))
        raise ValueError("Invalid bands!")

    if isinstance(F_star, list):
        F_star = np.array(F_star)

    F_vega = []
    for band in bands:
        F_vega.append(F_vega_dic[band])
    F_vega = np.array(F_vega)

    obs_mag = -2.5 * np.log10(F_star / F_vega)
    obs_err_mag = F_star_err / (0.4 * np.log(10) * F_star)

    return obs_mag, obs_err_mag


# Observed magnitude to flux
def obs_mag_to_flux(obs_mag, obs_err_mag, bands, F_vega_dic=None):
    """
    Convert observed magnitudes into fluxes (Jy) using Vega system.

    Args:
        obs_mag (array-like): Observed magnitudes.
        obs_err_mag (array-like): Uncertainties in magnitudes.
        bands (list of str): List of photometric bands.
        F_vega_dic (dict, optional): Dictionary of Vega fluxes in Jy. Defaults provided if None.

    Returns:
        tuple: (F_star, F_star_err) arrays in Jy.
    """
    if F_vega_dic is None:
        # From SVO service (all in Vega System)
        F_vega_dic = {
            # Gaia (EDR3)
            'G': 3228.75,
            'G_BP': 3552.01,
            'G_RP': 2554.95,
            # Tycho-2 revised by Mann & von Braun 2014
            'B_T': 3887.24,
            'V_T': 3692.28,
            # 2MASS
            'J': 1574.48,
            'H': 1011.27,
            'Ks': 654.78
        }

    print('obs_mag should be in list or array.')
    print('bands should be in list.')
    print('Output flux will be in the unit of Jy.')

    results = []
    for band in bands:
        result = band in F_vega_dic.keys()
        results.append(result)

    if not np.all(results):
        print('Available bands: ', list(F_vega_dic.keys()))
        raise ValueError("Invalid bands!")

    if isinstance(obs_mag, list):
        obs_mag = np.array(obs_mag)

    F_vega = []
    for band in bands:
        F_vega.append(F_vega_dic[band])
    F_vega = np.array(F_vega)

    F_star = F_vega * 10 ** (-0.4 * obs_mag)
    F_star_err = np.log(10) * 0.4 * F_star * obs_err_mag

    return F_star, F_star_err




def remove_extinction(obs_mags_no_cor, obs_err_mags_no_cor, bands_list, extinction, extinction_err, extinction_type,
                      R_V=3.16, R_V_err=0.15, print_flag=True, A_lambda_over_EBP_RP_dic=None, A_lambda_over_A_V_dic=None):
    """
    Correct observed magnitudes for interstellar extinction.

    Args:
        obs_mags_no_cor (array-like): Observed magnitudes without extinction correction.
        obs_err_mags_no_cor (array-like): Uncertainties in observed magnitudes.
        bands_list (list of str): List of photometric bands.
        extinction (float): Extinction value.
        extinction_err (float): Uncertainty of extinction.
        extinction_type (str): Extinction type ('A_V', 'EB_V', or 'EBP_RP').
        R_V (float): Total-to-selective extinction ratio (default 3.16).
        R_V_err (float): Uncertainty in R_V.
        print_flag (bool): Whether to print per-band extinction corrections.

    Returns:
        tuple: (corrected magnitudes, corrected uncertainties).
    """

    # (EB_V: B. J. Taylor 2006) and (R_V: Wang & Chen 2019)
    # Here A_V is the extinction of the Johnson V band

    obs_mags_no_cor = np.asarray(obs_mags_no_cor, dtype=float)
    obs_err_mags_no_cor = np.asarray(obs_err_mags_no_cor, dtype=float)

    if A_lambda_over_EBP_RP_dic is None:
    # A_lambda/E(G_BP-G_RP) from Wang & Chen 2019
        A_lambda_over_EBP_RP_dic = {
            'G': 1.890, 'G_err': 0.015,
            'G_BP': 2.429, 'G_BP_err': 0.015,
            'G_RP': 1.429, 'G_RP_err': 0.015,
            'B_J': 3.151, 'B_J_err': 0.027,
            'V_J': 2.394, 'V_J_err': 0.018,
            'B_T': 3.265, 'B_T_err': 0.045,  # (Gu calculated on 20250816)
            'V_T': 2.526, 'V_T_err': 0.019,  # (Gu calculated on 20250816)
            'J': 0.582, 'J_err': 0.011,
            'H': 0.313, 'H_err': 0.014,
            'Ks': 0.186, 'Ks_err': 0.009
        }

    # old; using F99 with Vega and plus offsets
    #
    # if A_lambda_over_A_V_dic is None:
    #     # A_lambda/A_v from Wang & Chen 2019
    #     A_lambda_over_A_V_dic = {
    #         'G': 0.789, 'G_err': 0.005,
    #         'G_BP': 1.002, 'G_BP_err': 0.007,
    #         'G_RP': 0.589, 'G_RP_err': 0.004,
    #         'B_J': 1.317, 'B_J_err': 0.016,
    #         'V_J': 1, 'V_J_err': 0,
    #         'B_T': 1.364, 'B_T_err': 0.016,  # (Gu calculated on 20250816)
    #         'V_T': 1.055, 'V_T_err': 0,  # (Gu calculated on 20250816)
    #         'J': 0.243, 'J_err': 0.004,
    #         'H': 0.131, 'H_err': 0.006,
    #         'Ks': 0.078, 'Ks_err': 0.004
    #     }

    # new; CCM with WC19's new measurements
    #
    if A_lambda_over_A_V_dic is None:
        # A_lambda/A_v from Wang & Chen 2019
        A_lambda_over_A_V_dic = {
            'G': 0.837, 'G_err': 0.021,
            'G_BP': 1.101, 'G_BP_err': 0.028,
            'G_RP': 0.585, 'G_RP_err': 0.015,
            'B_J': 1.313, 'B_J_err': 0.033,
            'V_J': 0.998, 'V_J_err': 0.025,
            'B_T': 1.376, 'B_T_err': 0.034,
            'V_T': 1.050, 'V_T_err': 0.026,
            'J': 0.24, 'J_err': 0.002,
            'H': 0.133, 'H_err': 0.002,
            'Ks': 0.076, 'Ks_err': 0.002,
        }

    # E(B-V) = 2.394/3.16 * E(BP-RP) = 0.758 * E(BP-RP)
    # E(BP-RP) = 3.16/2.394 E(B-V) = 1.320 * E(B-V)
    # A_V = 3.16 * E(B-V)
    # A_V = 2.394 * E(B-V)


    if isinstance(obs_mags_no_cor, list):
        obs_mags_no_cor = np.array(obs_mags_no_cor)
    if isinstance(obs_err_mags_no_cor, list):
        obs_err_mags_no_cor = np.array(obs_err_mags_no_cor)

    correction = []
    correction_err = []

    if extinction_type in ['A_V', 'EB_V']:
        if extinction_type == 'A_V':
            A_V = extinction
            A_V_err = extinction_err
        elif extinction_type == 'EB_V':
            A_V = R_V * extinction
            A_V_err = np.sqrt((extinction * R_V_err) ** 2 + (extinction_err * R_V) ** 2)
        for band_lambda in bands_list:
            if band_lambda not in A_lambda_over_A_V_dic or band_lambda + '_err' not in A_lambda_over_A_V_dic:
                raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
            A_lambda_over_A_V = A_lambda_over_A_V_dic[band_lambda]
            A_lambda_over_A_V_err = A_lambda_over_A_V_dic[band_lambda + '_err']
            correction_lambda = A_lambda_over_A_V * A_V
            correction_err_lambda = np.sqrt((A_lambda_over_A_V * A_V_err) ** 2 + (A_V * A_lambda_over_A_V_err) ** 2)
            if print_flag == True:
                print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
            correction.append(correction_lambda)
            correction_err.append(correction_err_lambda)

    elif extinction_type == 'EBP_RP':
        EBP_RP = extinction
        EBP_RP_err = extinction_err
        for band_lambda in bands_list:
            if band_lambda not in A_lambda_over_EBP_RP_dic or band_lambda + '_err' not in A_lambda_over_EBP_RP_dic:
                raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
            A_lambda_over_EBP_RP = A_lambda_over_EBP_RP_dic[band_lambda]
            A_lambda_over_EBP_RP_err = A_lambda_over_EBP_RP_dic[band_lambda + '_err']
            correction_lambda = A_lambda_over_EBP_RP * EBP_RP
            correction_err_lambda = np.sqrt((A_lambda_over_EBP_RP * EBP_RP_err) ** 2 + (EBP_RP * A_lambda_over_EBP_RP_err) ** 2)
            if print_flag == True:
                print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
            correction.append(correction_lambda)
            correction_err.append(correction_err_lambda)

    if len(correction) == 0 and len(correction_err) == 0:
        obs_mags_removed = obs_mags_no_cor
        obs_err_mags_removed = obs_err_mags_no_cor
    else:
        obs_mags_removed = obs_mags_no_cor - np.array(correction)
        obs_err_mags_removed = np.sqrt(obs_err_mags_no_cor ** 2 + np.array(correction_err) ** 2)

    return obs_mags_removed, obs_err_mags_removed



def add_extinction(obs_mags_model, obs_err_mags_model, bands_list, extinction, extinction_err, extinction_type,
                   R_V=3.16, R_V_err=0.15, print_flag=True, A_lambda_over_EBP_RP_dic=None, A_lambda_over_A_V_dic=None):
    """
    Apply extinction to model magnitudes.

    Args:
        obs_mags_model (array-like): Model magnitudes without extinction.
        obs_err_mags_model (array-like): Uncertainties in model magnitudes.
        bands_list (list of str): List of photometric bands.
        extinction (float): Extinction value.
        extinction_err (float): Uncertainty of extinction.
        extinction_type (str): Extinction type ('A_V', 'EB_V', or 'EBP_RP').
        R_V (float): Total-to-selective extinction ratio (default 3.16).
        R_V_err (float): Uncertainty in R_V.
        print_flag (bool): Whether to print per-band extinction corrections.

    Returns:
        tuple: (extincted magnitudes, extincted uncertainties).
    """

    # (EB_V: B. J. Taylor 2006) and (R_V: Wang & Chen 2019)
    # Here A_V is the extinction of the Johnson V band

    if A_lambda_over_EBP_RP_dic is None:
    # A_lambda/E(G_BP-G_RP) from Wang & Chen 2019
        A_lambda_over_EBP_RP_dic = {
            'G': 1.890, 'G_err': 0.015,
            'G_BP': 2.429, 'G_BP_err': 0.015,
            'G_RP': 1.429, 'G_RP_err': 0.015,
            'B_J': 3.151, 'B_J_err': 0.027,
            'V_J': 2.394, 'V_J_err': 0.018,
            'B_T': 3.265, 'B_T_err': 0.045,  # (Gu calculated on 20250816)
            'V_T': 2.526, 'V_T_err': 0.019,  # (Gu calculated on 20250816)
            'J': 0.582, 'J_err': 0.011,
            'H': 0.313, 'H_err': 0.014,
            'Ks': 0.186, 'Ks_err': 0.009
        }

    if A_lambda_over_A_V_dic is None:
        # A_lambda/A_v from Wang & Chen 2019
        A_lambda_over_A_V_dic = {
            'G': 0.789, 'G_err': 0.005,
            'G_BP': 1.002, 'G_BP_err': 0.007,
            'G_RP': 0.589, 'G_RP_err': 0.004,
            'B_J': 1.317, 'B_J_err': 0.016,
            'V_J': 1, 'V_J_err': 0,
            'B_T': 1.364, 'B_T_err': 0.016,  # (Gu calculated on 20250816)
            'V_T': 1.055, 'V_T_err': 0,  # (Gu calculated on 20250816)
            'J': 0.243, 'J_err': 0.004,
            'H': 0.131, 'H_err': 0.006,
            'Ks': 0.078, 'Ks_err': 0.004
        }

    # E(B-V) = 2.396/3.16 * E(BP-RP) = 0.758 * E(BP-RP)
    # E(BP-RP) = 3.16/2.396 E(B-V) = 1.319 * E(B-V)

    if isinstance(obs_mags_model, list):
        obs_mags_model = np.array(obs_mags_model)
    if isinstance(obs_err_mags_model, list):
        obs_err_mags_model = np.array(obs_err_mags_model)

    correction = []
    correction_err = []

    if extinction_type in ['A_V', 'EB_V']:
        if extinction_type == 'A_V':
            A_V = extinction
            A_V_err = extinction_err
        elif extinction_type == 'EB_V':
            A_V = R_V * extinction
            A_V_err = np.sqrt((extinction * R_V_err) ** 2 + (extinction_err * R_V) ** 2)
        for band_lambda in bands_list:
            if band_lambda not in A_lambda_over_A_V_dic or band_lambda + '_err' not in A_lambda_over_A_V_dic:
                raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
            A_lambda_over_A_V = A_lambda_over_A_V_dic[band_lambda]
            A_lambda_over_A_V_err = A_lambda_over_A_V_dic[band_lambda + '_err']
            correction_lambda = A_lambda_over_A_V * A_V
            correction_err_lambda = np.sqrt((A_lambda_over_A_V * A_V_err) ** 2 + (A_V * A_lambda_over_A_V_err) ** 2)
            if print_flag == True:
                print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
            correction.append(correction_lambda)
            correction_err.append(correction_err_lambda)

    elif extinction_type == 'EBP_RP':
        EBP_RP = extinction
        EBP_RP_err = extinction_err
        for band_lambda in bands_list:
            if band_lambda not in A_lambda_over_EBP_RP_dic or band_lambda + '_err' not in A_lambda_over_EBP_RP_dic:
                raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
            A_lambda_over_EBP_RP = A_lambda_over_EBP_RP_dic[band_lambda]
            A_lambda_over_EBP_RP_err = A_lambda_over_EBP_RP_dic[band_lambda + '_err']
            correction_lambda = A_lambda_over_EBP_RP * EBP_RP
            correction_err_lambda = np.sqrt((A_lambda_over_EBP_RP * EBP_RP_err) ** 2 + (EBP_RP * A_lambda_over_EBP_RP_err) ** 2)
            if print_flag == True:
                print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
            correction.append(correction_lambda)
            correction_err.append(correction_err_lambda)

    obs_mags_model_extincted = obs_mags_model + np.array(correction)
    mags_ext_err_model_extincted = np.sqrt(np.array(obs_err_mags_model) ** 2 + np.array(correction_err) ** 2)

    return obs_mags_model_extincted, mags_ext_err_model_extincted




def get_info_from_simbad(name):
    hip_id = np.nan
    hd_id = np.nan
    gaia_dr3_id = np.nan
    tyc_id = np.nan
    twomass_id = np.nan
    other_id = np.nan

    # NEW
    sp_type = np.nan
    otype = np.nan
    ra_deg = np.nan
    dec_deg = np.nan

    from astroquery.simbad import Simbad
    from astropy.coordinates import SkyCoord

    try:
        if not isinstance(name, str) or name.strip() == "":
            return hip_id, hd_id, gaia_dr3_id, tyc_id, twomass_id, other_id, sp_type, otype, ra_deg, dec_deg

        def _norm(s):
            if s is None:
                return None
            if hasattr(s, "decode"):
                try:
                    s = s.decode("utf-8", errors="ignore")
                except Exception:
                    pass
            s = str(s)

            import re
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _out(x):
            if x is None:
                return np.nan
            if isinstance(x, float) and np.isnan(x):
                return np.nan
            s = _norm(x)
            return s if s not in ("", "nan", "None") else np.nan

        # -------- 1) SIMBAD 主名称 + coordinates + sp_type + otype --------
        main_id = None
        try:
            sim = Simbad()
            # 关键：coordinates -> 返回 result['ra'], result['dec']
            sim.add_votable_fields("coordinates", "otype", "sp_type")
            q = sim.query_object(name)

            if q is not None and len(q) > 0:
                row = q[0]
                # MAIN_ID 可能在这个 query 里也有
                if "MAIN_ID" in q.colnames:
                    main_id = _norm(row["MAIN_ID"])
                else:
                    # 兜底再用一次默认 query_object
                    q2 = Simbad.query_object(name)
                    if q2 is not None and len(q2) > 0 and "MAIN_ID" in q2.colnames:
                        main_id = _norm(q2[0]["MAIN_ID"])

                # ra/dec：用 SkyCoord 强制当度解析，再取 deg（最稳）
                if ("ra" in q.colnames) and ("dec" in q.colnames):
                    ra_val = row["ra"]
                    dec_val = row["dec"]
                    ra_str = _norm(ra_val)
                    dec_str = _norm(dec_val)
                    if ra_str is not None and dec_str is not None:
                        import astropy.units as u
                        coord = SkyCoord(f"{ra_str} {dec_str}", unit=(u.deg, u.deg), frame="icrs")
                        ra_deg = float(coord.ra.deg)
                        dec_deg = float(coord.dec.deg)

                # sp_type / otype
                if "sp_type" in q.colnames:
                    sp_type = _out(row["sp_type"])
                if "otype" in q.colnames:
                    otype = _out(row["otype"])

        except Exception:
            main_id = None

        # -------- 2) 所有 cross-identifiers --------
        ids = Simbad.query_objectids(name)
        if ids is None or len(ids) == 0:
            if main_id is not None and not (
                main_id.startswith("HIP ")
                or main_id.startswith("HD ")
                or main_id.startswith("Gaia")
            ):
                other_id = _norm(main_id)

            return (
                _out(hip_id), _out(hd_id), _out(gaia_dr3_id),
                _out(tyc_id), _out(twomass_id), _out(other_id),
                _out(sp_type), _out(otype), ra_deg, dec_deg
            )

        # 识别 SIMBAD 的 cross-ID 列名
        if "ID" in ids.colnames:
            colname = "ID"
        elif "id" in ids.colnames:
            colname = "id"
        else:
            colname = ids.colnames[0]

        other_candidates = []

        # -------- 3) 遍历 cross-IDs --------
        for val in ids[colname]:
            s = _norm(val)
            if not s:
                continue

            if s.startswith("HIP "):
                try:
                    hip_id = str(int(s.split()[1]))
                except Exception:
                    pass
                continue

            if s.startswith("HD "):
                try:
                    hd_id = str(int(s.split()[1]))
                except Exception:
                    pass
                continue

            if s.startswith("Gaia DR3"):
                try:
                    gaia_dr3_id = str(int(s.split()[-1]))
                except Exception:
                    pass
                continue

            if s.startswith("TYC "):
                tyc_id = _norm(s.split("TYC ", 1)[1])
                other_candidates.append(_norm("TYC " + tyc_id))
                continue

            if s.startswith("2MASS "):
                twomass_id = _norm(s.split("2MASS ", 1)[1])
                other_candidates.append(_norm("2MASS " + twomass_id))
                continue

            if s.startswith("Gaia "):
                continue

            other_candidates.append(s)

        # -------- 4) 选择 other_id --------
        if (
            main_id is not None
            and not main_id.startswith("HIP ")
            and not main_id.startswith("HD ")
            and not main_id.startswith("Gaia ")
        ):
            other_id = _norm(main_id)
        elif other_candidates:
            filtered = [
                s for s in other_candidates
                if not s.startswith("TYC ") and not s.startswith("2MASS ")
            ]

            prefix_priority = ["V* ", "HR ", "BD ", "CD ", "CPD ", "GJ "]

            chosen = None
            for prefix in prefix_priority:
                for s in filtered:
                    if s.startswith(prefix):
                        chosen = s
                        break
                if chosen:
                    break

            if chosen is None:
                chosen = filtered[0] if filtered else other_candidates[0]

            other_id = _norm(chosen)

        # -------- 5) 输出 --------
        return (
            _out(hip_id), _out(hd_id), _out(gaia_dr3_id),
            _out(tyc_id), _out(twomass_id), _out(other_id),
            _out(sp_type), _out(otype), ra_deg, dec_deg
        )

    except Exception:
        return hip_id, hd_id, gaia_dr3_id, tyc_id, twomass_id, other_id, sp_type, otype, ra_deg, dec_deg




def query_tycho_2(tyc_id):
    """
    Query Tycho-2 photometry (B_T, V_T) and their uncertainties by TYC ID.
    Catalog order (stop at first match):
      1) I/259/tyc2
      2) I/259/suppl_1
      3) I/259/suppl_2

    Parameters
    ----------
    tyc_id : str
        e.g. "TYC 1395-2605-1" or "1395-2605-1"

    Returns
    -------
    bt, vt, e_bt, e_vt : float
        B_T, V_T, and their 1-sigma uncertainties. If not found, all NaN.
    """
    from astroquery.vizier import Vizier

    def _nan():
        return np.nan

    def _safe_float(x):

        try:
            if x is None:
                return np.nan
            s = str(x).strip()
            if s in ("", "--", "nan", "None"):
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    # ---- parse TYC id ----
    if tyc_id is None:
        return (_nan(),) * 4

    s = str(tyc_id).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return (_nan(),) * 4

    import re
    s = re.sub(r"^\s*TYC\s+", "", s, flags=re.IGNORECASE).strip()
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return (_nan(),) * 4

    tyc1, tyc2, tyc3 = map(int, m.groups())

    # ---- query catalogs in order ----
    catalogs = ["I/259/tyc2", "I/259/suppl_1", "I/259/suppl_2"]

    # Tycho-2 typical columns:
    # BTmag, VTmag, e_BTmag, e_VTmag
    columns = ["TYC1", "TYC2", "TYC3", "BTmag", "VTmag", "e_BTmag", "e_VTmag"]
    v = Vizier(columns=columns, row_limit=5)

    for cat in catalogs:
        try:
            res = v.query_constraints(
                catalog=cat,
                TYC1=tyc1,
                TYC2=tyc2,
                TYC3=tyc3,
            )
            if res is None or len(res) == 0:
                continue

            tab = res[0]
            if len(tab) == 0:
                continue

            row = tab[0]

            bt = _safe_float(row["BTmag"]) if "BTmag" in tab.colnames else _nan()
            vt = _safe_float(row["VTmag"]) if "VTmag" in tab.colnames else _nan()
            e_bt = _safe_float(row["e_BTmag"]) if "e_BTmag" in tab.colnames else _nan()
            e_vt = _safe_float(row["e_VTmag"]) if "e_VTmag" in tab.colnames else _nan()

            # Stop at first catalog that has a matched row (as you requested)
            return bt, vt, e_bt, e_vt

        except Exception:
            # network / Vizier hiccup -> try next catalog
            continue

    return (_nan(),) * 4



def _norm_2mass_id(twomass_id: str):
    import re

    s = str(twomass_id).strip()
    s = re.sub(r"^\s*2MASS\s+", "", s, flags=re.IGNORECASE).strip()
    if s.startswith("J"):
        s_noj = s[1:]
    else:
        s_noj = s
        s = "J" + s  # 规范化带J版本（备用）
    return s, s_noj

def query_2mass(twomass_id):
    from astroquery.vizier import Vizier

    def _nan(): return np.nan
    def _safe_float(x):
        try:
            if x is None: return np.nan
            s = str(x).strip()
            if s in ("", "--", "nan", "None"): return np.nan
            return float(x)
        except Exception:
            return np.nan
    def _safe_str(x):
        if x is None: return np.nan
        s = str(x).strip()
        return s if s not in ("", "--", "nan", "None") else np.nan

    if twomass_id is None:
        return (_nan(),) * 12

    sJ, s_noJ = _norm_2mass_id(twomass_id)

    catalog = "II/246/out"
    columns = ["RAJ2000","DEJ2000","2MASS",
               "Jmag","e_Jmag","Hmag","e_Hmag","Kmag","e_Kmag",
               "Qflg","Rflg","Bflg","Cflg","Xflg","Aflg"]

    v = Vizier(columns=columns, row_limit=10)

    # 只尝试真正的ID列：2MASS
    for keyval in [s_noJ, sJ]:
        try:
            res = v.query_constraints(catalog=catalog, **{"2MASS": keyval})
            if res is not None and len(res) > 0 and len(res[0]) > 0:
                tab = res[0]
                row = tab[0]

                j  = _safe_float(row["Jmag"])
                h  = _safe_float(row["Hmag"])
                ks = _safe_float(row["Kmag"])

                e_j  = _safe_float(row["e_Jmag"])
                e_h  = _safe_float(row["e_Hmag"])
                e_ks = _safe_float(row["e_Kmag"])

                Qflg = _safe_str(row["Qflg"])
                Rflg = _safe_str(row["Rflg"])
                Bflg = _safe_str(row["Bflg"])
                Cflg = _safe_str(row["Cflg"])
                Xflg = _safe_str(row["Xflg"])
                Aflg = _safe_str(row["Aflg"])

                return j, h, ks, e_j, e_h, e_ks, Qflg, Rflg, Bflg, Cflg, Xflg, Aflg
        except Exception:
            continue

    return (_nan(),) * 12


def assess_2mass_phot_quality(
    J_mag, H_mag, Ks_mag,
    J_err_mag, H_err_mag, Ks_err_mag,
    Q_2mass, R_2mass, B_2mass, C_2mass, X_2mass, A_2mass,
    *,
    max_err_warn=0.05,
    max_err_bad=0.10,
    strict=True,
    verbose=True
):
    """
    Assess 2MASS photometric quality and emit warnings if any item looks problematic.

    Parameters
    ----------
    strict : bool
        If True, use stricter criteria:
          - Qflg must be A
          - Rflg must be one of {1,2,3}
          - Bflg must be 1
          - global C/X/A flags must all be zero
        If False, use more permissive criteria:
          - Qflg in {A,B} is acceptable
          - Rflg in {1,2,3} is acceptable
          - Bflg > 1 is warned but not necessarily fatal
    max_err_warn, max_err_bad : float
        Magnitude uncertainty thresholds for warning/bad.

    Returns
    -------
    result : dict
        {
          "ok": bool,
          "warnings": [str, ...],
          "per_band": {
              "J": {"ok": bool, "issues": [...]},
              "H": ...,
              "Ks": ...
          }
        }
    """

    def _isfinite(x):
        return x is not None and np.isfinite(x)

    def _to_str(x):
        if x is None:
            return ""
        if isinstance(x, bytes):
            return x.decode("utf-8", errors="ignore")
        return str(x)

    def _safe_char_at(s, i):
        s = _to_str(s).strip()
        if len(s) <= i:
            return None
        return s[i]

    def _nonzero_flag(s):
        s = _to_str(s).strip()
        if s in ("", "nan", "None"):
            return False
        try:
            return int(float(s)) != 0
        except Exception:
            return s != "0"

    warnings = []
    per_band = {}

    # --- parse flags ---
    qJ = _safe_char_at(Q_2mass, 0); qH = _safe_char_at(Q_2mass, 1); qK = _safe_char_at(Q_2mass, 2)
    rJ = _safe_char_at(R_2mass, 0); rH = _safe_char_at(R_2mass, 1); rK = _safe_char_at(R_2mass, 2)
    bJ = _safe_char_at(B_2mass, 0); bH = _safe_char_at(B_2mass, 1); bK = _safe_char_at(B_2mass, 2)

    C = _to_str(C_2mass).strip()
    X = _to_str(X_2mass).strip()
    A = _to_str(A_2mass).strip()

    contam = _nonzero_flag(C)
    xtalk = _nonzero_flag(X)
    assoc = _nonzero_flag(A)

    # --- global warnings ---
    if contam:
        warnings.append(f"Warning: 2MASS Cflg={C} indicates possible contamination/artifacts.")
    if xtalk:
        warnings.append(f"Warning: 2MASS Xflg={X} indicates possible cross-talk contamination.")
    if assoc:
        warnings.append(f"Warning: 2MASS Aflg={A} indicates non-standard association/complex source.")

    # --- helper to evaluate a single band ---
    def _eval_band(band, mag, emag, q, r, b):
        issues = []
        ok = True

        # Magnitude existence
        if not _isfinite(mag):
            issues.append("missing magnitude (NaN)")
            ok = False

        # Magnitude uncertainty
        if not _isfinite(emag):
            issues.append("missing uncertainty (NaN)")
            ok = False
        else:
            if emag >= max_err_bad:
                issues.append(f"large mag error ({emag:.3f} >= {max_err_bad})")
                ok = False
            elif emag >= max_err_warn:
                issues.append(f"mag error is somewhat large ({emag:.3f} >= {max_err_warn})")

        # Qflg checks
        if q is None:
            issues.append("missing Qflg")
            ok = False
        else:
            if q in ("U", "E", "F", "X"):
                issues.append(f"Qflg={q} indicates non-detection or unusable photometry")
                ok = False
            elif q == "D":
                if strict:
                    issues.append("Qflg=D indicates low-quality photometry")
                    ok = False
                else:
                    issues.append("Qflg=D indicates low-quality photometry")
            else:
                if strict:
                    if q != "A":
                        issues.append(f"Qflg={q} (strict expects A)")
                        ok = False
                else:
                    if q not in ("A", "B"):
                        issues.append(f"Qflg={q} (expects A/B)")
                        ok = False

        # Rflg checks
        # Good/default acceptable values are 1, 2, 3
        # Problematic values are typically 0, 4, 6, 9
        if r is None:
            issues.append("missing Rflg")
            ok = False
        else:
            if r not in ("1", "2", "3"):
                issues.append(f"Rflg={r} indicates non-standard or low-quality photometry")
                ok = False

        # Bflg checks
        # 0: no reliable detection or inconsistent deblending
        # 1: single component fit / acceptable
        # >1: blended source
        if b is None:
            issues.append("missing Bflg")
            ok = False
        else:
            try:
                b_int = int(b)
                if b_int == 0:
                    issues.append("Bflg=0 indicates no reliable detection or inconsistent deblending")
                    ok = False
                elif b_int > 1:
                    issues.append(f"Bflg={b} indicates blending ({b} components fit)")
                    if strict:
                        ok = False
            except Exception:
                issues.append(f"Bflg={b} is not a standard value")
                ok = False

        # Global flags affect all bands
        if strict and (contam or xtalk or assoc):
            issues.append("global flags (C/X/A) indicate potential issues")
            ok = False

        return ok, issues

    # --- evaluate each band ---
    okJ, issuesJ = _eval_band("J", J_mag, J_err_mag, qJ, rJ, bJ)
    okH, issuesH = _eval_band("H", H_mag, H_err_mag, qH, rH, bH)
    okK, issuesK = _eval_band("Ks", Ks_mag, Ks_err_mag, qK, rK, bK)

    per_band["J"] = {"ok": okJ, "issues": issuesJ}
    per_band["H"] = {"ok": okH, "issues": issuesH}
    per_band["Ks"] = {"ok": okK, "issues": issuesK}

    # --- collect band warnings ---
    for band, info in per_band.items():
        if info["issues"]:
            warnings.append(f"Warning: 2MASS {band}-band issues: " + "; ".join(info["issues"]) + ".")

    overall_ok = (okJ and okH and okK) and (not (strict and (contam or xtalk or assoc)))

    result = {
        "ok": overall_ok,
        "warnings": warnings,
        "per_band": per_band
    }

    if verbose and warnings:
        for w in warnings:
            print(w)

    return result




def query_gaia_dr3_vizier(gaia_dr3_id, row_limit=1, verbose=True, snr_lower_limit=10, n_lower_limit=5):
    """
    Same as your version, with robustness fixes:
      - Photometric uncertainties are computed ONLY by flux propagation:
            sigma_mag = (2.5 / ln(10)) * (flux_error / flux)
        (no fallback to CDS mag_error columns)
      - Add robust handling for missing columns / division by zero
      - Add optional warnings for low n_obs and low flux_over_error
      - Make E(BP-RP) retrieval optional (NaN if not present)
    """
    from astroquery.vizier import Vizier

    MAG_ERR_FACTOR = 2.5 / np.log(10.0)  # 1.085736...

    def _nan():
        return np.nan

    def _safe_float(x):
        try:
            if x is None:
                return np.nan
            s = str(x).strip()
            if s in ("", "--", "nan", "None"):
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _safe_str(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        return s if s not in ("", "--", "nan", "None") else np.nan

    def _pick_first(row, colset, candidates):
        for c in candidates:
            if c in colset:
                return row[c]
        return None

    def _safe_div(a, b):
        a = _safe_float(a)
        b = _safe_float(b)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return np.nan
        return a / b

    def _to_pyfloat(x):
        # unify np.float64 -> float
        try:
            return float(x)
        except Exception:
            return np.nan

    # ---- normalize id (avoid float precision loss) ----
    sid = "" if gaia_dr3_id is None else str(gaia_dr3_id).strip()
    if sid == "" or sid.lower() in ("nan", "none"):
        out = {
            "source_id": sid,
            "photometry": {"G": _nan(), "BP": _nan(), "RP": _nan(), "e_G": _nan(), "e_BP": _nan(), "e_RP": _nan()},
            "astrometry": {"parallax": _nan(), "e_parallax": _nan(), "RUWE": _nan()},
            "rotation": {"Vbroad": _nan(), "e_Vbroad": _nan()},
            "Pbin": _nan(),
            "flags": {"is_nss": False, "is_variable": False, "is_spurious": False},
            "warnings": ["Empty Gaia DR3 ID."],
            "raw": {},
            "EBP_RP": _nan(),
            "EBP_RP_lower_error": _nan(),
            "EBP_RP_upper_error": _nan(),
        }
        if verbose:
            print(out["warnings"][0])
        return out

    warnings = []
    raw = {}

    phot = {"G": _nan(), "BP": _nan(), "RP": _nan(), "e_G": _nan(), "e_BP": _nan(), "e_RP": _nan()}
    astro = {"parallax": _nan(), "e_parallax": _nan(), "RUWE": _nan()}
    rot = {"Vbroad": _nan(), "e_Vbroad": _nan()}
    Pbin = _nan()

    # optional outputs
    EBP_RP = _nan()
    EBP_RP_lower_error = _nan()
    EBP_RP_upper_error = _nan()

    # -----------------------------
    # 1) I/355 main
    # -----------------------------
    v355_cols = [
        "Source", "source_id",

        # mags (two naming styles)
        "Gmag", "BPmag", "RPmag",
        "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",

        # fluxes / flux errors (two naming styles)
        "FG", "e_FG", "RFG",
        "FBP", "e_FBP", "RFBP",
        "FRP", "e_FRP", "RFRP",

        # obs counts (two naming styles)
        "o_Gmag", "o_BPmag", "o_RPmag",
        "phot_g_n_obs", "phot_bp_n_obs", "phot_rp_n_obs",

        # astrometry
        "Plx", "e_Plx", "parallax", "parallax_error",

        # vbroad
        "Vbroad", "e_Vbroad", "Vbroad_error", "o_Vbroad",

        # RUWE
        "RUWE", "ruwe",

        # (optional, might not exist in this table view)
        "E(BP-RP)", "b_E(BP-RP)", "B_E(BP-RP)",
        "E(BP/RP)", "b_E(BP/RP)", "B_E(BP/RP)",
    ]
    v355 = Vizier(columns=v355_cols, row_limit=row_limit)

    try:
        res355 = v355.query_constraints(catalog="I/355", Source=sid)
        if res355 is None or len(res355) == 0 or len(res355[0]) == 0:
            res355 = v355.query_constraints(catalog="I/355", source_id=sid)

        if res355 is not None and len(res355) > 0 and len(res355[0]) > 0:
            tab = res355[0]
            row = tab[0]
            cols = set(tab.colnames)

            # ---- magnitudes ----
            phot["G"] = _safe_float(_pick_first(row, cols, ["Gmag", "phot_g_mean_mag"]))
            phot["BP"] = _safe_float(_pick_first(row, cols, ["BPmag", "phot_bp_mean_mag"]))
            phot["RP"] = _safe_float(_pick_first(row, cols, ["RPmag", "phot_rp_mean_mag"]))

            # ---- fluxes ----
            g_flux = _safe_float(_pick_first(row, cols, ["phot_g_mean_flux", "FG"]))
            g_flux_err = _safe_float(_pick_first(row, cols, ["phot_g_mean_flux_error", "e_FG"]))
            bp_flux = _safe_float(_pick_first(row, cols, ["phot_bp_mean_flux", "FBP"]))
            bp_flux_err = _safe_float(_pick_first(row, cols, ["phot_bp_mean_flux_error", "e_FBP"]))
            rp_flux = _safe_float(_pick_first(row, cols, ["phot_rp_mean_flux", "FRP"]))
            rp_flux_err = _safe_float(_pick_first(row, cols, ["phot_rp_mean_flux_error", "e_FRP"]))

            # flux_over_err: prefer provided RFG/RFBP/RFRP, else compute
            g_flux_over_err = _safe_float(_pick_first(row, cols, ["phot_g_mean_flux_over_error", "RFG"]))
            bp_flux_over_err = _safe_float(_pick_first(row, cols, ["phot_bp_mean_flux_over_error", "RFBP"]))
            rp_flux_over_err = _safe_float(_pick_first(row, cols, ["phot_rp_mean_flux_over_error", "RFRP"]))

            if not np.isfinite(g_flux_over_err):
                g_flux_over_err = _safe_div(g_flux, g_flux_err)
            if not np.isfinite(bp_flux_over_err):
                bp_flux_over_err = _safe_div(bp_flux, bp_flux_err)
            if not np.isfinite(rp_flux_over_err):
                rp_flux_over_err = _safe_div(rp_flux, rp_flux_err)

            # ---- obs counts (Gaia archive style OR CDS style) ----
            g_n_obs = _safe_float(_pick_first(row, cols, ["phot_g_n_obs", "o_Gmag"]))
            bp_n_obs = _safe_float(_pick_first(row, cols, ["phot_bp_n_obs", "o_BPmag"]))
            rp_n_obs = _safe_float(_pick_first(row, cols, ["phot_rp_n_obs", "o_RPmag"]))

            # warning for low obs (do not interrupt)
            low_obs = []
            if np.isfinite(g_n_obs) and g_n_obs < n_lower_limit:
                low_obs.append(f"G (n_obs={int(g_n_obs)})")
            if np.isfinite(bp_n_obs) and bp_n_obs < n_lower_limit:
                low_obs.append(f"BP (n_obs={int(bp_n_obs)})")
            if np.isfinite(rp_n_obs) and rp_n_obs < n_lower_limit:
                low_obs.append(f"RP (n_obs={int(rp_n_obs)})")
            if low_obs:
                warnings.append(
                    f"Warning: Gaia photometry has <{n_lower_limit} observations in " + ", ".join(low_obs) + "."
                )

            # warning for low SNR (do not interrupt)
            low_snr = []
            if np.isfinite(g_flux_over_err) and g_flux_over_err < snr_lower_limit:
                low_snr.append(f"G (snr={g_flux_over_err:.1f})")
            if np.isfinite(bp_flux_over_err) and bp_flux_over_err < snr_lower_limit:
                low_snr.append(f"BP (snr={bp_flux_over_err:.1f})")
            if np.isfinite(rp_flux_over_err) and rp_flux_over_err < snr_lower_limit:
                low_snr.append(f"RP (snr={rp_flux_over_err:.1f})")
            if low_snr:
                warnings.append(
                    f"Warning: Gaia photometry has flux_over_error <{snr_lower_limit} in " + ", ".join(low_snr) + "."
                )

            # ---- photometric uncertainties (ONLY your method) ----
            # per-band compute; avoid vector divide problems
            eG = np.nan if (not np.isfinite(g_flux) or not np.isfinite(g_flux_err) or g_flux <= 0) else MAG_ERR_FACTOR * (g_flux_err / g_flux)
            eBP = np.nan if (not np.isfinite(bp_flux) or not np.isfinite(bp_flux_err) or bp_flux <= 0) else MAG_ERR_FACTOR * (bp_flux_err / bp_flux)
            eRP = np.nan if (not np.isfinite(rp_flux) or not np.isfinite(rp_flux_err) or rp_flux <= 0) else MAG_ERR_FACTOR * (rp_flux_err / rp_flux)

            phot["e_G"] = _to_pyfloat(eG)
            phot["e_BP"] = _to_pyfloat(eBP)
            phot["e_RP"] = _to_pyfloat(eRP)

            # ---- parallax ----
            astro["parallax"] = _safe_float(_pick_first(row, cols, ["Plx", "parallax"]))
            astro["e_parallax"] = _safe_float(_pick_first(row, cols, ["e_Plx", "parallax_error"]))

            # ---- RUWE ----
            astro["RUWE"] = _safe_float(_pick_first(row, cols, ["RUWE", "ruwe"]))

            # ---- vbroad + uncertainty ----
            rot["Vbroad"] = _safe_float(_pick_first(row, cols, ["Vbroad"]))
            rot["e_Vbroad"] = _safe_float(_pick_first(row, cols, ["e_Vbroad", "Vbroad_error"]))

            # ---- optional E(BP-RP) ----
            # different mirrors may name it E(BP-RP) or E(BP/RP)
            EBP_RP = _safe_float(_pick_first(row, cols, ["E(BP-RP)", "E(BP/RP)"]))
            EBP_RP_lower_error = _safe_float(_pick_first(row, cols, ["b_E(BP-RP)", "b_E(BP/RP)"]))
            EBP_RP_upper_error = _safe_float(_pick_first(row, cols, ["B_E(BP-RP)", "B_E(BP/RP)"]))

        else:
            warnings.append(f"I/355: No match for source_id={sid} (VizieR).")

    except Exception as e:
        warnings.append(f"I/355 query failed: {type(e).__name__}: {e}")

    # -----------------------------
    # 1.5) I/355/paramp: Pbin
    # -----------------------------
    try:
        vparamp = Vizier(columns=["**"], row_limit=1)
        res_p = vparamp.query_constraints(catalog="I/355/paramp", Source=sid)
        if res_p is None or len(res_p) == 0 or len(res_p[0]) == 0:
            res_p = vparamp.query_constraints(catalog="I/355/paramp", source_id=sid)

        if res_p is not None and len(res_p) > 0 and len(res_p[0]) > 0:
            tabp = res_p[0]
            rowp = tabp[0]
            colsp = set(tabp.colnames)
            Pbin = _safe_float(_pick_first(rowp, colsp, ["Pbin", "pbin", "p_bin", "p_binary"]))
    except Exception as e:
        warnings.append(f"I/355/paramp query failed: {type(e).__name__}: {e}")

    # -----------------------------
    # 2) I/357 Non-single stars
    # -----------------------------
    is_nss = False
    try:
        v357 = Vizier(columns=["**"], row_limit=1)
        res357 = v357.query_constraints(catalog="I/357", Source=sid)
        if res357 is None or len(res357) == 0 or len(res357[0]) == 0:
            res357 = v357.query_constraints(catalog="I/357", source_id=sid)

        if res357 is not None and len(res357) > 0 and len(res357[0]) > 0:
            is_nss = True
            tab = res357[0]
            row = tab[0]
            raw["I/357"] = {c: _safe_str(row[c]) for c in tab.colnames[:30]}
            warnings.append("Warning: source appears in I/357 (Non-single stars). Treat as binary/multiple candidate.")
    except Exception as e:
        warnings.append(f"I/357 query failed: {type(e).__name__}: {e}")

    # -----------------------------
    # 3) I/358 Variability (STRICT): varisum OR vclassre
    # -----------------------------
    is_variable = False
    var_tables = []
    try:
        v_varisum = Vizier(columns=["**"], row_limit=1)
        res_varisum = v_varisum.query_constraints(catalog="I/358/varisum", Source=sid)
        if res_varisum is None or len(res_varisum) == 0 or len(res_varisum[0]) == 0:
            res_varisum = v_varisum.query_constraints(catalog="I/358/varisum", source_id=sid)
        if res_varisum is not None and len(res_varisum) > 0 and len(res_varisum[0]) > 0:
            is_variable = True
            tab = res_varisum[0]
            row = tab[0]
            raw["I/358/varisum"] = {c: _safe_str(row[c]) for c in tab.colnames[:30]}
            var_tables.append("varisum")

        v_vclass = Vizier(columns=["**"], row_limit=1)
        res_vclass = v_vclass.query_constraints(catalog="I/358/vclassre", Source=sid)
        if res_vclass is None or len(res_vclass) == 0 or len(res_vclass[0]) == 0:
            res_vclass = v_vclass.query_constraints(catalog="I/358/vclassre", source_id=sid)
        if res_vclass is not None and len(res_vclass) > 0 and len(res_vclass[0]) > 0:
            is_variable = True
            tab = res_vclass[0]
            row = tab[0]
            raw["I/358/vclassre"] = {c: _safe_str(row[c]) for c in tab.colnames[:30]}
            var_tables.append("vclassre")

        if is_variable:
            warnings.append(
                "Warning: source appears in Gaia DR3 variability tables (" + ", ".join(var_tables) + "). Treat as variable candidate."
            )
    except Exception as e:
        warnings.append(f"I/358 variability query failed: {type(e).__name__}: {e}")

    # -----------------------------
    # 4) Holl+2023 spurious signals (J/A+A/674/A25)
    # -----------------------------
    is_spurious = False
    try:
        vA25 = Vizier(columns=["**"], row_limit=1)
        resA25 = vA25.query_constraints(catalog="J/A+A/674/A25", Source=sid)
        if resA25 is not None and len(resA25) > 0 and len(resA25[0]) > 0:
            is_spurious = True
            tab = resA25[0]
            row = tab[0]
            raw["J/A+A/674/A25"] = {c: _safe_str(row[c]) for c in tab.colnames[:40]}
            warnings.append("Warning: source appears in Holl+2023 spurious-signal catalog (J/A+A/674/A25). Use astrometry with caution.")
    except Exception as e:
        warnings.append(f"J/A+A/674/A25 query failed: {type(e).__name__}: {e}")

    out = {
        "source_id": sid,
        "photometry": phot,
        "astrometry": astro,
        "rotation": rot,
        "Pbin": Pbin,
        "flags": {"is_nss": is_nss, "is_variable": is_variable, "is_spurious": is_spurious},
        "warnings": warnings,
        "raw": raw,

        # optional: may be NaN if not provided by the table view
        "EBP_RP": EBP_RP,
        "EBP_RP_lower_error": EBP_RP_lower_error,
        "EBP_RP_upper_error": EBP_RP_upper_error,
    }

    if verbose and warnings:
        for w in warnings:
            print(w)

    return out



def print_summary_card_aligned(
    target_name,
    simbad_info,
    gaia_res,
    tycho_vals,
    mass_vals,
    *,
    total_width=120,
    col1=34,
    col2=45,
    col3=40,
):
    """
    Stable 3-column aligned summary card.

    Updates:
    1) Distance(pc) now includes propagated uncertainty from Gaia parallax:
         d = 1000/plx(mas),  sigma_d = 1000*sigma_plx / plx^2
    2) Middle-column labels have ':' aligned to the longest label (dynamic).
       e.g. "E(BP-RP)" and "Parallax" won't shift the ':' positions.
    """

    def isfin(x):
        try:
            return x is not None and np.isfinite(float(x))
        except Exception:
            return False

    def fnum(x, w=8, p=4):
        if not isfin(x):
            return "nan".rjust(w)
        return f"{float(x):{w}.{p}f}"

    def fval(x, default="nan"):
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        s = str(x).strip()
        return s if s.lower() not in ("nan", "none", "") else default

    def row3(a, b, c):
        a = (a if a is not None else "").ljust(col1)
        b = (b if b is not None else "").ljust(col2)
        c = (c if c is not None else "").ljust(col3)
        return f"{a} | {b} | {c}"

    # --- unpack SIMBAD ---
    HIP, HD, GaiaID, TYC, TMASS, Other, SpT, ObjT, RA, DEC = simbad_info

    # --- Gaia ---
    phot  = gaia_res.get("photometry", {})
    astro = gaia_res.get("astrometry", {})
    rot   = gaia_res.get("rotation", {})
    flags = gaia_res.get("flags", {})

    EBP_RP    = gaia_res.get("EBP_RP", np.nan)
    EBP_RP_lo = gaia_res.get("EBP_RP_lower_error", np.nan)
    EBP_RP_hi = gaia_res.get("EBP_RP_upper_error", np.nan)

    plx  = astro.get("parallax", np.nan)
    eplx = astro.get("e_parallax", np.nan)

    # distance + uncertainty (pc)
    if isfin(plx) and float(plx) > 0:
        plx_f = float(plx)
        dist_pc = 1000.0 / plx_f
        if isfin(eplx):
            eplx_f = float(eplx)
            dist_pc_err = 1000.0 * eplx_f / (plx_f ** 2)
        else:
            dist_pc_err = np.nan
    else:
        dist_pc = np.nan
        dist_pc_err = np.nan

    # --- Tycho + 2MASS ---
    BT, VT, eBT, eVT = tycho_vals
    J, H, Ks, eJ, eH, eKs, Q, R, B, C, X, A = mass_vals

    # --- header ---
    W = total_width
    print("=" * W)
    print(f"Target : {target_name}")
    print(f"IDs    : Gaia DR3 {fval(GaiaID)} | HIP {fval(HIP)} | HD {fval(HD)} | 2MASS {fval(TMASS)}")
    # print(f"Type   : {fval(SpT)} | {fval(ObjT)}")
    print("-" * W)

    # -----------------------------
    # Middle column: ':' aligned
    # -----------------------------
    mid_labels = ["G", "G_BP", "G_RP", "Parallax", "RUWE", "Vbroad", "E(BP-RP)"]
    labw = max(len(x) for x in mid_labels)

    def mid_kv(label, rhs):
        # "label<labw> : rhs"
        return f"{label:<{labw}} : {rhs}"

    mid_G = mid_kv("G", f"{fnum(phot.get('G'),8,4)} ± {fnum(phot.get('e_G'),8,4)} mag")
    mid_BP = mid_kv("G_BP", f"{fnum(phot.get('BP'),8,4)} ± {fnum(phot.get('e_BP'),8,4)} mag")
    mid_RP = mid_kv("G_RP", f"{fnum(phot.get('RP'),8,4)} ± {fnum(phot.get('e_RP'),8,4)} mag")
    mid_Plx = mid_kv("Parallax", f"{fnum(plx,8,4)} ± {fnum(eplx,8,4)} mas")
    mid_RUWE = mid_kv("RUWE", f"{fnum(astro.get('RUWE'),8,4)}")
    mid_Vb = mid_kv(
        "Vbroad",
        f"{fnum(rot.get('Vbroad'),8,)} ± {fnum(rot.get('e_Vbroad'),8,2)} km/s"
    )

    if isfin(EBP_RP):
        mid_E_rhs = (
            f"{fnum(EBP_RP,8,4)} "
            f"(-{fnum(EBP_RP_lo,6,4)}, +{fnum(EBP_RP_hi,6,4)}) mag"
        )
    else:
        mid_E_rhs = "nan"
    mid_E = mid_kv("E(BP-RP)", mid_E_rhs)

    # -----------------------------
    # Right column
    # -----------------------------
    right_BT = f"B_T : {fnum(BT,8,4)} ± {fnum(eBT,8,4)} mag"
    right_VT = f"V_T : {fnum(VT,8,4)} ± {fnum(eVT,8,4)} mag"
    right_sep = "-" * 25
    right_J  = f"J   : {fnum(J,8,4)} ± {fnum(eJ,8,4)} mag"
    right_H  = f"H   : {fnum(H,8,4)} ± {fnum(eH,8,4)} mag"
    right_K  = f"Ks  : {fnum(Ks,8,4)} ± {fnum(eKs,8,4)} mag"
    # right_f  = f"Q={fval(Q)}    R={fval(R)}    B={fval(B)}"

    # -----------------------------
    # Left column
    # -----------------------------
    left_RA  =  f"RA            : {fnum(RA,8,4).strip()}"
    left_Dec =  f"Dec           : {fnum(DEC,8,4).strip()}"
    left_SpT =  f"Spectral Type : {fval(SpT):<10}"
    left_Obj =  f"Object Type   : {fval(ObjT):<10}"
    left_Dist = f"Distance      : {fnum(dist_pc,8,2).strip()} ± {fnum(dist_pc_err,8,2).strip()} pc"

    # --- lines (all 3 columns) ---
    print(row3(left_RA,   mid_G,   right_BT))
    print(row3(left_Dec,  mid_BP,  right_VT))
    print(row3(left_SpT,  mid_RP,  right_sep))
    print(row3(left_Obj,  mid_Plx, right_J))
    print(row3(left_Dist, mid_RUWE, right_H))
    print(row3("",        mid_Vb,   right_K))
    print(row3("",        mid_E,    ""))

    print("-" * W)

    print(
        f"{'Gaia flags':<12}: "
        f"{'NSS=' + str(flags.get('is_nss', False)):<10} | "
        f"{'Var=' + str(flags.get('is_variable', False)):<10} | "
        f"{'Spurious=' + str(flags.get('is_spurious', False)):<14}"
    )

    print(
        f"{'2MASS flags':<12}: "
        f"{'Q=' + fval(Q):<10} | "
        f"{'R=' + fval(R):<10} | "
        f"{'B=' + fval(B):<14}"
    )

    print("=" * W)



def query_data(target_name, max_retries=3, return_df=False, verbose=True):
    """
    Query SIMBAD + Tycho-2 + 2MASS + Gaia DR3 (VizieR) for a target.

    Retry policy:
      - For each query step (SIMBAD / Tycho-2 / 2MASS / Gaia), retry up to max_retries
        if the returned result contains any NaN / None in critical fields.
      - Does NOT interrupt the pipeline; it just retries and keeps the latest result.
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def _has_nan_like(x):
        """True if x contains any None/NaN (works for scalars, lists/tuples, dicts, nested)."""
        if x is None:
            return True

        # numpy scalar
        try:
            if np.isscalar(x):
                import pandas as pd
                return bool(pd.isna(x))
        except Exception:
            pass

        # dict
        if isinstance(x, dict):
            for v in x.values():
                if _has_nan_like(v):
                    return True
            return False

        # list/tuple/np.ndarray
        if isinstance(x, (list, tuple, np.ndarray)):
            for v in x:
                if _has_nan_like(v):
                    return True
            return False

        # other objects -> treat as not-NaN
        return False

    def _retry_call(fn, fn_name, args=(), kwargs=None, ok_pred=None):
        """
        Retry wrapper.
        - ok_pred: function(result)->bool indicating "result is good enough".
                  If None: uses "no NaN-like" criterion.
        """
        if kwargs is None:
            kwargs = {}

        last = None
        for i_try in range(1, max_retries + 1):
            try:
                last = fn(*args, **kwargs)
            except Exception as e:
                last = None
                if verbose:
                    print(f"Warning: {fn_name} raised {type(e).__name__}: {e}. Retry {i_try}/{max_retries}.")
                continue

            if ok_pred is None:
                ok = (last is not None) and (not _has_nan_like(last))
            else:
                ok = (last is not None) and bool(ok_pred(last))

            if ok:
                return last

            if verbose:
                print(f"Warning: {fn_name} returned NaN fields. Retry {i_try}/{max_retries} for {target_name}")

        return last  # return best-effort even if still has NaN

    # -----------------------------
    # 1) SIMBAD query with retry
    # -----------------------------
    def _ok_simbad(res):
        # res should be 10-tuple
        import pandas as pd
        if res is None or not isinstance(res, (list, tuple)) or len(res) != 10:
            return False
        # must have ids and coordinates (you can relax if needed)
        HIP, HD, GaiaID, TYC, TMASS, Other, SpT, ObjT, RA, DEC = res
        # at minimum: GaiaID + RA/DEC should be valid to continue reliably
        return (not pd.isna(GaiaID)) and np.isfinite(float(RA)) and np.isfinite(float(DEC))

    name_res = _retry_call(
        get_info_from_simbad,
        "SIMBAD(get_info_from_simbad)",
        args=(target_name,),
        ok_pred=_ok_simbad
    )

    if name_res is None:
        raise RuntimeError("Simbad query returned None repeatedly.")

    (HIP_ID, HD_ID, Gaia_DR3_ID, TYC_ID, TwoMass_ID,
     Other_Name, Spectral_Type, Object_Type, RA, DEC) = name_res

    # -----------------------------
    # 2) Tycho-2 photometry (retry)
    # -----------------------------
    def _ok_tycho(res):
        import pandas as pd
        # expect 4-tuple
        if res is None or not isinstance(res, (list, tuple)) or len(res) != 4:
            return False
        BT, VT, eBT, eVT = res
        # allow missing if no TYC id
        if pd.isna(TYC_ID) or TYC_ID in ("nan", "None", "", None):
            return True
        # otherwise require BT/VT (+ errors optional but preferred)
        return np.isfinite(BT) and np.isfinite(VT)

    result_tycho2 = _retry_call(
        query_tycho_2,
        "Tycho-2(query_tycho_2)",
        args=(TYC_ID,),
        ok_pred=_ok_tycho
    )
    B_T_mag, V_T_mag, B_T_err_mag, V_T_err_mag = result_tycho2 if result_tycho2 is not None else (np.nan,)*4

    # -----------------------------
    # 3) 2MASS photometry (retry)
    # -----------------------------
    def _ok_2mass(res):
        import pandas as pd
        # expect 12-tuple
        if res is None or not isinstance(res, (list, tuple)) or len(res) != 12:
            return False
        J, H, Ks, eJ, eH, eKs, Q, R, B, C, X, A = res
        # allow missing if no 2MASS id
        if pd.isna(TwoMass_ID) or TwoMass_ID in ("nan", "None", "", None):
            return True
        # otherwise require J/H/Ks and Q/R/B flags to exist
        return np.isfinite(J) and np.isfinite(H) and np.isfinite(Ks) and (not pd.isna(Q))

    result_2mass = _retry_call(
        query_2mass,
        "2MASS(query_2mass)",
        args=(TwoMass_ID,),
        ok_pred=_ok_2mass
    )
    if result_2mass is None:
        result_2mass = (np.nan,)*12

    (J_mag, H_mag, Ks_mag,
     J_err_mag, H_err_mag, Ks_err_mag,
     Q_2mass, R_2mass, B_2mass, C_2mass, X_2mass, A_2mass) = result_2mass

    quality_2mass = assess_2mass_phot_quality(*result_2mass, strict=True, verbose=verbose)

    # -----------------------------
    # 4) Gaia DR3 (VizieR) query (retry)
    # -----------------------------
    def _ok_gaia(res):
        if res is None or not isinstance(res, dict):
            return False
        phot = res.get("photometry", {})
        astro = res.get("astrometry", {})
        # require at least G/BP/RP and plx
        return (
            np.isfinite(phot.get("G", np.nan))
            and np.isfinite(phot.get("BP", np.nan))
            and np.isfinite(phot.get("RP", np.nan))
            and np.isfinite(astro.get("parallax", np.nan))
        )

    gaia_dr3_res = _retry_call(
        query_gaia_dr3_vizier,
        "GaiaDR3(query_gaia_dr3_vizier)",
        args=(Gaia_DR3_ID,),
        kwargs={"verbose": verbose},
        ok_pred=_ok_gaia
    )

    if gaia_dr3_res is None:
        gaia_dr3_res = {
            "source_id": str(Gaia_DR3_ID),
            "photometry": {"G": np.nan, "BP": np.nan, "RP": np.nan, "e_G": np.nan, "e_BP": np.nan, "e_RP": np.nan},
            "astrometry": {"parallax": np.nan, "e_parallax": np.nan, "RUWE": np.nan},
            "rotation": {"Vbroad": np.nan, "e_Vbroad": np.nan},
            "Pbin": np.nan,
            "flags": {"is_nss": False, "is_variable": False, "is_spurious": False},
            "warnings": ["Gaia DR3 query failed repeatedly."],
            "raw": {}
        }

    # photometry
    G_mag = gaia_dr3_res["photometry"]["G"]
    G_BP_mag = gaia_dr3_res["photometry"]["BP"]
    G_RP_mag = gaia_dr3_res["photometry"]["RP"]

    G_err_mag = gaia_dr3_res["photometry"]["e_G"]
    G_BP_err_mag = gaia_dr3_res["photometry"]["e_BP"]
    G_RP_err_mag = gaia_dr3_res["photometry"]["e_RP"]

    # astrometry
    parallax_obs = gaia_dr3_res["astrometry"]["parallax"]
    parallax_obs_err = gaia_dr3_res["astrometry"]["e_parallax"]
    RUWE = gaia_dr3_res["astrometry"].get("RUWE", np.nan)

    # rotation (vbroad)
    vsini_obs = gaia_dr3_res["rotation"]["Vbroad"]
    vsini_obs_err = gaia_dr3_res["rotation"]["e_Vbroad"]

    # Pbin
    P_binary = gaia_dr3_res.get("Pbin", np.nan)

    # flags / warnings / raw
    gaia_flags = gaia_dr3_res.get("flags", {})
    gaia_warnings = gaia_dr3_res.get("warnings", [])
    gaia_raw = gaia_dr3_res.get("raw", {})

    # E(BP-RP)
    EBP_RP_obs = gaia_dr3_res.get("EBP_RP", np.nan)
    EBP_RP_obs_lower = gaia_dr3_res.get("EBP_RP_lower_error", np.nan)
    EBP_RP_obs_upper = gaia_dr3_res.get("EBP_RP_upper_error", np.nan)

    # -----------------------------
    # 5) Pack outputs
    # -----------------------------
    obs_mags = np.array(
        [G_mag, G_BP_mag, G_RP_mag, B_T_mag, V_T_mag, J_mag, H_mag, Ks_mag],
        dtype=float
    )
    obs_err_mags = np.array(
        [G_err_mag, G_BP_err_mag, G_RP_err_mag, B_T_err_mag, V_T_err_mag, J_err_mag, H_err_mag, Ks_err_mag],
        dtype=float
    )

    additional_info = {
        # ids + simbad info
        "target_name": target_name,
        "HIP_ID": HIP_ID,
        "HD_ID": HD_ID,
        "Gaia_DR3_ID": Gaia_DR3_ID,
        "TYC_ID": TYC_ID,
        "TwoMass_ID": TwoMass_ID,
        "Other_Name": Other_Name,
        "Spectral_Type": Spectral_Type,
        "Object_Type": Object_Type,
        "RA": RA,
        "DEC": DEC,

        # 2mass flags + quality
        "Q_2mass": Q_2mass,
        "R_2mass": R_2mass,
        "B_2mass": B_2mass,
        "C_2mass": C_2mass,
        "X_2mass": X_2mass,
        "A_2mass": A_2mass,
        "2mass_ok": bool(quality_2mass["ok"]),
        "2mass_warnings": quality_2mass.get("warnings", []),

        # Gaia extras
        "RUWE": RUWE,
        "P_binary": P_binary,
        "gaia_flags": gaia_flags,
        "gaia_warnings": gaia_warnings,
        "gaia_raw": gaia_raw,

        # E(BP-RP)
        "EBP_RP": EBP_RP_obs,
        "EBP_RP_lower_error": EBP_RP_obs_lower,
        "EBP_RP_upper_error": EBP_RP_obs_upper,
    }

    # -----------------------------
    # 6) print query results
    # -----------------------------
    print_summary_card_aligned(target_name, name_res, gaia_dr3_res, result_tycho2, result_2mass)

    # -----------------------------
    # 7) Return dataframe if requested
    # -----------------------------
    if return_df:
        row = {
            # ids / meta
            "target_name": target_name,
            "HIP_ID": HIP_ID,
            "HD_ID": HD_ID,
            "Gaia_DR3_ID": Gaia_DR3_ID,
            "TYC_ID": TYC_ID,
            "TwoMass_ID": TwoMass_ID,
            "Other_Name": Other_Name,
            "Spectral_Type": Spectral_Type,
            "Object_Type": Object_Type,
            "RA": RA,
            "DEC": DEC,

            # Gaia mags
            "G_mag": G_mag, "G_err": G_err_mag,
            "G_BP_mag": G_BP_mag, "G_BP_err": G_BP_err_mag,
            "G_RP_mag": G_RP_mag, "G_RP_err": G_RP_err_mag,

            # Tycho mags
            "B_T_mag": B_T_mag, "B_T_err": B_T_err_mag,
            "V_T_mag": V_T_mag, "V_T_err": V_T_err_mag,

            # 2MASS mags
            "J_mag": J_mag, "J_err": J_err_mag,
            "H_mag": H_mag, "H_err": H_err_mag,
            "Ks_mag": Ks_mag, "Ks_err": Ks_err_mag,

            # Gaia astrometry/rotation
            "parallax": parallax_obs,
            "parallax_err": parallax_obs_err,
            "vbroad": vsini_obs,
            "vbroad_err": vsini_obs_err,
            "RUWE": RUWE,
            "P_binary": P_binary,

            # Gaia reddening
            "EBP_RP": EBP_RP_obs,
            "EBP_RP_lower_error": EBP_RP_obs_lower,
            "EBP_RP_upper_error": EBP_RP_obs_upper,

            # flags
            "Q_2mass": Q_2mass,
            "R_2mass": R_2mass,
            "B_2mass": B_2mass,
            "C_2mass": C_2mass,
            "X_2mass": X_2mass,
            "A_2mass": A_2mass,
            "2mass_ok": bool(quality_2mass["ok"]),
            "gaia_is_nss": bool(gaia_flags.get("is_nss", False)),
            "gaia_is_variable": bool(gaia_flags.get("is_variable", False)),
            "gaia_is_spurious": bool(gaia_flags.get("is_spurious", False)),

            # warnings as strings
            "2mass_warnings": "; ".join(map(str, quality_2mass.get("warnings", []))),
            "gaia_warnings": "; ".join(map(str, gaia_warnings)),
        }
        import pandas as pd

        return pd.DataFrame([row])

    return obs_mags, obs_err_mags, vsini_obs, vsini_obs_err, parallax_obs, parallax_obs_err, EBP_RP_obs, EBP_RP_obs_lower, EBP_RP_obs_upper, additional_info


