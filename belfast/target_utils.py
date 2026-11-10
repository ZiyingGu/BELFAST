import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from belfast.config import row_limit, A_lambda_over_A_V_dic_default, A_lambda_over_EBP_RP_dic_default


def cal_parsec(parallax, parallax_err=None):
    """
    Convert parallax (mas) to distance (pc).

    Args:
        parallax (float or array-like): Parallax in milliarcseconds (mas).
        parallax_err (float or array-like, optional): Parallax uncertainty in mas.

    Returns:
        float or array-like: Distance in parsecs (pc), if parallax_err is None.
        tuple: (distance_pc, distance_err_pc), if parallax_err is provided.
    """
    distance_pc = 1000 / parallax

    if parallax_err is None:
        return distance_pc

    distance_err_pc = distance_pc * parallax_err / parallax
    return distance_pc, distance_err_pc



def cal_abs_mag(obs_mags, obs_err_mags, parallax_obs, parallax_obs_err, max_sigma_adjustment_parameter=None):
    """
    Calculate absolute magnitudes and uncertainties from apparent magnitudes and parallax.

    Args:
        obs_mags (array-like): Observed apparent magnitudes in mag.
        obs_err_mags (array-like): Uncertainties in apparent magnitudes in mag.
        parallax_obs (float): Observed parallax in mas.
        parallax_obs_err (float): Uncertainty in parallax in mas.
        max_sigma_adjustment_parameter (float, optional): Additional magnitude uncertainty in mag.

    Returns:
        tuple: (abs_mags, abs_err_mags), flattened NumPy arrays in mag.
    """
    obs_mags = np.asarray(obs_mags, dtype=float)
    obs_err_mags = np.asarray(obs_err_mags, dtype=float)

    if obs_mags.shape != obs_err_mags.shape:
        raise ValueError("obs_mags and obs_err_mags must have the same shape.")
    if not np.all(np.isfinite(obs_mags)) or not np.all(np.isfinite(obs_err_mags)):
        raise ValueError("Magnitudes and their uncertainties must be finite.")
    if np.any(obs_err_mags < 0):
        raise ValueError("obs_err_mags must be non-negative.")

    parallax_obs = float(parallax_obs)
    parallax_obs_err = float(parallax_obs_err)

    if not np.isfinite(parallax_obs) or parallax_obs <= 0:
        raise ValueError("parallax_obs must be finite and positive.")
    if not np.isfinite(parallax_obs_err) or parallax_obs_err < 0:
        raise ValueError("parallax_obs_err must be finite and non-negative.")

    if max_sigma_adjustment_parameter is not None:
        max_sigma_adjustment_parameter = float(max_sigma_adjustment_parameter)
        if not np.isfinite(max_sigma_adjustment_parameter) or max_sigma_adjustment_parameter < 0:
            raise ValueError("max_sigma_adjustment_parameter must be finite and non-negative.")
        obs_err_mags = np.sqrt(obs_err_mags**2 + max_sigma_adjustment_parameter**2)

    abs_mags = obs_mags - 5 * np.log10(100 / parallax_obs)
    abs_err_mags = np.sqrt(obs_err_mags**2 + (5 * parallax_obs_err / (np.log(10) * parallax_obs))**2)

    return abs_mags.flatten(), abs_err_mags.flatten()



# Calculate observed magnitudes from sampled parameters, without propagating uncertainties.
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



def remove_extinction(
    obs_mags_no_cor, obs_err_mags_no_cor, bands_list, extinction, extinction_err, extinction_type,
    R_V=3.16, R_V_err=0.15, verbose=True, A_lambda_over_EBP_RP_dic=None, A_lambda_over_A_V_dic=None
):
    """
    Correct observed magnitudes for interstellar extinction and propagate uncertainties.

    Args:
        obs_mags_no_cor (array-like): Observed magnitudes before extinction correction, in mag.
        obs_err_mags_no_cor (array-like): Uncertainties in observed magnitudes, in mag.
        bands_list (list of str): Photometric bands corresponding to the input magnitudes.
        extinction (float): Extinction value, defined by extinction_type.
        extinction_err (float): Uncertainty in extinction.
        extinction_type (str): Extinction definition: 'A_V', 'EB_V', or 'EBP_RP'.
        R_V (float): Total-to-selective extinction ratio A_V / E(B-V). Defaults to 3.16.
        R_V_err (float): Uncertainty in R_V. Defaults to 0.15.
        verbose (bool): Whether to print the extinction correction and uncertainty for each band.
        A_lambda_over_EBP_RP_dic (dict, optional): Band extinction coefficients relative to E(G_BP - G_RP).
        A_lambda_over_A_V_dic (dict, optional): Band extinction coefficients relative to A_V.

    Returns:
        tuple: (obs_mags_removed, obs_err_mags_removed), flattened NumPy arrays in mag.
    """
    if extinction_type not in ('A_V', 'EB_V', 'EBP_RP'):
        raise ValueError("extinction_type must be 'A_V', 'EB_V', or 'EBP_RP'.")

    obs_mags_no_cor = np.asarray(obs_mags_no_cor, dtype=float)
    obs_err_mags_no_cor = np.asarray(obs_err_mags_no_cor, dtype=float)
    bands_list = list(bands_list)

    if obs_mags_no_cor.ndim != 1 or obs_err_mags_no_cor.ndim != 1:
        raise ValueError("Observed magnitudes and uncertainties must be one-dimensional.")
    if obs_mags_no_cor.shape != obs_err_mags_no_cor.shape or len(bands_list) != obs_mags_no_cor.size:
        raise ValueError("Magnitudes, uncertainties, and bands_list must have the same length.")
    if not np.all(np.isfinite(obs_mags_no_cor)) or not np.all(np.isfinite(obs_err_mags_no_cor)):
        raise ValueError("Observed magnitudes and uncertainties must contain only finite values.")
    if np.any(obs_err_mags_no_cor < 0):
        raise ValueError("obs_err_mags_no_cor must be non-negative.")

    extinction = float(extinction)
    extinction_err = float(extinction_err)
    R_V = float(R_V)
    R_V_err = float(R_V_err)

    if not np.isfinite(extinction) or not np.isfinite(extinction_err) or extinction_err < 0:
        raise ValueError("extinction must be finite and extinction_err must be finite and non-negative.")
    if not np.isfinite(R_V) or not np.isfinite(R_V_err) or R_V_err < 0:
        raise ValueError("R_V must be finite and R_V_err must be finite and non-negative.")

    if A_lambda_over_EBP_RP_dic is None:
        A_lambda_over_EBP_RP_dic = A_lambda_over_EBP_RP_dic_default
    if A_lambda_over_A_V_dic is None:
        A_lambda_over_A_V_dic = A_lambda_over_A_V_dic_default

    if extinction_type == 'A_V':
        A_V, A_V_err = extinction, extinction_err
        coefficients = A_lambda_over_A_V_dic
        coefficient_label = 'A_V'
    elif extinction_type == 'EB_V':
        A_V = R_V * extinction
        A_V_err = np.sqrt((extinction * R_V_err)**2 + (R_V * extinction_err)**2)
        coefficients = A_lambda_over_A_V_dic
        coefficient_label = 'A_V'
    else:
        coefficients = A_lambda_over_EBP_RP_dic
        coefficient_label = 'E_BP_RP'

    correction = []
    correction_err = []

    for band_lambda in bands_list:
        if band_lambda not in coefficients or f'{band_lambda}_err' not in coefficients:
            raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")

        coefficient = coefficients[band_lambda]
        coefficient_err = coefficients[f'{band_lambda}_err']

        if not np.isfinite(coefficient) or not np.isfinite(coefficient_err) or coefficient_err < 0:
            raise ValueError(f"Invalid extinction coefficients for band: {band_lambda}")

        if extinction_type == 'EBP_RP':
            correction_lambda = coefficient * extinction
            correction_err_lambda = np.sqrt((coefficient * extinction_err)**2 + (extinction * coefficient_err)**2)
        else:
            correction_lambda = coefficient * A_V
            correction_err_lambda = np.sqrt((coefficient * A_V_err)**2 + (A_V * coefficient_err)**2)

        if verbose:
            print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")

        correction.append(correction_lambda)
        correction_err.append(correction_err_lambda)

    correction = np.asarray(correction)
    correction_err = np.asarray(correction_err)

    obs_mags_removed = obs_mags_no_cor - correction
    obs_err_mags_removed = np.sqrt(obs_err_mags_no_cor**2 + correction_err**2)

    return obs_mags_removed.flatten(), obs_err_mags_removed.flatten()



def add_extinction(
    obs_mags_model, obs_err_mags_model, bands_list, extinction, extinction_err, extinction_type,
    R_V=3.16, R_V_err=0.15, verbose=True, A_lambda_over_EBP_RP_dic=None, A_lambda_over_A_V_dic=None
):
    """
    Apply interstellar extinction to model magnitudes and propagate uncertainties.

    Args:
        obs_mags_model (array-like): Model magnitudes without extinction, in mag.
        obs_err_mags_model (array-like): Uncertainties in model magnitudes, in mag.
        bands_list (list of str): Photometric bands corresponding to the input magnitudes.
        extinction (float): Extinction value, defined by extinction_type.
        extinction_err (float): Uncertainty in extinction.
        extinction_type (str): Extinction definition: 'A_V', 'EB_V', or 'EBP_RP'.
        R_V (float): Total-to-selective extinction ratio A_V / E(B-V). Defaults to 3.16.
        R_V_err (float): Uncertainty in R_V. Defaults to 0.15.
        verbose (bool): Whether to print the extinction correction and uncertainty for each band.
        A_lambda_over_EBP_RP_dic (dict, optional): Band extinction coefficients relative to E(G_BP - G_RP).
        A_lambda_over_A_V_dic (dict, optional): Band extinction coefficients relative to A_V.

    Returns:
        tuple: (obs_mags_model_extincted, mags_ext_err_model_extincted), flattened NumPy arrays in mag.
    """
    if extinction_type not in ('A_V', 'EB_V', 'EBP_RP'):
        raise ValueError("extinction_type must be 'A_V', 'EB_V', or 'EBP_RP'.")

    obs_mags_model = np.asarray(obs_mags_model, dtype=float)
    obs_err_mags_model = np.asarray(obs_err_mags_model, dtype=float)
    bands_list = list(bands_list)

    if obs_mags_model.ndim != 1 or obs_err_mags_model.ndim != 1:
        raise ValueError("Model magnitudes and uncertainties must be one-dimensional.")
    if obs_mags_model.shape != obs_err_mags_model.shape or len(bands_list) != obs_mags_model.size:
        raise ValueError("Magnitudes, uncertainties, and bands_list must have the same length.")
    if not np.all(np.isfinite(obs_mags_model)) or not np.all(np.isfinite(obs_err_mags_model)):
        raise ValueError("Model magnitudes and uncertainties must contain only finite values.")
    if np.any(obs_err_mags_model < 0):
        raise ValueError("obs_err_mags_model must be non-negative.")

    extinction = float(extinction)
    extinction_err = float(extinction_err)
    R_V = float(R_V)
    R_V_err = float(R_V_err)

    if not np.isfinite(extinction) or not np.isfinite(extinction_err) or extinction_err < 0:
        raise ValueError("extinction must be finite and extinction_err must be finite and non-negative.")
    if not np.isfinite(R_V) or not np.isfinite(R_V_err) or R_V_err < 0:
        raise ValueError("R_V must be finite and R_V_err must be finite and non-negative.")

    if A_lambda_over_EBP_RP_dic is None:
        A_lambda_over_EBP_RP_dic = A_lambda_over_EBP_RP_dic_default
    if A_lambda_over_A_V_dic is None:
        A_lambda_over_A_V_dic = A_lambda_over_A_V_dic_default

    if extinction_type == 'A_V':
        A_V, A_V_err = extinction, extinction_err
        coefficients = A_lambda_over_A_V_dic
    elif extinction_type == 'EB_V':
        A_V = R_V * extinction
        A_V_err = np.sqrt((extinction * R_V_err)**2 + (R_V * extinction_err)**2)
        coefficients = A_lambda_over_A_V_dic
    else:
        coefficients = A_lambda_over_EBP_RP_dic

    correction = []
    correction_err = []

    for band_lambda in bands_list:
        if band_lambda not in coefficients or f'{band_lambda}_err' not in coefficients:
            raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")

        coefficient = coefficients[band_lambda]
        coefficient_err = coefficients[f'{band_lambda}_err']

        if not np.isfinite(coefficient) or not np.isfinite(coefficient_err) or coefficient_err < 0:
            raise ValueError(f"Invalid extinction coefficients for band: {band_lambda}")

        if extinction_type == 'EBP_RP':
            correction_lambda = coefficient * extinction
            correction_err_lambda = np.sqrt((coefficient * extinction_err)**2 + (extinction * coefficient_err)**2)
        else:
            correction_lambda = coefficient * A_V
            correction_err_lambda = np.sqrt((coefficient * A_V_err)**2 + (A_V * coefficient_err)**2)

        if verbose:
            print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")

        correction.append(correction_lambda)
        correction_err.append(correction_err_lambda)

    correction = np.asarray(correction)
    correction_err = np.asarray(correction_err)

    obs_mags_model_extincted = obs_mags_model + correction
    mags_ext_err_model_extincted = np.sqrt(obs_err_mags_model**2 + correction_err**2)

    return obs_mags_model_extincted.flatten(), mags_ext_err_model_extincted.flatten()



# Querying catalogs:

def get_info_from_simbad(name):
    """
    Query SIMBAD for object identifiers and basic astronomical information.

    Args:
        name (str): Object name or identifier recognized by SIMBAD.

    Returns:
        tuple: A 10-element tuple containing, in order:
            1. HIP_ID (str)
            2. HD_ID (str)
            3. Gaia_DR3_ID (str)
            4. TYC_ID (str)
            5. TwoMass_ID (str)
            6. Other_Name (str)
            7. Spectral_Type (str)
            8. Object_Type (str)
            9. RA_deg (float)
            10. DEC_deg (float)

        Missing string values are returned as empty strings. Missing coordinates
        are returned as np.nan.

    Notes:
        Identifier values are kept as strings to avoid precision loss, especially
        for Gaia DR3 source IDs. Query failures return available default values.
    """
    import re

    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad

    # Initialize output values.
    hip_id = hd_id = gaia_dr3_id = tyc_id = twomass_id = other_id = ""
    sp_type = otype = ""
    ra_deg = dec_deg = np.nan

    def _norm(value):
        """Normalize a SIMBAD value to a stripped string, or None if missing."""
        if value is None:
            return None

        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")

        value = re.sub(r"\s+", " ", str(value)).strip()
        return None if value.lower() in ("", "nan", "none", "--") else value

    def _out_str(value):
        """Return a normalized string, using an empty string for missing values."""
        value = _norm(value)
        return value if value is not None else ""

    def _result():
        """Return the result tuple in the required order."""
        return (
            _out_str(hip_id),
            _out_str(hd_id),
            _out_str(gaia_dr3_id),
            _out_str(tyc_id),
            _out_str(twomass_id),
            _out_str(other_id),
            _out_str(sp_type),
            _out_str(otype),
            ra_deg,
            dec_deg,
        )

    if not isinstance(name, str) or not name.strip():
        return _result()

    # 1. Query the main identifier, coordinates, spectral type, and object type.
    main_id = None

    try:
        sim = Simbad()
        sim.add_votable_fields("coordinates", "otype", "sp_type")
        query = sim.query_object(name)

        if query is not None and len(query) > 0:
            row = query[0]
            columns = query.colnames

            if "MAIN_ID" in columns:
                main_id = _norm(row["MAIN_ID"])
            else:
                fallback = Simbad.query_object(name)
                if fallback is not None and len(fallback) > 0:
                    if "MAIN_ID" in fallback.colnames:
                        main_id = _norm(fallback[0]["MAIN_ID"])

            if "ra" in columns and "dec" in columns:
                ra_value = _norm(row["ra"])
                dec_value = _norm(row["dec"])

                if ra_value is not None and dec_value is not None:
                    coord = SkyCoord(
                        ra_value, dec_value, unit=(u.deg, u.deg), frame="icrs"
                    )
                    ra_deg = float(coord.ra.deg)
                    dec_deg = float(coord.dec.deg)

            if "sp_type" in columns:
                sp_type = _out_str(row["sp_type"])
            if "otype" in columns:
                otype = _out_str(row["otype"])

    except Exception:
        # Preserve any values already retrieved if this query fails.
        pass

    # 2. Query all cross-identifiers.
    try:
        ids = Simbad.query_objectids(name)
    except Exception:
        ids = None

    if ids is None or len(ids) == 0:
        if main_id and not main_id.startswith(("HIP ", "HD ", "Gaia")):
            other_id = _out_str(main_id)
        return _result()

    if "ID" in ids.colnames:
        id_column = "ID"
    elif "id" in ids.colnames:
        id_column = "id"
    else:
        id_column = ids.colnames[0]

    other_candidates = []

    # 3. Parse cross-identifiers.
    for value in ids[id_column]:
        identifier = _norm(value)
        if not identifier:
            continue

        if identifier.startswith("HIP "):
            hip_id = identifier.split(" ", 1)[1].strip()
            continue

        if identifier.startswith("HD "):
            hd_id = identifier.split(" ", 1)[1].strip()
            continue

        if identifier.startswith("Gaia DR3"):
            try:
                # Keep the Gaia source ID as a string to preserve its exact value.
                gaia_dr3_id = str(int(identifier.split()[-1]))
            except (ValueError, IndexError):
                pass
            continue

        if identifier.startswith("TYC "):
            tyc_id = _out_str(identifier.split(" ", 1)[1])
            other_candidates.append(f"TYC {tyc_id}")
            continue

        if identifier.startswith("2MASS "):
            twomass_id = _out_str(identifier.split(" ", 1)[1])
            other_candidates.append(f"2MASS {twomass_id}")
            continue

        if identifier.startswith("Gaia "):
            continue

        other_candidates.append(identifier)

    # 4. Select Other_Name, preferring the main identifier when appropriate.
    if main_id and not main_id.startswith(("HIP ", "HD ", "Gaia ")):
        other_id = _out_str(main_id)

    elif other_candidates:
        filtered = [
            identifier
            for identifier in other_candidates
            if not identifier.startswith(("TYC ", "2MASS "))
        ]

        prefix_priority = ("V* ", "HR ", "BD ", "CD ", "CPD ", "GJ ")
        other_id = next(
            (
                identifier
                for prefix in prefix_priority
                for identifier in filtered
                if identifier.startswith(prefix)
            ),
            filtered[0] if filtered else other_candidates[0],
        )

    return _result()



def query_tycho_2(tyc_id):
    """
    Query Tycho-2 photometry and uncertainties using a TYC identifier.

    Args:
        tyc_id (str): Tycho identifier, e.g. "TYC 1395-2605-1" or "1395-2605-1".

    Returns:
        tuple: (bt, vt, e_bt, e_vt), all floats. Missing or invalid values are
            returned as np.nan. Catalogs are queried in priority order, and the
            first catalog containing a matching row is used.
    """
    import re

    from astroquery.vizier import Vizier

    def _safe_float(value):
        """Convert a value to a finite float, returning np.nan if invalid."""
        if value is None:
            return np.nan
        try:
            result = float(str(value).strip())
            return result if np.isfinite(result) else np.nan
        except (TypeError, ValueError):
            return np.nan

    # Parse the TYC identifier.
    if tyc_id is None:
        return (np.nan,) * 4

    identifier = re.sub(r"^\s*TYC\s+", "", str(tyc_id), flags=re.IGNORECASE).strip()
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", identifier)
    if not match:
        return (np.nan,) * 4

    tyc1, tyc2, tyc3 = map(int, match.groups())

    # Query catalogs in priority order.
    catalogs = ("I/259/tyc2", "I/259/suppl_1", "I/259/suppl_2")
    columns = ["TYC1", "TYC2", "TYC3", "BTmag", "VTmag", "e_BTmag", "e_VTmag"]
    vizier = Vizier(columns=columns, row_limit=row_limit)

    for catalog in catalogs:
        try:
            result = vizier.query_constraints(
                catalog=catalog, TYC1=tyc1, TYC2=tyc2, TYC3=tyc3
            )
            if result is None or len(result) == 0 or len(result[0]) == 0:
                continue

            table = result[0]
            row = table[0]
            values = []
            for column in ("BTmag", "VTmag", "e_BTmag", "e_VTmag"):
                values.append(_safe_float(row[column]) if column in table.colnames else np.nan)

            return tuple(values)

        except Exception:
            # Try the next catalog if the query fails.
            continue

    return (np.nan,) * 4



def _norm_2mass_id(twomass_id: str):
    """
    Normalize a 2MASS identifier with and without the leading J.

    Args:
        twomass_id (str): A 2MASS identifier, optionally prefixed with "2MASS".

    Returns:
        tuple: (id_with_j, id_without_j). Returns ("", "") for missing input.
    """
    import re

    if twomass_id is None:
        return "", ""

    s = str(twomass_id).strip()
    if s.lower() in ("", "nan", "none", "--"):
        return "", ""

    s = re.sub(r"^2MASS\s+", "", s, flags=re.IGNORECASE).strip()
    s = s[1:] if s.upper().startswith("J") else s
    return "J" + s, s


def query_2mass(twomass_id):
    """
    Query 2MASS photometry, uncertainties, and quality flags by identifier.

    Args:
        twomass_id (str): A 2MASS identifier, with or without the leading J.

    Returns:
        tuple: (J, H, Ks, e_J, e_H, e_Ks, Qflg, Rflg, Bflg, Cflg, Xflg, Aflg).
            Missing numeric values are np.nan; missing flag values are empty strings.
            Returns 12 missing values if no matching record is found.
    """
    from astroquery.vizier import Vizier

    def _safe_float(value):
        """Convert a value to a finite float, returning np.nan if invalid."""
        if value is None:
            return np.nan
        try:
            result = float(str(value).strip())
            return result if np.isfinite(result) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _safe_str(value):
        """Normalize a string value, returning an empty string if missing."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        s = str(value).strip()
        return "" if s.lower() in ("", "--", "nan", "none") else s

    id_with_j, id_without_j = _norm_2mass_id(twomass_id)
    if not id_without_j:
        return (np.nan,) * 6 + ("",) * 6

    catalog = "II/246/out"
    columns = [
        "RAJ2000", "DEJ2000", "2MASS", "Jmag", "e_Jmag", "Hmag", "e_Hmag",
        "Kmag", "e_Kmag", "Qflg", "Rflg", "Bflg", "Cflg", "Xflg", "Aflg",
    ]
    vizier = Vizier(columns=columns, row_limit=row_limit)

    for identifier in (id_without_j, id_with_j):
        try:
            result = vizier.query_constraints(catalog=catalog, **{"2MASS": identifier})
            if result is None or len(result) == 0 or len(result[0]) == 0:
                continue

            table = result[0]
            row = table[0]

            def _get(column, converter):
                """Retrieve and convert a column value, handling absent columns."""
                return converter(row[column]) if column in table.colnames else converter(None)

            return (
                _get("Jmag", _safe_float),
                _get("Hmag", _safe_float),
                _get("Kmag", _safe_float),
                _get("e_Jmag", _safe_float),
                _get("e_Hmag", _safe_float),
                _get("e_Kmag", _safe_float),
                _get("Qflg", _safe_str),
                _get("Rflg", _safe_str),
                _get("Bflg", _safe_str),
                _get("Cflg", _safe_str),
                _get("Xflg", _safe_str),
                _get("Aflg", _safe_str),
            )

        except Exception:
            # Try the next identifier format if the query fails.
            continue

    return (np.nan,) * 6 + ("",) * 6


def assess_2mass_phot_quality(
    J_mag, H_mag, Ks_mag, J_err_mag, H_err_mag, Ks_err_mag,
    Q_2mass, R_2mass, B_2mass, C_2mass, X_2mass, A_2mass,
    *, max_err_warn=0.05, max_err_bad=0.10, strict=True, verbose=True
):
    """
    Assess 2MASS photometric quality and report potential issues.

    Args:
        J_mag (float): J-band magnitude.
        H_mag (float): H-band magnitude.
        Ks_mag (float): Ks-band magnitude.
        J_err_mag (float): J-band magnitude uncertainty.
        H_err_mag (float): H-band magnitude uncertainty.
        Ks_err_mag (float): Ks-band magnitude uncertainty.
        Q_2mass (str): 2MASS photometric quality flag.
        R_2mass (str): 2MASS read flag.
        B_2mass (str): 2MASS blend flag.
        C_2mass (str): 2MASS contamination flag.
        X_2mass (str): 2MASS cross-talk flag.
        A_2mass (str): 2MASS artifact/association flag.
        max_err_warn (float): Uncertainty threshold for a warning.
        max_err_bad (float): Uncertainty threshold for rejection.
        strict (bool): Whether to enforce stricter quality criteria.
        verbose (bool): Whether to print generated warnings.

    Returns:
        dict: Contains an overall "ok" flag, a list of "warnings", and per-band
            results under "per_band", including each band's "ok" status and issues.

    Raises:
        ValueError: If the uncertainty thresholds are negative or incorrectly ordered.
    """
    if not (0 <= max_err_warn <= max_err_bad):
        raise ValueError("Require 0 <= max_err_warn <= max_err_bad.")

    def _isfinite(value):
        """Return whether a value can be interpreted as a finite number."""
        try:
            return value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _to_str(value):
        """Convert a flag value to a normalized string."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        s = str(value).strip()
        return "" if s.lower() in ("", "nan", "none", "--") else s

    def _safe_char_at(value, index):
        """Return a flag character, or None if the requested character is missing."""
        s = _to_str(value)
        return s[index] if len(s) > index else None

    def _nonzero_flag(value):
        """Return whether a global flag indicates a nonzero or nonstandard value."""
        s = _to_str(value)
        if not s:
            return False
        try:
            return float(s) != 0
        except ValueError:
            return s != "0"

    C, X, A = _to_str(C_2mass), _to_str(X_2mass), _to_str(A_2mass)
    contam, xtalk, assoc = _nonzero_flag(C), _nonzero_flag(X), _nonzero_flag(A)
    warnings = []

    if contam:
        warnings.append(f"Warning: 2MASS Cflg={C} indicates possible contamination or artifacts.")
    if xtalk:
        warnings.append(f"Warning: 2MASS Xflg={X} indicates possible cross-talk contamination.")
    if assoc:
        warnings.append(f"Warning: 2MASS Aflg={A} indicates a non-standard source association.")

    def _eval_band(mag, emag, q, r, b):
        """Evaluate one band's photometry and return its status and issues."""
        issues = []
        ok = True

        if not _isfinite(mag):
            issues.append("missing or invalid magnitude")
            ok = False

        if not _isfinite(emag):
            issues.append("missing or invalid uncertainty")
            ok = False
        elif emag >= max_err_bad:
            issues.append(f"large magnitude uncertainty ({emag:.3f} >= {max_err_bad})")
            ok = False
        elif emag >= max_err_warn:
            issues.append(f"elevated magnitude uncertainty ({emag:.3f} >= {max_err_warn})")

        if q is None:
            issues.append("missing Qflg")
            ok = False
        elif q in ("U", "E", "F", "X"):
            issues.append(f"Qflg={q} indicates non-detection or unusable photometry")
            ok = False
        elif q == "D":
            issues.append("Qflg=D indicates low-quality photometry")
            if strict:
                ok = False
        elif strict and q != "A":
            issues.append(f"Qflg={q} (strict mode requires A)")
            ok = False
        elif not strict and q not in ("A", "B"):
            issues.append(f"Qflg={q} (expected A or B)")
            ok = False

        if r is None:
            issues.append("missing Rflg")
            ok = False
        elif r not in ("1", "2", "3"):
            issues.append(f"Rflg={r} indicates non-standard or low-quality photometry")
            ok = False

        if b is None:
            issues.append("missing Bflg")
            ok = False
        else:
            try:
                b_int = int(b)
                if b_int == 0:
                    issues.append("Bflg=0 indicates an unreliable detection or inconsistent deblending")
                    ok = False
                elif b_int > 1:
                    issues.append(f"Bflg={b} indicates blending ({b} components fitted)")
                    if strict:
                        ok = False
            except (TypeError, ValueError):
                issues.append(f"Bflg={b} is not a standard integer value")
                ok = False

        if strict and (contam or xtalk or assoc):
            issues.append("global C/X/A flags indicate potential issues")
            ok = False

        return ok, issues

    flags = (
        [_safe_char_at(Q_2mass, i) for i in range(3)],
        [_safe_char_at(R_2mass, i) for i in range(3)],
        [_safe_char_at(B_2mass, i) for i in range(3)],
    )
    bands = {
        "J": (J_mag, J_err_mag, flags[0][0], flags[1][0], flags[2][0]),
        "H": (H_mag, H_err_mag, flags[0][1], flags[1][1], flags[2][1]),
        "Ks": (Ks_mag, Ks_err_mag, flags[0][2], flags[1][2], flags[2][2]),
    }

    per_band = {}
    for band, values in bands.items():
        ok, issues = _eval_band(*values)
        per_band[band] = {"ok": ok, "issues": issues}
        if issues:
            warnings.append(f"Warning: 2MASS {band}-band issues: {'; '.join(issues)}.")

    overall_ok = all(info["ok"] for info in per_band.values())
    result = {"ok": overall_ok, "warnings": warnings, "per_band": per_band}

    if verbose:
        for warning in warnings:
            print(warning)

    return result



def query_gaia_dr3_vizier(gaia_dr3_id, verbose=True, snr_lower_limit=10, n_lower_limit=5):
    """
    Query Gaia DR3 data from VizieR, with a Gaia Archive fallback for essential missing data.

    Args:
        gaia_dr3_id: Gaia DR3 source identifier.
        verbose: Whether to print query warnings and diagnostic messages.
        snr_lower_limit: Threshold below which photometric flux-over-error warnings are issued.
        n_lower_limit: Threshold below which photometric observation-count warnings are issued.

    Returns:
        A dictionary containing the source ID, photometry and flux-based uncertainties, astrometry,
        rotational broadening, binary probability, quality flags, warnings, raw catalog data, and
        optional E(BP-RP) extinction values. Missing or unavailable values are represented by NaN.
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

    def _table_value(row, name, default=np.nan):
        try:
            return row[name]
        except Exception:
            return default

    def _query_gaia_archive(sid_value):
        try:
            from astroquery.gaia import Gaia
            query = (
                "SELECT source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, "
                "phot_g_mean_flux, phot_bp_mean_flux, phot_rp_mean_flux, "
                "phot_g_mean_flux_error, phot_bp_mean_flux_error, phot_rp_mean_flux_error, "
                "parallax, parallax_error, ruwe, ra, dec, pmra, pmdec, pmra_error, pmdec_error, "
                "radial_velocity, radial_velocity_error, "
                "vbroad, vbroad_error "
                f"FROM gaiadr3.gaia_source WHERE source_id={sid_value}"
            )
            job = Gaia.launch_job(query)
            tab = job.get_results()
            if len(tab) == 0:
                return None
            return tab[0]
        except Exception as e:
            warnings.append(f"Gaia Archive fallback failed: {type(e).__name__}: {e}")
            return None

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
            "EBP_RP_lower": _nan(),
            "EBP_RP_upper": _nan(),
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
    EBP_RP_lower = _nan()
    EBP_RP_upper = _nan()

    # -----------------------------
    # 1) I/355 main
    # -----------------------------
    v355_cols = [
        "Source", "source_id",

        # coordinates
        "RA_ICRS", "DE_ICRS", "RAJ2000", "DEJ2000",
        "ra", "dec",

        # proper motions
        "pmRA", "e_pmRA", "pmDE", "e_pmDE",
        "pmra", "pmra_error", "pmdec", "pmdec_error",

        # radial velocity
        "RV", "e_RV",
        "radial_velocity", "radial_velocity_error",

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
        if verbose:
            if res355 is None or len(res355) == 0:
                print(f"VizieR I/355 returned no matches for source_id={sid}")
            # else:
            #     print(res355)

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
            eG = np.nan if (not np.isfinite(g_flux) or not np.isfinite(
                g_flux_err) or g_flux <= 0 or g_flux_err < 0) else MAG_ERR_FACTOR * (g_flux_err / g_flux)
            eBP = np.nan if (not np.isfinite(bp_flux) or not np.isfinite(
                bp_flux_err) or bp_flux <= 0 or bp_flux_err < 0) else MAG_ERR_FACTOR * (bp_flux_err / bp_flux)
            eRP = np.nan if (not np.isfinite(rp_flux) or not np.isfinite(
                rp_flux_err) or rp_flux <= 0 or rp_flux_err < 0) else MAG_ERR_FACTOR * (rp_flux_err / rp_flux)

            phot["e_G"] = eG
            phot["e_BP"] = eBP
            phot["e_RP"] = eRP

            # ---- coordinates ----
            astro["ra"] = _safe_float(_pick_first(row, cols, ["RA_ICRS", "RAJ2000", "ra"]))
            astro["dec"] = _safe_float(_pick_first(row, cols, ["DE_ICRS", "DEJ2000", "dec"]))

            # ---- proper motions ----
            astro["pmra"] = _safe_float(_pick_first(row, cols, ["pmRA", "pmra"]))
            astro["pmdec"] = _safe_float(_pick_first(row, cols, ["pmDE", "pmdec"]))
            astro["pmra_error"] = _safe_float(_pick_first(row, cols, ["e_pmRA", "pmra_error"]))
            astro["pmdec_error"] = _safe_float(_pick_first(row, cols, ["e_pmDE", "pmdec_error"]))

            # ---- parallax ----
            astro["parallax"] = _safe_float(_pick_first(row, cols, ["Plx", "parallax"]))
            astro["e_parallax"] = _safe_float(_pick_first(row, cols, ["e_Plx", "parallax_error"]))

            # ---- radial velocity ----
            astro["rv"] = _safe_float(_pick_first(row, cols, ["RV", "radial_velocity"]))
            astro["rv_error"] = _safe_float(_pick_first(row, cols, ["e_RV", "radial_velocity_error"]))

            # ---- RUWE ----
            astro["RUWE"] = _safe_float(_pick_first(row, cols, ["RUWE", "ruwe"]))

            # ---- vbroad + uncertainty ----
            rot["Vbroad"] = _safe_float(_pick_first(row, cols, ["Vbroad"]))
            rot["e_Vbroad"] = _safe_float(_pick_first(row, cols, ["e_Vbroad", "Vbroad_error"]))

            # ---- optional E(BP-RP) ----
            # different mirrors may name it E(BP-RP) or E(BP/RP)
            EBP_RP = _safe_float(_pick_first(row, cols, ["E(BP-RP)", "E(BP/RP)"]))
            EBP_RP_lower = _safe_float(_pick_first(row, cols, ["b_E(BP-RP)", "b_E(BP/RP)"]))
            EBP_RP_upper = _safe_float(_pick_first(row, cols, ["B_E(BP-RP)", "B_E(BP/RP)"]))

        else:
            warnings.append(f"I/355: No match for source_id={sid} (VizieR).")

    except Exception as e:
        warnings.append(f"I/355 query failed: {type(e).__name__}: {e}")

    # fallback to direct Gaia Archive if VizieR could not resolve the source
    if not np.isfinite(phot.get("G", np.nan)) or not np.isfinite(phot.get("BP", np.nan)) or not np.isfinite(phot.get("RP", np.nan)) or not np.isfinite(astro.get("parallax", np.nan)):
        gaia_row = _query_gaia_archive(sid)
        if gaia_row is not None:
            phot["G"] = _safe_float(_table_value(gaia_row, "phot_g_mean_mag"))
            phot["BP"] = _safe_float(_table_value(gaia_row, "phot_bp_mean_mag"))
            phot["RP"] = _safe_float(_table_value(gaia_row, "phot_rp_mean_mag"))

            g_flux = _safe_float(_table_value(gaia_row, "phot_g_mean_flux"))
            g_flux_err = _safe_float(_table_value(gaia_row, "phot_g_mean_flux_error"))
            bp_flux = _safe_float(_table_value(gaia_row, "phot_bp_mean_flux"))
            bp_flux_err = _safe_float(_table_value(gaia_row, "phot_bp_mean_flux_error"))
            rp_flux = _safe_float(_table_value(gaia_row, "phot_rp_mean_flux"))
            rp_flux_err = _safe_float(_table_value(gaia_row, "phot_rp_mean_flux_error"))

            g_flux_over_err = _safe_float(_table_value(gaia_row, "phot_g_mean_flux_over_error", np.nan))
            bp_flux_over_err = _safe_float(_table_value(gaia_row, "phot_bp_mean_flux_over_error", np.nan))
            rp_flux_over_err = _safe_float(_table_value(gaia_row, "phot_rp_mean_flux_over_error", np.nan))
            if not np.isfinite(g_flux_over_err):
                g_flux_over_err = _safe_div(g_flux, g_flux_err)
            if not np.isfinite(bp_flux_over_err):
                bp_flux_over_err = _safe_div(bp_flux, bp_flux_err)
            if not np.isfinite(rp_flux_over_err):
                rp_flux_over_err = _safe_div(rp_flux, rp_flux_err)

            phot["e_G"] = MAG_ERR_FACTOR * (g_flux_err / g_flux) if np.isfinite(g_flux) and np.isfinite(
                g_flux_err) and g_flux > 0 and g_flux_err >= 0 else np.nan
            phot["e_BP"] = MAG_ERR_FACTOR * (bp_flux_err / bp_flux) if np.isfinite(
                bp_flux) and np.isfinite(bp_flux_err) and bp_flux > 0 and bp_flux_err >= 0 else np.nan
            phot["e_RP"] = MAG_ERR_FACTOR * (rp_flux_err / rp_flux) if np.isfinite(
                rp_flux) and np.isfinite(rp_flux_err) and rp_flux > 0 and rp_flux_err >= 0 else np.nan

            astro["ra"] = _safe_float(_table_value(gaia_row, "ra"))
            astro["dec"] = _safe_float(_table_value(gaia_row, "dec"))
            astro["pmra"] = _safe_float(_table_value(gaia_row, "pmra"))
            astro["pmdec"] = _safe_float(_table_value(gaia_row, "pmdec"))
            astro["pmra_error"] = _safe_float(_table_value(gaia_row, "pmra_error"))
            astro["pmdec_error"] = _safe_float(_table_value(gaia_row, "pmdec_error"))
            astro["parallax"] = _safe_float(_table_value(gaia_row, "parallax"))
            astro["e_parallax"] = _safe_float(_table_value(gaia_row, "parallax_error"))
            astro["rv"] = _safe_float(_table_value(gaia_row, "radial_velocity"))
            astro["rv_error"] = _safe_float(_table_value(gaia_row, "radial_velocity_error"))
            astro["RUWE"] = _safe_float(_table_value(gaia_row, "ruwe"))

            raw["gaia_archive_fallback"] = {
                "source_id": sid,
                "query": "gaia_dr3",
            }
            
            # ---- vbroad + uncertainty ----
            rot["Vbroad"] = _safe_float(_table_value(gaia_row, "vbroad"))
            rot["e_Vbroad"] = _safe_float(_table_value(gaia_row, "vbroad_error"))

    # -----------------------------
    # 1.5) I/355/paramp: Pbin
    # -----------------------------
    try:
        vparamp = Vizier(columns=["**"], row_limit=row_limit)
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
        v357 = Vizier(columns=["**"], row_limit=row_limit)
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

    for catalog, table_name in [
        ("I/358/varisum", "varisum"),
        ("I/358/vclassre", "vclassre"),
    ]:
        try:
            vizier = Vizier(columns=["**"], row_limit=row_limit)
            result = vizier.query_constraints(catalog=catalog, Source=sid)
            if result is None or len(result) == 0 or len(result[0]) == 0:
                result = vizier.query_constraints(catalog=catalog, source_id=sid)

            if result is not None and len(result) > 0 and len(result[0]) > 0:
                is_variable = True
                tab = result[0]
                row = tab[0]
                raw[catalog] = {c: _safe_str(row[c]) for c in tab.colnames[:30]}
                var_tables.append(table_name)

        except Exception as e:
            warnings.append(f"{catalog} query failed: {type(e).__name__}: {e}")

    if is_variable:
        warnings.append(
            "Warning: source appears in Gaia DR3 variability tables ("
            + ", ".join(var_tables)
            + "). Treat as variable candidate."
        )

    # -----------------------------
    # 4) Holl+2023 spurious signals (J/A+A/674/A25)
    # -----------------------------
    is_spurious = False
    try:
        vA25 = Vizier(columns=["**"], row_limit=row_limit)
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
        "EBP_RP_lower": EBP_RP_lower,
        "EBP_RP_upper": EBP_RP_upper,
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
    total_width=None,
    col1=34,
    col2=45,
    col3=40,
):
    """
    Print a three-column aligned summary card containing general target information, Gaia DR3
    astrometry, photometry and flags, and Tycho-2 / 2MASS photometry.

    Args:
        target_name: Name of the target.
        simbad_info: SIMBAD identifiers and metadata, including coordinates and spectral type.
        gaia_res: Dictionary containing Gaia DR3 photometry, astrometry, rotation, flags, and
            extinction-related values.
        tycho_vals: Tycho-2 magnitudes and their uncertainties.
        mass_vals: 2MASS magnitudes, their uncertainties, and quality flags.
        total_width: Total width of the printed card. Defaults to the sum of the column widths
            plus separators.
        col1: Width of the general information column.
        col2: Width of the Gaia DR3 column.
        col3: Width of the Tycho-2 / 2MASS column.

    Returns:
        None. Prints the formatted summary card to standard output.
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

    def format_percent(x, w=7, p=2):
        """
        Format Gaia DR3 Pbin / P_binary.

        If 0 <= x <= 1, treat it as probability and convert to percent.
        If 1 < x <= 100, treat it as already percent.
        """
        if not isfin(x):
            return "nan"
        x = float(x)
        if 0.0 <= x <= 1.0:
            x = 100.0 * x
        return f"{x:{w}.{p}f}%"

    # --- unpack SIMBAD ---
    HIP, HD, GaiaID, TYC, TMASS, Other, SpT, ObjT, RA, DEC = simbad_info

    # --- Gaia ---
    phot  = gaia_res.get("photometry", {})
    astro = gaia_res.get("astrometry", {})
    rot   = gaia_res.get("rotation", {})
    flags = gaia_res.get("flags", {})

    EBP_RP    = gaia_res.get("EBP_RP", np.nan)
    EBP_RP_lo = gaia_res.get("EBP_RP_lower", np.nan)
    EBP_RP_hi = gaia_res.get("EBP_RP_upper", np.nan)

    # Compatible with both names
    P_binary = gaia_res.get("Pbin", gaia_res.get("P_binary", np.nan))

    plx  = astro.get("parallax", np.nan)
    eplx = astro.get("e_parallax", np.nan)

    # distance + uncertainty
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

    # --- width ---
    if total_width is None:
        total_width = col1 + col2 + col3 + 6

    W = total_width

    # -----------------------------
    # Middle column: ':' aligned
    # -----------------------------
    mid_labels = [
        "G", "G_BP", "G_RP", "Parallax", "RUWE",
        "Vbroad", "E(BP-RP)", "P_binary"
    ]
    labw = max(len(x) for x in mid_labels)

    def mid_kv(label, rhs):
        return f"{label:<{labw}} : {rhs}"

    mid_G = mid_kv(
        "G",
        f"{fnum(phot.get('G'),8,4)} ± {fnum(phot.get('e_G'),8,4)} mag"
    )
    mid_BP = mid_kv(
        "G_BP",
        f"{fnum(phot.get('BP'),8,4)} ± {fnum(phot.get('e_BP'),8,4)} mag"
    )
    mid_RP = mid_kv(
        "G_RP",
        f"{fnum(phot.get('RP'),8,4)} ± {fnum(phot.get('e_RP'),8,4)} mag"
    )
    mid_Plx = mid_kv(
        "Parallax",
        f"{fnum(plx,8,4)} ± {fnum(eplx,8,4)} mas"
    )
    mid_RUWE = mid_kv(
        "RUWE",
        f"{fnum(astro.get('RUWE'),8,4)}"
    )
    mid_Vb = mid_kv(
        "Vbroad",
        f"{fnum(rot.get('Vbroad'),8,4)} ± {fnum(rot.get('e_Vbroad'),8,2)} km/s"
    )

    if isfin(EBP_RP):
        if isfin(EBP_RP_lo):
            EBP_RP_lo_err = float(EBP_RP) - float(EBP_RP_lo)
        else:
            EBP_RP_lo_err = np.nan

        if isfin(EBP_RP_hi):
            EBP_RP_hi_err = float(EBP_RP_hi) - float(EBP_RP)
        else:
            EBP_RP_hi_err = np.nan

        mid_E_rhs = (
            f"{fnum(EBP_RP, 8, 4)} "
            f"(-{fnum(EBP_RP_lo_err, 6, 4)}, +{fnum(EBP_RP_hi_err, 6, 4)}) mag"
        )
    else:
        mid_E_rhs = "nan"

    mid_E = mid_kv("E(BP-RP)", mid_E_rhs)
    mid_Pbin = mid_kv("P_binary", format_percent(P_binary))

    # -----------------------------
    # Right column
    # -----------------------------
    right_BT = f"B_T : {fnum(BT,8,4)} ± {fnum(eBT,8,4)} mag"
    right_VT = f"V_T : {fnum(VT,8,4)} ± {fnum(eVT,8,4)} mag"
    right_sep = "-" * min(40, col3)
    right_2mass_header = "2MASS"
    right_J  = f"J   : {fnum(J,8,4)} ± {fnum(eJ,8,4)} mag"
    right_H  = f"H   : {fnum(H,8,4)} ± {fnum(eH,8,4)} mag"
    right_K  = f"Ks  : {fnum(Ks,8,4)} ± {fnum(eKs,8,4)} mag"

    # -----------------------------
    # Left column
    # -----------------------------
    left_RA   = f"RA            : {fnum(RA,8,4).strip()}"
    left_Dec  = f"Dec           : {fnum(DEC,8,4).strip()}"
    left_SpT  = f"Spectral Type : {fval(SpT):<10}"
    left_Obj  = f"Object Type   : {fval(ObjT):<10}"
    left_Dist = f"Distance      : {fnum(dist_pc,8,2).strip()} ± {fnum(dist_pc_err,8,2).strip()} pc"

    # -----------------------------
    # Print card
    # -----------------------------
    print("=" * W)
    print(f"Target : {target_name}")
    print(f"IDs    : Gaia DR3 {fval(GaiaID)} | HIP {fval(HIP)} | HD {fval(HD)} | 2MASS {fval(TMASS)}")
    print("-" * W)

    print(row3("General info", "Gaia DR3", "Tycho-2"))
    print(row3(left_RA,   mid_G,    right_BT))
    print(row3(left_Dec,  mid_BP,   right_VT))
    print(row3(left_SpT,  mid_RP,   right_sep))
    print(row3(left_Obj,  mid_Plx,  right_2mass_header))
    print(row3(left_Dist, mid_RUWE, right_J))
    print(row3("",        mid_Vb,   right_H))
    print(row3("",        mid_E,    right_K))
    print(row3("",        mid_Pbin, ""))

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
    Query SIMBAD, Tycho-2, 2MASS, and Gaia DR3 for a target.

    Identifier columns are always strings; missing identifiers are stored as "".
    Each query step retries up to max_retries times if critical fields are invalid.
    Failed queries do not interrupt the pipeline.

    Args:
        target_name: Target identifier.
        max_retries: Maximum attempts per query.
        return_df: Whether to return results as a DataFrame.
        verbose: Whether to print warnings and the summary.

    Returns:
        Query results as a tuple or a one-row DataFrame.
    """

    def _has_nan_like(x):
        """Check whether a value or nested container contains None or NaN."""
        if x is None:
            return True
        try:
            if np.isscalar(x):
                return bool(pd.isna(x))
        except Exception:
            pass
        if isinstance(x, dict):
            return any(_has_nan_like(v) for v in x.values())
        if isinstance(x, (list, tuple, np.ndarray)):
            return any(_has_nan_like(v) for v in x)
        return False

    def _retry_call(fn, fn_name, args=(), kwargs=None, ok_pred=None):
        """Retry a query until its result passes validation or attempts are exhausted."""
        kwargs = kwargs or {}
        last = None

        for i_try in range(1, max_retries + 1):
            try:
                last = fn(*args, **kwargs)
            except Exception as e:
                last = None
                if verbose:
                    print(f"Warning: {fn_name} raised {type(e).__name__}: {e}. Retry {i_try}/{max_retries}.")
                continue

            ok = last is not None and (
                bool(ok_pred(last)) if ok_pred is not None else not _has_nan_like(last)
            )
            if ok:
                return last

            if verbose:
                print(f"Warning: {fn_name} returned NaN fields. Retry {i_try}/{max_retries} for {target_name}")

        return last

    def _normalize_id(value):
        """Normalize an identifier to a string without converting it to a number."""
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        s = str(value).strip()
        return "" if s.lower() in ("", "nan", "none", "null", "--") else s

    # 1. SIMBAD
    def _ok_simbad(res):
        if not isinstance(res, (list, tuple)) or len(res) != 10:
            return False
        try:
            return np.isfinite(float(res[8])) and np.isfinite(float(res[9]))
        except (TypeError, ValueError):
            return False

    name_res = _retry_call(get_info_from_simbad, "SIMBAD(get_info_from_simbad)",
                           args=(target_name,), ok_pred=_ok_simbad)
    if name_res is None:
        raise RuntimeError("Simbad query returned None repeatedly.")

    (HIP_ID, HD_ID, Gaia_DR3_ID, TYC_ID, TwoMass_ID, Other_Name, Spectral_Type,
     Object_Type, RA, DEC) = name_res

    HIP_ID, HD_ID, Gaia_DR3_ID, TYC_ID, TwoMass_ID, Other_Name = map(
        _normalize_id, (HIP_ID, HD_ID, Gaia_DR3_ID, TYC_ID, TwoMass_ID, Other_Name)
    )
    Spectral_Type, Object_Type = map(_normalize_id, (Spectral_Type, Object_Type))

    # 2. Tycho-2
    def _ok_tycho(res):
        if not isinstance(res, (list, tuple)) or len(res) != 4:
            return False
        if TYC_ID == "":
            return True
        return np.isfinite(res[0]) and np.isfinite(res[1])

    result_tycho2 = _retry_call(query_tycho_2, "Tycho-2(query_tycho_2)",
                                args=(TYC_ID,), ok_pred=_ok_tycho)
    if result_tycho2 is None:
        result_tycho2 = (np.nan,) * 4
    B_T_mag, V_T_mag, B_T_err_mag, V_T_err_mag = result_tycho2

    # 3. 2MASS
    def _ok_2mass(res):
        if not isinstance(res, (list, tuple)) or len(res) != 12:
            return False
        if TwoMass_ID == "":
            return True
        return all(np.isfinite(res[i]) for i in range(3)) and not pd.isna(res[6])

    result_2mass = _retry_call(query_2mass, "2MASS(query_2mass)",
                               args=(TwoMass_ID,), ok_pred=_ok_2mass)
    if result_2mass is None:
        result_2mass = (np.nan,) * 12

    (J_mag, H_mag, Ks_mag, J_err_mag, H_err_mag, Ks_err_mag, Q_2mass, R_2mass,
     B_2mass, C_2mass, X_2mass, A_2mass) = result_2mass

    quality_2mass = assess_2mass_phot_quality(*result_2mass, strict=True, verbose=verbose)

    # 4. Gaia DR3
    def _ok_gaia(res):
        if not isinstance(res, dict):
            return False
        phot, astro = res.get("photometry", {}), res.get("astrometry", {})
        return all(np.isfinite(data.get(key, np.nan)) for data, key in (
            (phot, "G"), (phot, "BP"), (phot, "RP"), (astro, "parallax")
        ))

    gaia_dr3_res = _retry_call(query_gaia_dr3_vizier, "GaiaDR3(query_gaia_dr3_vizier)",
                               args=(Gaia_DR3_ID,), kwargs={"verbose": verbose}, ok_pred=_ok_gaia)

    if gaia_dr3_res is None:
        gaia_dr3_res = {
            "source_id": Gaia_DR3_ID,
            "photometry": dict.fromkeys(("G", "BP", "RP", "e_G", "e_BP", "e_RP"), np.nan),
            "astrometry": dict.fromkeys((
                "ra", "dec", "ra_error", "dec_error", "pmra", "pmdec", "pmra_error",
                "pmdec_error", "parallax", "e_parallax", "rv", "rv_error", "RUWE"
            ), np.nan),
            "rotation": {"Vbroad": np.nan, "e_Vbroad": np.nan},
            "Pbin": np.nan,
            "flags": {"is_nss": False, "is_variable": False, "is_spurious": False},
            "warnings": ["Gaia DR3 query failed repeatedly."],
            "raw": {},
        }

    phot = gaia_dr3_res["photometry"]
    astro = gaia_dr3_res["astrometry"]
    rotation = gaia_dr3_res["rotation"]

    G_mag, G_BP_mag, G_RP_mag = phot["G"], phot["BP"], phot["RP"]
    G_err_mag, G_BP_err_mag, G_RP_err_mag = phot["e_G"], phot["e_BP"], phot["e_RP"]

    gaia_ra, gaia_dec = astro.get("ra", RA), astro.get("dec", DEC)
    gaia_ra_error, gaia_dec_error = astro.get("ra_error", np.nan), astro.get("dec_error", np.nan)
    gaia_pmra, gaia_pmdec = astro.get("pmra", np.nan), astro.get("pmdec", np.nan)
    gaia_pmra_error, gaia_pmdec_error = astro.get("pmra_error", np.nan), astro.get("pmdec_error", np.nan)
    parallax_obs, parallax_obs_err = astro.get("parallax", np.nan), astro.get("e_parallax", np.nan)
    RUWE = astro.get("RUWE", np.nan)
    gaia_rv, gaia_rv_error = astro.get("rv", np.nan), astro.get("rv_error", np.nan)

    vsini_obs, vsini_obs_err = rotation["Vbroad"], rotation["e_Vbroad"]
    P_binary = gaia_dr3_res.get("Pbin", np.nan)
    gaia_flags = gaia_dr3_res.get("flags", {})
    gaia_warnings = gaia_dr3_res.get("warnings", [])
    gaia_raw = gaia_dr3_res.get("raw", {})

    EBP_RP_obs = gaia_dr3_res.get("EBP_RP", np.nan)
    EBP_RP_obs_lower = gaia_dr3_res.get("EBP_RP_lower", np.nan)
    EBP_RP_obs_upper = gaia_dr3_res.get("EBP_RP_upper", np.nan)
    EBP_RP_obs_lower_error = EBP_RP_obs_lower - EBP_RP_obs
    EBP_RP_obs_upper_error = EBP_RP_obs_upper - EBP_RP_obs

    # 5. Pack outputs
    obs_mags = np.array([G_mag, G_BP_mag, G_RP_mag, B_T_mag, V_T_mag,
                         J_mag, H_mag, Ks_mag], dtype=float)
    obs_err_mags = np.array([G_err_mag, G_BP_err_mag, G_RP_err_mag, B_T_err_mag,
                             V_T_err_mag, J_err_mag, H_err_mag, Ks_err_mag], dtype=float)

    additional_info = {
        "target_name": target_name,
        "HIP_ID": HIP_ID, "HD_ID": HD_ID, "Gaia_DR3_ID": Gaia_DR3_ID,
        "TYC_ID": TYC_ID, "TwoMass_ID": TwoMass_ID, "Other_Name": Other_Name,
        "Spectral_Type": Spectral_Type, "Object_Type": Object_Type, "RA": RA, "DEC": DEC,
        "Q_2mass": Q_2mass, "R_2mass": R_2mass, "B_2mass": B_2mass,
        "C_2mass": C_2mass, "X_2mass": X_2mass, "A_2mass": A_2mass,
        "2mass_ok": bool(quality_2mass["ok"]),
        "2mass_warnings": quality_2mass.get("warnings", []),
        "RUWE": RUWE, "P_binary": P_binary,
        "gaia_flags": gaia_flags, "gaia_warnings": gaia_warnings, "gaia_raw": gaia_raw,
        "EBP_RP": EBP_RP_obs,
        "EBP_RP_lower_error": EBP_RP_obs_lower_error,
        "EBP_RP_upper_error": EBP_RP_obs_upper_error,
    }

    if verbose:
        print_summary_card_aligned(target_name, name_res, gaia_dr3_res, result_tycho2, result_2mass)

    # 6. Return DataFrame
    if return_df:
        row = {
            "target_name": str(target_name),
            "HIP_ID": HIP_ID, "HD_ID": HD_ID, "Gaia_DR3_ID": Gaia_DR3_ID,
            "TYC_ID": TYC_ID, "TwoMass_ID": TwoMass_ID, "Other_Name": Other_Name,
            "Spectral_Type": Spectral_Type, "Object_Type": Object_Type, "RA": RA, "DEC": DEC,
            "G_mag": G_mag, "G_err": G_err_mag, "G_BP_mag": G_BP_mag, "G_BP_err": G_BP_err_mag,
            "G_RP_mag": G_RP_mag, "G_RP_err": G_RP_err_mag,
            "B_T_mag": B_T_mag, "B_T_err": B_T_err_mag, "V_T_mag": V_T_mag, "V_T_err": V_T_err_mag,
            "J_mag": J_mag, "J_err": J_err_mag, "H_mag": H_mag, "H_err": H_err_mag,
            "Ks_mag": Ks_mag, "Ks_err": Ks_err_mag,
            "parallax": parallax_obs, "parallax_err": parallax_obs_err,
            "vbroad": vsini_obs, "vbroad_err": vsini_obs_err, "RUWE": RUWE, "P_binary": P_binary,
            "EBP_RP": EBP_RP_obs,
            "EBP_RP_lower_error": EBP_RP_obs_lower_error,
            "EBP_RP_upper_error": EBP_RP_obs_upper_error,
            "Q_2mass": Q_2mass, "R_2mass": R_2mass, "B_2mass": B_2mass,
            "C_2mass": C_2mass, "X_2mass": X_2mass, "A_2mass": A_2mass,
            "2mass_ok": bool(quality_2mass["ok"]),
            "gaia_is_nss": bool(gaia_flags.get("is_nss", False)),
            "gaia_is_variable": bool(gaia_flags.get("is_variable", False)),
            "gaia_is_spurious": bool(gaia_flags.get("is_spurious", False)),
            "2mass_warnings": "; ".join(map(str, quality_2mass.get("warnings", []))),
            "gaia_warnings": "; ".join(map(str, gaia_warnings)),
        }

        df = pd.DataFrame([row])
        id_columns = ["HIP_ID", "HD_ID", "Gaia_DR3_ID", "TYC_ID", "TwoMass_ID", "Other_Name"]
        for col in id_columns:
            if col in df.columns:
                df[col] = df[col].map(_normalize_id).astype("string")

        return df

    return (obs_mags, obs_err_mags, vsini_obs, vsini_obs_err, parallax_obs, parallax_obs_err,
            EBP_RP_obs, EBP_RP_obs_lower, EBP_RP_obs_upper, additional_info)



# Following codes for P_risk calculation

def _print_section(title):
    """Print a section heading.

    Args:
        title: Text to display as the section heading.

    Returns:
        None.
    """
    print(f"\n## {title}\n")


def _print_pair(label1, value1, label2, value2, width=26):
    """Print two label-value pairs in aligned columns.

    Args:
        label1: Label for the first value.
        value1: First value to display.
        label2: Label for the second value.
        value2: Second value to display.
        width: Minimum width allocated to each label-value pair.

    Returns:
        None.
    """
    print(f"{label1:<18}{str(value1):>{width - 18}} | {label2:<18}{str(value2):>{width - 18}}")


def _get_sample_reliable_mini_ranges(samples_flat, param_names, boundary_grid):
    """Determine the reliable Mini range for each sample's MH and omega_ini values.

    Args:
        samples_flat: Two-dimensional array of flattened posterior samples.
        param_names: Parameter names corresponding to the sample columns.
        boundary_grid: Dictionary containing MH and omega_ini grids and their reliable Mini ranges.

    Returns:
        A tuple (low, high) of arrays containing the lower and upper reliable Mini bounds for
        each sample. Samples outside the grid or without valid coverage receive NaN bounds.
    """
    samples_flat = np.asarray(samples_flat, dtype=float)
    mh_idx, omega_idx = param_names.index("MH"), param_names.index("omega_ini")
    mh_grid, omega_grid = boundary_grid["mh_grid"], boundary_grid["omega_grid"]
    coverage = boundary_grid["coverage_lookup"]
    mh, omega = samples_flat[:, mh_idx], samples_flat[:, omega_idx]
    n = len(samples_flat)
    low, high = np.full(n, np.nan), np.full(n, np.nan)

    inside = (
        np.isfinite(mh) & np.isfinite(omega) & (mh >= mh_grid[0]) &
        (mh <= mh_grid[-1]) & (omega >= omega_grid[0]) & (omega <= omega_grid[-1])
    )
    if not np.any(inside):
        return low, high

    idx = np.flatnonzero(inside)
    mh, omega = mh[idx], omega[idx]
    brackets = []
    for values, grid in ((mh, mh_grid), (omega, omega_grid)):
        lo = np.clip(np.searchsorted(grid, values, side="right") - 1, 0, len(grid) - 1)
        hi = np.clip(np.searchsorted(grid, values, side="left"), 0, len(grid) - 1)
        brackets.extend((lo, hi))

    codes = np.column_stack(brackets)
    unique_codes, inverse = np.unique(codes, axis=0, return_inverse=True)

    for i, (m0, m1, o0, o1) in enumerate(unique_codes):
        keys = {
            (float(mh_grid[m]), float(omega_grid[o]))
            for m in {m0, m1} for o in {o0, o1}
        }
        if not keys.issubset(coverage):
            continue
        infos = [coverage[key] for key in keys]
        lo = max(info["reliable_min"] for info in infos)
        hi = min(info["reliable_max"] for info in infos)
        if hi < lo:
            continue
        target = idx[inverse == i]
        low[target], high[target] = lo, hi

    return low, high


def build_regular_critical_boundary_grid(
    critical_points_df,
    samples_flat,
    param_names,
    allow_posterior_mini_outside=False,
    verbose=True,
):
    """Build a regular critical-boundary grid and its interpolators.

    Args:
        critical_points_df: DataFrame containing MH, omega_ini, Mini, logTe, and logL critical points.
        samples_flat: Two-dimensional array of flattened posterior samples.
        param_names: Parameter names corresponding to the columns of samples_flat.
        allow_posterior_mini_outside: Whether to allow posterior Mini samples outside their reliable
            sample-specific boundary coverage.
        verbose: Whether to print a summary of the constructed boundary grid.

    Returns:
        A dictionary containing the regularized boundary grids, interpolators, reliable Mini coverage,
        extrapolation information, and posterior and boundary mass ranges.

    Raises:
        KeyError: If required columns or parameters are missing.
        ValueError: If posterior samples are invalid, the grid is insufficient, required grid
            combinations are missing, or posterior masses exceed reliable coverage.
    """
    GLOBAL_MASS_MIN, GLOBAL_MASS_MAX = 1.40, 5.00
    MASS_MARGIN, MASS_STEP = 0.30, 0.01
    MAX_LOW_MASS_EXTRAPOLATION = 0.10
    required_columns = ["MH", "omega_ini", "Mini", "logTe", "logL"]
    missing = [c for c in required_columns if c not in critical_points_df.columns]
    if missing:
        raise KeyError(f"Missing critical-point columns: {missing}")

    param_names = list(param_names)
    for name in ("Mini", "MH", "omega_ini"):
        if name not in param_names:
            raise KeyError(f"'{name}' not found in param_names.")

    samples_flat = np.asarray(samples_flat, dtype=float)
    mini_idx, mh_idx, omega_idx = (
        param_names.index(k) for k in ("Mini", "MH", "omega_ini")
    )
    sample_mini, sample_mh, sample_omega = (
        samples_flat[:, i] for i in (mini_idx, mh_idx, omega_idx)
    )
    finite_mini = sample_mini[np.isfinite(sample_mini)]
    finite_mh = sample_mh[np.isfinite(sample_mh)]
    finite_omega = sample_omega[np.isfinite(sample_omega)]
    for values, name in (
        (finite_mini, "Mini"),
        (finite_mh, "MH"),
        (finite_omega, "omega_ini"),
    ):
        if values.size == 0:
            raise ValueError(f"No finite posterior {name} samples.")

    critical_df = critical_points_df[required_columns].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    critical_df = critical_df[
        critical_df["Mini"].between(GLOBAL_MASS_MIN, GLOBAL_MASS_MAX)
    ]
    critical_df = critical_df.groupby(
        ["MH", "omega_ini", "Mini"], as_index=False
    )[["logTe", "logL"]].mean()
    all_mh = np.sort(critical_df["MH"].unique().astype(float))
    all_omega = np.sort(critical_df["omega_ini"].unique().astype(float))
    if len(all_mh) < 2 or len(all_omega) < 2:
        raise ValueError("At least two grid points are required for both MH and omega_ini.")

    def select_grid(grid, vmin, vmax):
        lo = max(np.searchsorted(grid, vmin, side="right") - 1, 0)
        hi = min(np.searchsorted(grid, vmax, side="left"), len(grid) - 1)
        selected = grid[lo:hi + 1]
        if len(selected) == 1:
            i = np.searchsorted(grid, selected[0])
            selected = (
                grid[:2] if i == 0
                else grid[-2:] if i == len(grid) - 1
                else grid[i:i + 2]
            )
        return selected

    mh_grid = select_grid(all_mh, finite_mh.min(), finite_mh.max())
    omega_grid = select_grid(all_omega, finite_omega.min(), finite_omega.max())
    if len(mh_grid) < 2 or len(omega_grid) < 2:
        raise ValueError("At least two posterior-relevant grid points are required per axis.")

    critical_df = critical_df[
        critical_df["MH"].isin(mh_grid) & critical_df["omega_ini"].isin(omega_grid)
    ].copy()
    coverage_lookup, failed_groups = {}, []

    for mh in mh_grid:
        for omega in omega_grid:
            group = critical_df[
                np.isclose(critical_df["MH"], mh, rtol=0, atol=1e-12) &
                np.isclose(critical_df["omega_ini"], omega, rtol=0, atol=1e-12)
            ]
            mini = np.sort(group["Mini"].unique())
            if len(mini) < 2:
                failed_groups.append({
                    "MH": mh,
                    "omega_ini": omega,
                    "reason": "missing combination" if len(mini) == 0
                    else "fewer than two mass points",
                })
                continue
            original_min, original_max = float(mini[0]), float(mini[-1])
            coverage_lookup[(float(mh), float(omega))] = {
                "original_min": original_min,
                "original_max": original_max,
                "reliable_min": max(GLOBAL_MASS_MIN, original_min - MAX_LOW_MASS_EXTRAPOLATION),
                "reliable_max": min(GLOBAL_MASS_MAX, original_max),
            }

    if failed_groups:
        raise ValueError(
            "Some posterior-relevant (MH, omega_ini) combinations are missing:\n" +
            pd.DataFrame(failed_groups).to_string(index=False)
        )

    coverage_info = {
        "mh_grid": mh_grid,
        "omega_grid": omega_grid,
        "coverage_lookup": coverage_lookup,
    }
    reliable_low, reliable_high = _get_sample_reliable_mini_ranges(
        samples_flat, param_names, coverage_info
    )
    mini_valid = (
        np.isfinite(sample_mini) & np.isfinite(reliable_low) &
        np.isfinite(reliable_high) & (sample_mini >= reliable_low) &
        (sample_mini <= reliable_high)
    )

    if not allow_posterior_mini_outside and not np.all(mini_valid):
        bad = np.flatnonzero(~mini_valid)
        i = bad[0]
        raise ValueError(
            "Posterior Mini extends outside its sample-specific reliable boundary coverage.\n"
            f"N invalid: {len(bad):,}; example Mini: {sample_mini[i]:.3f}; "
            f"reliable range: {reliable_low[i]:.3f} -- {reliable_high[i]:.3f}"
        )

    mass_min = max(GLOBAL_MASS_MIN, finite_mini.min() - MASS_MARGIN)
    mass_max = min(GLOBAL_MASS_MAX, finite_mini.max() + MASS_MARGIN)
    if mass_max <= mass_min:
        raise ValueError("No valid Mini range remains for the critical boundary.")
    mass_grid = mass_min + np.arange(
        int(np.floor((mass_max - mass_min) / MASS_STEP)) + 1
    ) * MASS_STEP
    if mass_max - mass_grid[-1] > 1e-10:
        mass_grid = np.append(mass_grid, mass_max)

    grid_shape = (len(mh_grid), len(omega_grid), len(mass_grid))
    logte_grid, logl_grid = np.full(grid_shape, np.nan), np.full(grid_shape, np.nan)
    regularized_rows, extrapolated_groups = [], []

    for im, mh in enumerate(mh_grid):
        for io, omega in enumerate(omega_grid):
            group = critical_df[
                np.isclose(critical_df["MH"], mh, rtol=0, atol=1e-12) &
                np.isclose(critical_df["omega_ini"], omega, rtol=0, atol=1e-12)
            ]
            mini = group["Mini"].to_numpy(float)
            y = group[["logTe", "logL"]].to_numpy(float)
            mini, unique_idx = np.unique(mini, return_index=True)
            y = y[unique_idx]
            values = np.full((len(mass_grid), 2), np.nan)
            inside = (mass_grid >= mini[0]) & (mass_grid <= mini[-1])
            values[inside] = PchipInterpolator(
                mini, y, axis=0, extrapolate=False
            )(mass_grid[inside])

            below, above = mass_grid < mini[0], mass_grid > mini[-1]
            if np.any(below):
                slope = (y[1] - y[0]) / (mini[1] - mini[0])
                values[below] = y[0] + (mass_grid[below] - mini[0])[:, None] * slope
            if np.any(above):
                slope = (y[-1] - y[-2]) / (mini[-1] - mini[-2])
                values[above] = y[-1] + (mass_grid[above] - mini[-1])[:, None] * slope
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Non-finite regularized boundary at MH={mh:.5f}, omega={omega:.3f}."
                )

            logte_grid[im, io], logl_grid[im, io] = values[:, 0], values[:, 1]
            info = coverage_lookup[(float(mh), float(omega))]
            if info["reliable_min"] < info["original_min"]:
                extrapolated_groups.append({
                    "MH": float(mh),
                    "omega_ini": float(omega),
                    "original_min_mass": info["original_min"],
                    "extrapolated_to": info["reliable_min"],
                    "gap": info["original_min"] - info["reliable_min"],
                })
            regularized_rows.append(pd.DataFrame({
                "MH": mh,
                "omega_ini": omega,
                "Mini": mass_grid,
                "logTe": values[:, 0],
                "logL": values[:, 1],
            }))

    regularized_df = pd.concat(regularized_rows, ignore_index=True)
    extrapolated_df = pd.DataFrame(extrapolated_groups)
    axes = (mh_grid, omega_grid, mass_grid)
    boundary_values = np.stack((logte_grid, logl_grid), axis=-1)
    interp_boundary = RegularGridInterpolator(
        axes, boundary_values, method="linear", bounds_error=False, fill_value=np.nan
    )
    interp_logte = RegularGridInterpolator(
        axes, logte_grid, bounds_error=False, fill_value=np.nan
    )
    interp_logl = RegularGridInterpolator(
        axes, logl_grid, bounds_error=False, fill_value=np.nan
    )
    common_low = max(x["reliable_min"] for x in coverage_lookup.values())
    common_high = min(x["reliable_max"] for x in coverage_lookup.values())

    boundary_grid = {
        "mh_grid": mh_grid,
        "omega_grid": omega_grid,
        "mass_grid": mass_grid,
        "logte_grid": logte_grid,
        "logl_grid": logl_grid,
        "interp_boundary": interp_boundary,
        "interp_logte": interp_logte,
        "interp_logl": interp_logl,
        "regularized_df": regularized_df,
        "extrapolated_groups": extrapolated_df,
        "coverage_lookup": coverage_lookup,
        "posterior_mini_range": (float(finite_mini.min()), float(finite_mini.max())),
        "boundary_mini_range": (float(mass_grid[0]), float(mass_grid[-1])),
        "common_reliable_mini_range": (float(common_low), float(common_high)),
        "global_mini_range": (GLOBAL_MASS_MIN, GLOBAL_MASS_MAX),
    }

    if verbose:
        _print_section("Boundary grid")
        _print_pair(
            "MH", f"{mh_grid.min():.5f} -- {mh_grid.max():.5f}",
            "omega", f"{omega_grid.min():.3f} -- {omega_grid.max():.3f}"
        )
        _print_pair("MH grid", f"N={len(mh_grid)}", "omega grid", f"N={len(omega_grid)}")
        _print_pair(
            "Posterior Mini", f"{finite_mini.min():.3f} -- {finite_mini.max():.3f}",
            "Boundary Mini", f"{mass_grid[0]:.3f} -- {mass_grid[-1]:.3f}"
        )
        _print_pair(
            "Mass grid", f"N={len(mass_grid)}",
            "Low-M extrap.", f"N={len(extrapolated_df)}"
        )

    return boundary_grid


def classify_samples_in_batches(
    samples_flat,
    blobs_flat,
    param_names,
    output_list,
    boundary_grid,
    batch_size=5000,
    show_progress=True,
):
    """Classify posterior samples as safe or risky relative to the critical boundary.

    Args:
        samples_flat: Two-dimensional array of flattened posterior samples.
        blobs_flat: Two-dimensional array of derived quantities associated with each sample.
        param_names: Parameter names corresponding to the columns of samples_flat.
        output_list: Names corresponding to the columns of blobs_flat.
        boundary_grid: Dictionary containing the critical-boundary grids, interpolator, and reliable
            Mini coverage information.
        batch_size: Number of samples to process per batch.
        show_progress: Whether to display the classification progress bar.

    Returns:
        A tuple containing a diagnostic DataFrame for all samples and an integer label array,
        where 1 indicates safe and 0 indicates risky or invalid.

    Raises:
        ValueError: If the sample arrays have different row counts or batch_size is not a positive
            integer.
        ValueError: If required parameter or output names are missing.
        TypeError: If show_progress is not a boolean.
    """
    samples_flat, blobs_flat = np.asarray(samples_flat, float), np.asarray(blobs_flat, float)
    if len(samples_flat) != len(blobs_flat):
        raise ValueError("samples_flat and blobs_flat must have the same number of rows.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress must be bool.")

    param_names, output_list = list(param_names), list(output_list)
    mini_idx, mh_idx, omega_idx = (
        param_names.index(k) for k in ("Mini", "MH", "omega_ini")
    )
    logte_idx, logl_idx = output_list.index("logTe"), output_list.index("logL")
    mh_grid, omega_grid, mass_grid = (
        np.asarray(boundary_grid[k], float)
        for k in ("mh_grid", "omega_grid", "mass_grid")
    )
    interp_boundary = boundary_grid["interp_boundary"]
    n_samples, n_mass = len(samples_flat), len(mass_grid)
    all_mini, all_mh, all_omega = (
        samples_flat[:, i] for i in (mini_idx, mh_idx, omega_idx)
    )
    all_logte, all_logl = blobs_flat[:, logte_idx], blobs_flat[:, logl_idx]
    all_low, all_high = _get_sample_reliable_mini_ranges(
        samples_flat, param_names, boundary_grid
    )

    safe_labels = np.zeros(n_samples, dtype=np.int8)
    valid_results = np.zeros(n_samples, dtype=bool)
    signed_side_results = np.full(n_samples, np.nan)
    distance_results = np.full(n_samples, np.nan)
    nearest_mass_results = np.full(n_samples, np.nan)
    nearest_logte_results = np.full(n_samples, np.nan)
    nearest_logl_results = np.full(n_samples, np.nan)
    reason_results = np.full(n_samples, "invalid", dtype=object)

    with tqdm(
        total=n_samples,
        desc="Classifying samples",
        unit="samples",
        disable=not show_progress,
    ) as pbar:
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            idx = np.arange(start, stop)
            mini, mh, omega = all_mini[start:stop], all_mh[start:stop], all_omega[start:stop]
            px, py = all_logte[start:stop], all_logl[start:stop]
            low, high = all_low[start:stop], all_high[start:stop]
            finite = (
                np.isfinite(mini) & np.isfinite(mh) & np.isfinite(omega) &
                np.isfinite(px) & np.isfinite(py) & np.isfinite(low) & np.isfinite(high)
            )
            inside_grid = (
                (mh >= mh_grid[0]) & (mh <= mh_grid[-1]) &
                (omega >= omega_grid[0]) & (omega <= omega_grid[-1])
            )
            inside_mass = (mini >= low) & (mini <= high)
            usable = finite & inside_grid & inside_mass
            reason_results[idx[~finite]] = "non-finite value"
            reason_results[idx[finite & ~inside_grid]] = "MH or omega_ini outside boundary grid"
            reason_results[idx[finite & inside_grid & ~inside_mass]] = (
                "Mini outside sample-specific reliable boundary coverage"
            )
            if not np.any(usable):
                pbar.update(stop - start)
                continue

            idx, mh, omega, px, py, low, high = (
                a[usable] for a in (idx, mh, omega, px, py, low, high)
            )
            n = len(idx)
            points = np.empty((n, n_mass, 3))
            points[:, :, 0], points[:, :, 1], points[:, :, 2] = (
                mh[:, None], omega[:, None], mass_grid
            )
            boundary_xy = np.asarray(
                interp_boundary(points.reshape(-1, 3)), float
            ).reshape(n, n_mass, 2)
            x, y = boundary_xy[:, :, 0], boundary_xy[:, :, 1]
            valid_mass = (
                (mass_grid[None, :] >= low[:, None]) &
                (mass_grid[None, :] <= high[:, None]) &
                np.isfinite(x) & np.isfinite(y)
            )
            enough = valid_mass.sum(axis=1) >= 2
            reason_results[idx[~enough]] = "fewer than two reliable boundary points"
            if not np.any(enough):
                pbar.update(stop - start)
                continue

            idx, x, y, valid_mass, px, py = (
                a[enough] for a in (idx, x, y, valid_mass, px, py)
            )
            n = len(idx)
            xm, ym = np.where(valid_mass, x, np.nan), np.where(valid_mass, y, np.nan)
            xmin, xmax = np.nanmin(xm, axis=1), np.nanmax(xm, axis=1)
            ymin, ymax = np.nanmin(ym, axis=1), np.nanmax(ym, axis=1)
            xs, ys = xmax - xmin, ymax - ymin
            xs[(~np.isfinite(xs)) | (xs == 0)] = 1.0
            ys[(~np.isfinite(ys)) | (ys == 0)] = 1.0
            xc, yc = np.nanmean(xm, axis=1), np.nanmean(ym, axis=1)
            xn, yn = (x - xc[:, None]) / xs[:, None], (y - yc[:, None]) / ys[:, None]
            px, py = (px - xc) / xs, (py - yc) / ys
            sx, sy = xn[:, 1:] - xn[:, :-1], yn[:, 1:] - yn[:, :-1]
            valid_segment = valid_mass[:, :-1] & valid_mass[:, 1:]
            length2 = sx**2 + sy**2
            valid_segment &= np.isfinite(length2) & (length2 > 0)
            safe_length2 = np.where(valid_segment, length2, 1.0)
            dx, dy = px[:, None] - xn[:, :-1], py[:, None] - yn[:, :-1]
            fraction = np.clip(
                (dx * sx + dy * sy) / safe_length2, 0.0, 1.0
            )
            rx = px[:, None] - (xn[:, :-1] + fraction * sx)
            ry = py[:, None] - (yn[:, :-1] + fraction * sy)
            distance2 = rx**2 + ry**2
            distance2[~valid_segment] = np.inf
            nearest = np.argmin(distance2, axis=1)
            rows = np.arange(n)
            nearest_d2 = distance2[rows, nearest]
            has_segment = np.isfinite(nearest_d2)
            reason_results[idx[~has_segment]] = "no valid boundary segment"
            if not np.any(has_segment):
                pbar.update(stop - start)
                continue

            rows, idx, nearest = rows[has_segment], idx[has_segment], nearest[has_segment]
            fraction = fraction[rows, nearest]
            side = (
                sx[rows, nearest] * ry[rows, nearest] -
                sy[rows, nearest] * rx[rows, nearest]
            )
            is_safe = side < 0
            mass0, mass1 = mass_grid[nearest], mass_grid[nearest + 1]
            nearest_mass = mass0 + fraction * (mass1 - mass0)
            nearest_logte = x[rows, nearest] + fraction * (
                x[rows, nearest + 1] - x[rows, nearest]
            )
            nearest_logl = y[rows, nearest] + fraction * (
                y[rows, nearest + 1] - y[rows, nearest]
            )
            safe_labels[idx], valid_results[idx] = is_safe.astype(np.int8), True
            signed_side_results[idx] = side
            distance_results[idx] = np.sqrt(nearest_d2[has_segment])
            nearest_mass_results[idx], nearest_logte_results[idx] = nearest_mass, nearest_logte
            nearest_logl_results[idx] = nearest_logl
            reason_results[idx[is_safe]], reason_results[idx[~is_safe]] = "safe", "risky"
            pbar.update(stop - start)

    diagnostics_df = pd.DataFrame({
        "flat_index": np.arange(n_samples),
        "Mini": all_mini,
        "MH": all_mh,
        "omega_ini": all_omega,
        "reliable_Mini_min": all_low,
        "reliable_Mini_max": all_high,
        "logTe": all_logte,
        "logL": all_logl,
        "safe_label": safe_labels,
        "is_safe": valid_results & (safe_labels == 1),
        "valid": valid_results,
        "signed_side": signed_side_results,
        "boundary_distance": distance_results,
        "nearest_boundary_mass": nearest_mass_results,
        "nearest_boundary_logTe": nearest_logte_results,
        "nearest_boundary_logL": nearest_logl_results,
        "reason": reason_results,
    })
    return diagnostics_df, safe_labels


def classify_posterior_samples_safety(
    data,
    critical_points_df,
    batch_size=5000,
    ignore_out_of_range_mini=False,
    show_progress=True,
):
    """Classify posterior samples as safe or risky based on the critical boundary.

    Args:
        data: Loaded posterior data containing samples, blobs, parameter names, output names,
            discard count, and optionally true values for restoring fixed parameters.
        critical_points_df: DataFrame containing the critical boundary points.
        batch_size: Number of posterior samples processed per batch.
        ignore_out_of_range_mini: If True, remove samples outside the global or sample-specific
            reliable Mini range; otherwise, raise an error if any sample is outside these ranges.
        show_progress: Whether to display the classification progress bar.

    Returns:
        Dictionary containing filtered posterior samples and blobs, parameter names, safe labels,
        safe and risky samples, classification diagnostics, summary statistics, and the critical
        boundary grid.

    Raises:
        KeyError: If required posterior data keys or true values are missing.
        ValueError: If input shapes, parameter names, discard count, or Mini ranges are invalid,
            or if no samples remain after filtering.
        TypeError: If ignore_out_of_range_mini or show_progress is not a boolean.
    """
    GLOBAL_MINI_MIN, GLOBAL_MINI_MAX = 1.40, 5.00
    required = ["Mini", "Age", "MH", "omega_ini", "Inclination", "parallax"]
    required_keys = ["samples", "blobs", "param_names", "output_list", "n_discard"]
    missing = [k for k in required_keys if k not in data.files]
    if missing:
        raise KeyError(f"Missing keys in posterior data: {missing}")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(ignore_out_of_range_mini, (bool, np.bool_)):
        raise TypeError("ignore_out_of_range_mini must be bool.")
    if not isinstance(show_progress, (bool, np.bool_)):
        raise TypeError("show_progress must be bool.")

    samples = np.asarray(data["samples"], dtype=float)
    param_names = [str(x) for x in data["param_names"]]
    if samples.ndim != 3:
        raise ValueError(f"samples must have shape (steps, walkers, parameters). Got {samples.shape}.")
    if samples.shape[-1] != len(param_names):
        raise ValueError("samples dimension does not match param_names.")

    # Remove nuisance parameters and restore fixed parameters only when necessary.
    missing_params = [name for name in required if name not in param_names]
    nuisance = lambda name: name.startswith(("sigma", "extinction", "extinciton"))
    if missing_params or any(nuisance(name) for name in param_names):
        if missing_params:
            if "true_values" not in data.files:
                raise KeyError("Missing 'true_values' required to restore fixed parameters.")
            true_values = np.asarray(data["true_values"], dtype=object)
            if len(true_values) != len(required):
                raise ValueError(f"true_values must contain one value for each of {required}.")
        keep = [i for i, name in enumerate(param_names) if not nuisance(name)]
        samples, param_names = samples[:, :, keep], [param_names[i] for i in keep]
        for name in missing_params:
            value = true_values[required.index(name)]
            if value is None or pd.isna(value):
                raise ValueError(f"Missing parameter '{name}' has no valid fixed value in true_values.")
            i = required.index(name)
            samples = np.insert(samples, i, float(value), axis=2)
            param_names.insert(i, name)
        if param_names != required:
            raise ValueError(f"Unexpected parameter order after restoring fixed parameters: {param_names}")

    blobs = np.asarray(data["blobs"], dtype=float)
    output_list = [str(x) for x in data["output_list"]]
    n_discard = int(np.asarray(data["n_discard"]).item())
    if blobs.ndim != 3:
        raise ValueError(f"blobs must have shape (steps, walkers, outputs). Got {blobs.shape}.")
    if samples.shape[:2] != blobs.shape[:2]:
        raise ValueError("samples and blobs have inconsistent step/walker dimensions.")
    if blobs.shape[-1] != len(output_list):
        raise ValueError("blobs dimension does not match output_list.")
    if not 0 <= n_discard < samples.shape[0]:
        raise ValueError(f"Invalid n_discard={n_discard}.")

    mini_idx = param_names.index("Mini")
    samples_flat_original = samples[n_discard:].reshape(-1, samples.shape[-1])
    blobs_flat_original = blobs[n_discard:].reshape(-1, blobs.shape[-1])
    n_total_before_mini_cut = len(samples_flat_original)
    samples_flat, blobs_flat = samples_flat_original, blobs_flat_original

    mini = samples_flat[:, mini_idx]
    global_keep = (
        np.isfinite(mini) & (mini >= GLOBAL_MINI_MIN) & (mini <= GLOBAL_MINI_MAX)
    )
    n_removed_global = int(np.count_nonzero(~global_keep))
    if ignore_out_of_range_mini:
        samples_flat, blobs_flat = samples_flat[global_keep], blobs_flat[global_keep]
        if len(samples_flat) == 0:
            raise ValueError("No posterior samples remain after the global Mini cut.")
    elif not np.all(global_keep):
        raise ValueError(
            "Posterior Mini extends outside the global critical-boundary range "
            "1.40 <= Mini <= 5.00 Msun. \n"
            "You may use 'ignore_out_of_range_mini=True' to ignore samples outside the critical-boundary range."
        )

    preliminary = build_regular_critical_boundary_grid(
        critical_points_df,
        samples_flat,
        param_names,
        allow_posterior_mini_outside=True,
        verbose=False,
    )
    reliable_low, reliable_high = _get_sample_reliable_mini_ranges(
        samples_flat, param_names, preliminary
    )
    mini = samples_flat[:, mini_idx]
    boundary_keep = (
        np.isfinite(mini) & np.isfinite(reliable_low) & np.isfinite(reliable_high) &
        (mini >= reliable_low) & (mini <= reliable_high)
    )
    n_removed_boundary = int(np.count_nonzero(~boundary_keep))
    if ignore_out_of_range_mini:
        samples_flat, blobs_flat = samples_flat[boundary_keep], blobs_flat[boundary_keep]
        if len(samples_flat) == 0:
            raise ValueError("No posterior samples remain after the sample-specific reliable Mini cut.")
    elif not np.all(boundary_keep):
        bad = np.flatnonzero(~boundary_keep)
        i = bad[0]
        raise ValueError(
            "Posterior Mini extends outside its sample-specific reliable boundary coverage.\n"
            f"N invalid: {len(bad):,}; example Mini: {mini[i]:.3f}; "
            f"reliable range: {reliable_low[i]:.3f} -- {reliable_high[i]:.3f}"
        )

    n_removed_mini_cut = n_removed_global + n_removed_boundary
    p_removed_mini_cut = (
        n_removed_mini_cut / n_total_before_mini_cut if n_total_before_mini_cut else np.nan
    )
    boundary_grid = build_regular_critical_boundary_grid(
        critical_points_df,
        samples_flat,
        param_names,
        allow_posterior_mini_outside=False,
        verbose=False,
    )

    _print_section("Posterior")
    _print_pair("Samples", samples.shape, "Blobs", blobs.shape)
    _print_pair("After discard", f"{n_total_before_mini_cut:,}", "Batch size", f"{batch_size:,}")
    if ignore_out_of_range_mini:
        _print_pair(
            "Global Mini cut", f"{GLOBAL_MINI_MIN:.2f} -- {GLOBAL_MINI_MAX:.2f}",
            "Removed", f"{n_removed_global:,}"
        )
        _print_pair("Boundary cut", "sample-specific", "Removed", f"{n_removed_boundary:,}")
        _print_pair(
            "Removed frac.", f"{p_removed_mini_cut:.6f}",
            "Samples used", f"{len(samples_flat):,}"
        )

    _print_section("Boundary grid")
    mh_grid, omega_grid, mass_grid = (
        boundary_grid[k] for k in ("mh_grid", "omega_grid", "mass_grid")
    )
    _print_pair(
        "MH", f"{mh_grid.min():.5f} -- {mh_grid.max():.5f}",
        "omega", f"{omega_grid.min():.3f} -- {omega_grid.max():.3f}"
    )
    _print_pair("MH grid", f"N={len(mh_grid)}", "omega grid", f"N={len(omega_grid)}")
    retained_mini = samples_flat[:, mini_idx]
    _print_pair(
        "Posterior Mini", f"{retained_mini.min():.3f} -- {retained_mini.max():.3f}",
        "Boundary Mini", f"{mass_grid[0]:.3f} -- {mass_grid[-1]:.3f}"
    )
    _print_pair(
        "Mass grid", f"N={len(mass_grid)}",
        "Low-M extrap.", f"N={len(boundary_grid['extrapolated_groups'])}"
    )

    diagnostics_df, safe_labels = classify_samples_in_batches(
        samples_flat,
        blobs_flat,
        param_names,
        output_list,
        boundary_grid,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    valid = diagnostics_df["valid"].to_numpy(bool)
    safe_mask = valid & (safe_labels == 1)
    risky_mask = valid & (safe_labels == 0)
    samples_labeled = np.column_stack((samples_flat, safe_labels))
    n_total, n_valid = len(safe_labels), int(valid.sum())
    n_invalid, n_safe, n_risky = n_total - n_valid, int(safe_mask.sum()), int(risky_mask.sum())
    denominator = n_safe + n_risky
    p_safe = n_safe / denominator if denominator else np.nan
    p_risky = n_risky / denominator if denominator else np.nan
    valid_fraction = n_valid / n_total if n_total else np.nan

    summary = {
        "N_total_before_Mini_cut": int(n_total_before_mini_cut),
        "ignore_out_of_range_Mini": bool(ignore_out_of_range_mini),
        "Mini_min": float(GLOBAL_MINI_MIN),
        "Mini_max": float(GLOBAL_MINI_MAX),
        "N_removed_global_Mini_cut": n_removed_global,
        "N_removed_boundary_Mini_cut": n_removed_boundary,
        "N_removed_Mini_cut": n_removed_mini_cut,
        "P_removed_Mini_cut": float(p_removed_mini_cut),
        "N_total": n_total,
        "N_valid": n_valid,
        "N_invalid": n_invalid,
        "N_safe": n_safe,
        "N_risky": n_risky,
        "P_safe": float(p_safe),
        "P_risky": float(p_risky),
        "valid_fraction": float(valid_fraction),
        "n_discard": n_discard,
        "batch_size": int(batch_size),
        "boundary_Mini_min": float(mass_grid[0]),
        "boundary_Mini_max": float(mass_grid[-1]),
    }

    _print_section("Classification")
    _print_pair("Valid", f"{n_valid:,}", "Invalid", f"{n_invalid:,}")
    _print_pair("Safe", f"{n_safe:,}", "Risky", f"{n_risky:,}")
    _print_pair("P(safe)", f"{p_safe:.5f}", "P(risky)", f"{p_risky:.5f}")

    return {
        "samples": samples_flat,
        "blobs": blobs_flat,
        "param_names": param_names,
        "output_list": output_list,
        "samples_labeled": samples_labeled,
        "samples_labeled_columns": param_names + ["safe_label"],
        "safe_labels": safe_labels,
        "safe_samples": samples_flat[safe_mask],
        "risky_samples": samples_flat[risky_mask],
        "safe_blobs": blobs_flat[safe_mask],
        "risky_blobs": blobs_flat[risky_mask],
        "diagnostics": diagnostics_df,
        "summary": summary,
        "critical_boundary_grid": boundary_grid,
    }