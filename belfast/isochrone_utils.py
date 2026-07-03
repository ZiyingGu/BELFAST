
import astropy.units as u
import astropy.constants as ct
from datetime import datetime
import gzip
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d, RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from belfast.paths import DATA_DIR



def _open_text_maybe_gz(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def read_isochrones(iso_file, use_tqdm=True):
    """
    Reads an isochrone file (.dat or .dat.gz) and parses it into a list of pandas DataFrames.
    """

    with _open_text_maybe_gz(iso_file) as f:
        total_lines = sum(1 for _ in f)

    results = []
    i = 0
    data_list = []
    columns = None

    with _open_text_maybe_gz(iso_file) as f:

        if use_tqdm:
            iterable = tqdm(f, desc="Reading lines", total=total_lines)
        else:
            iterable = f

        for line in iterable:
            if "isochrone terminated" in line:
                if columns is not None and data_list:
                    df = pd.DataFrame(data_list, columns=columns)
                    df = df.astype("float64")
                    results.append(df)
                break

            if "#" in line and "Zini" not in line:
                continue

            elif "Zini" in line:
                if i != 0 and columns is not None:
                    df = pd.DataFrame(data_list, columns=columns)
                    df = df.astype("float64")
                    results.append(df)

                i += 1
                data_list = []
                columns = re.split(r"\s+", line.strip("#").strip())

            else:
                if columns is None:
                    continue

                data_row = re.split(r"\s+", line.strip())
                if len(data_row) == len(columns):
                    data_list.append(data_row)

        if use_tqdm:
            print("The number of isochrones:", i)

    return results


def rearrange_isochrones(iso_paths_list, selected_bands_list, saved_folder_dir, name_suffix=None, n_common=35):
    """
    Rearrange and merge multiple sets of isochrone files into a unified format.

    This function takes several series of isochrone files (from different band
    configurations), verifies that their first `n_common` columns are identical,
    and then merges their band-specific columns into a single output file for
    each isochrone step. The output files are saved to the specified directory,
    with optional suffixes and metadata headers.

    Parameters
    ----------
    iso_paths_list : list of str
        List of glob patterns, where each pattern corresponds to one series of
        isochrone files (e.g., different photometric systems).
        Each series must contain the same number of files.
    selected_bands_list : list of list of str
        A nested list specifying which bands to extract for each isochrone series.
        For example: [["J", "H", "Ks"], ["G", "BP", "RP"]].
        For each band, columns `{band}_fSB`, `{band}_f0`, `{band}_fk`, and
        `{band}_iXX` (XX=00–90) are included.
    saved_folder_dir : str
        Directory where the rearranged isochrone files will be written.
    name_suffix : str, optional
        Suffix appended to output filenames (e.g., "(merged)").
    n_common : int, optional
        Number of leading columns that must be identical across all input series
        (default: 35). These columns are preserved as the shared "core" part.

    Notes
    -----
    - The function checks that the first `n_common` columns are identical across
      all series for each isochrone file. If not, an AssertionError is raised.
    - Each output file includes a header with the UTC timestamp and the mapping
      of bands to input file patterns.
    - Within each file, blocks are separated by column headers (prefixed by "#")
      and terminated with "#isochrone terminated".

    Raises
    ------
    ValueError
        If the input series have different lengths, or if only one series is
        provided (no rearrangement needed).
    AssertionError
        If the first `n_common` columns are not identical across input series.

    """

    if name_suffix is not None:
        saved_folder_dir += f'_({name_suffix.replace(" ", "_")})'

    os.makedirs(saved_folder_dir, exist_ok=True)

    iso_file_list = []
    for iso_path_i in iso_paths_list:
        iso_file_i = glob(iso_path_i)
        iso_file_i.sort()
        iso_file_list.append(iso_file_i)

    len_iso_file_list = [len(iso_path_i) for iso_path_i in iso_file_list]
    if len(set(len_iso_file_list)) != 1:
        raise ValueError("'iso_paths_list' must contain the same number of isochrone files.")

    # all_df_merged_list = []
    for iso_file_i in range(len(iso_file_list[0])):
        # filename = os.path.basename(iso_file_list[0][iso_file_i])
        # filename_no_paren = re.sub(r"\s*\([^)]*\)", "", filename)
        # file_name_body, file_name_ext = os.path.splitext(filename_no_paren.strip())
        #
        # if name_suffix is not None:
        #     file_name_body += f' ({name_suffix})'
        # final_file_name = file_name_body + file_name_ext
        #
        # out_path = os.path.join(saved_folder_dir, final_file_name)
        # print(f'Now generating: {out_path}')

        filename = os.path.basename(iso_file_list[0][iso_file_i])
        filename_no_paren = re.sub(r"[\s_]*\([^)]*\)", "", filename).strip()

        # Compatible with both .dat and .dat.gz
        if filename_no_paren.endswith(".dat.gz"):
            file_name_body = filename_no_paren[:-len(".dat.gz")]
            file_name_ext = ".dat.gz"
        elif filename_no_paren.endswith(".dat"):
            file_name_body = filename_no_paren[:-len(".dat")]
            file_name_ext = ".dat"
        else:
            file_name_body, file_name_ext = os.path.splitext(filename_no_paren)

        if name_suffix is not None:
            file_name_body += f"_({name_suffix.replace(" ", "_")})"

        final_file_name = file_name_body + file_name_ext

        out_path = os.path.join(saved_folder_dir, final_file_name)
        print(f"Now rearranging: {out_path}")

        isos_list = []
        for iso_file_list_i in range(len(iso_file_list)):
            isos = read_isochrones(iso_file_list[iso_file_list_i][iso_file_i], use_tqdm=False)
            isos_list.append(isos)

        if isinstance(isos_list[0], pd.DataFrame):
            raise ValueError('Only one series of isochrones does not need rearrange.')
        if len(isos_list) != len(selected_bands_list):
            raise ValueError('The number of series of isochrones are different from the number of series of bands.')

        selected_columns_list = []
        for selected_bands_list_i in selected_bands_list:
            selected_columns_list_i = []
            for band_i in selected_bands_list_i:
                selected_columns_list_i += [band_i+'_fSB', band_i+'_f0', band_i+'_fk'] # The _fSBmag column will be discarded in the future.
                selected_columns_list_i += [band_i+'_i'+f"{inc:02d}" for inc in np.linspace(0, 90, 19, dtype=int)]
            selected_columns_list.append(selected_columns_list_i)

        df_merged_list = []
        for df_tuple in zip(*isos_list):
            for ind in range(len(df_tuple)-1):
                try:
                    pd.testing.assert_frame_equal(df_tuple[ind].iloc[:, :n_common], df_tuple[ind+1].iloc[:, :n_common])
                except AssertionError:
                    raise AssertionError(f"First {n_common} columns are not identical.")
            common_part = df_tuple[0].iloc[:, :n_common].copy()
            df_part_list = [common_part]
            for i in range(len(df_tuple)):
                df_part_list.append(df_tuple[i][selected_columns_list[i]].copy())

            df_merged = pd.concat(df_part_list, axis=1)
            df_merged_list.append(df_merged)
        # all_df_merged_list.append(df_merged_list)

        # with open(out_path, "w") as f:
        #     now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        #     f.write(f'# File rearranged at {now_utc} from:\n')
        #     for i in range(len(selected_bands_list)):
        #         f.write(f'# Band {", ".join(selected_bands_list[i])} from {iso_paths_list[i]}\n')
        #     for i, df in enumerate(df_merged_list):
        #         f.write('# ' + " ".join(df.columns) + "\n")
        #         df.to_csv(f, sep=" ", index=False, header=False, float_format="%.6g", na_rep="nan")
        #     f.write('#isochrone terminated')

        def smart_open(path, mode="wt"):
            if path.endswith(".gz"):
                return gzip.open(path, mode)
            else:
                return open(path, mode)

        with smart_open(out_path, "wt") as f:
            now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            f.write(f'# File rearranged at {now_utc} from:\n')

            for i in range(len(selected_bands_list)):
                f.write(f'# Band {", ".join(selected_bands_list[i])} from {iso_paths_list[i]}\n')

            for i, df in enumerate(df_merged_list):
                f.write('# ' + " ".join(df.columns) + "\n")
                df.to_csv(
                    f,
                    sep=" ",
                    index=False,
                    header=False,
                    float_format="%.6g",
                    na_rep="nan",
                )

            f.write('#isochrone terminated')
    # return all_df_merged_list



def check_rearranged_isochrones(rearranged_iso_path, original_iso_path_list, selected_bands_list, n_common=35, verbose=True):
    """
    Validate that rearranged isochrone files are consistent with their original inputs.

    This function compares the rearranged isochrones (already merged across bands)
    against the original isochrone series to ensure that:
    1. The first `n_common` columns remain identical across all files.
    2. The band-specific columns selected for each series match the original values.

    Parameters
    ----------
    rearranged_iso_path : str
        Glob pattern pointing to the rearranged isochrone files to be validated.
    original_iso_path_list : list of str
        List of glob patterns, each corresponding to one series of original
        isochrone files (e.g., different photometric systems).
    selected_bands_list : list of list of str
        A nested list specifying which bands were used when rearranging.
        For each band, the following columns are validated:
        `{band}_fSB`, `{band}_f0`, `{band}_fk`, and `{band}_iXX` (XX = 00–90).
    n_common : int, optional
        Number of leading columns that must be identical across all isochrone
        series (default: 35).

    Raises
    ------
    AssertionError
        If the first `n_common` columns or any of the selected band columns
        are not identical between the rearranged file and its original sources.
    ValueError
        If file counts in different series do not match.

    Notes
    -----
    - The function prints progress for each file being checked.
    - For every isochrone block, all series are compared column by column.
    - If no AssertionError is raised, the rearranged file is considered consistent.
    """

    print('Starting rearranged isochrone checks...')

    selected_columns_list = []
    for selected_bands_list_i in selected_bands_list:
        selected_columns_list_i = []
        for band_i in selected_bands_list_i:
            selected_columns_list_i += [band_i + '_fSB', band_i + '_f0',
                                        band_i + '_fk']  # The _fSBmag column will be discarded in the future.
            selected_columns_list_i += [band_i + '_i' + f"{inc:02d}" for inc in np.linspace(0, 90, 19, dtype=int)]
        selected_columns_list.append(selected_columns_list_i)

    rearranged_iso = glob(rearranged_iso_path)
    rearranged_iso.sort()

    original_iso_list = []
    for iso_serires_i in range(len(original_iso_path_list)):
        original_iso_i = glob(original_iso_path_list[iso_serires_i])
        original_iso_i.sort()
        original_iso_list.append(original_iso_i)

    for iso_file_index in range(len(rearranged_iso)):
        if verbose:
            print(f'Now working on {rearranged_iso[iso_file_index]}')
            print(f'\treading: {rearranged_iso[iso_file_index]}')
        rearranged_iso_iso_file_index = read_isochrones(rearranged_iso[iso_file_index],use_tqdm=False)

        iso_file_index_list = []
        for iso_serires_i in range(len(original_iso_path_list)):
            if verbose:
                print(f'\treading: {original_iso_list[iso_serires_i][iso_file_index]}')
            iso_file_index_list.append(read_isochrones(original_iso_list[iso_serires_i][iso_file_index], use_tqdm=False))


        for file_i in range(len(iso_file_index_list[0])):
            try:
                for iso_serires_i in range(len(original_iso_path_list)):
                    pd.testing.assert_frame_equal(rearranged_iso_iso_file_index[file_i].iloc[:, :n_common],
                                                  iso_file_index_list[iso_serires_i][file_i].iloc[:, :n_common])
                    pd.testing.assert_frame_equal(rearranged_iso_iso_file_index[file_i][selected_columns_list[iso_serires_i]],
                                                  iso_file_index_list[iso_serires_i][file_i][selected_columns_list[iso_serires_i]])
            except AssertionError:
                raise AssertionError(f"NOT identical!")
        if verbose:
            print('\tIdentical!')

    print('Done! All rearranged isochrone files were correctly combined!')


def restructure_inclination(df):
    """
    Convert inclination columns like G_RP_i00, G_RP_i05 ... into a single column
    'inclination', keeping one column per passband.

    Parameters
    ----------
    df : pandas.DataFrame
        Original wide dataframe

    Returns
    -------
    pandas.DataFrame
        Restructured dataframe with inclination column
    """

    df = df.copy()

    # 找到所有 inclination 列
    inc_cols = [c for c in df.columns if re.search(r"_i\d\d$", c)]

    # 找到所有 passband 名字
    bands = sorted(set(c.rsplit("_i", 1)[0] for c in inc_cols))

    # 所有 inclination
    inc_vals = sorted(set(int(c.split("_i")[1]) for c in inc_cols))

    # 非 inclination 的列
    base_cols = [c for c in df.columns if c not in inc_cols]

    rows = []

    for inc in inc_vals:
        tmp = df[base_cols].copy()
        tmp["Inclination_deg"] = inc
        tmp["Inclination"] = np.round(np.deg2rad(inc), 3)

        for band in bands:
            col = f"{band}_i{inc:02d}"
            if col in df.columns:
                tmp[band] = df[col]

        rows.append(tmp)

    new_df = pd.concat(rows, ignore_index=True)

    return new_df



def add_iso_inc(df_iso_list, omega_ini):
    df_inc_list = []

    for df_i in df_iso_list:
        df_inc = restructure_inclination(df_i)
        idx_vtaneq = df_inc.columns.get_loc("vtaneq")
        vsini = df_inc['vtaneq'].values * np.sin(df_inc['Inclination'].values)
        df_inc.insert(idx_vtaneq + 1, "vsini", vsini)

        true_angular_velocity = df_inc['vtaneq'].values * 1000 * u.m / u.s / (ct.R_sun * df_inc['Req'].values)
        keplerian_angular_velocity = (df_inc['Mass'].values * ct.GM_sun / (ct.R_sun * df_inc['Req'].values) ** 3) ** 0.5
        omega_k = true_angular_velocity / keplerian_angular_velocity
        idx_omega = df_inc.columns.get_loc("omega")
        df_inc.insert(idx_omega, "omega_ini", omega_ini)
        idx_omega = df_inc.columns.get_loc("omega")
        df_inc.insert(idx_omega + 1, "omega_k", omega_k)

        df_inc_list.append(df_inc)

    df_inc_final = pd.concat(df_inc_list, ignore_index=True)

    return df_inc_final




def PARSECv2_collect_data(
    iso_list,
    bands_list=None,
    abs_mags_observed=None,
    abs_err_mags_observed=None,
    photo_err_scaler=None,
    use_tqdm=True,
    label_list=None,
    logAge_list=None,
    omega_ini_list=None,
    Inclination_list=None,
):
    """
    Collect and filter PARSEC v2 isochrone data.

    Notes
    -----
    - Filtering is applied as early as possible to reduce memory usage.
    - If input filter lists are long, they are converted to sets first.
    """

    def _validate_optional_list(name, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            if use_tqdm:
                print(f"Collecting grids with {name} = {value}")
            return set(value)
        raise TypeError(f"{name} should be a list, tuple, or np.ndarray.")

    logAge_set = _validate_optional_list("logAge", logAge_list)
    omega_ini_set = _validate_optional_list("omega_ini", omega_ini_list)
    Inclination_set = _validate_optional_list("Inclination", Inclination_list)

    if label_list is None:
        label_list = [1.0]
        # 0=PMS, 1=MS, 2=SGB, 3=RGB, (4,5,6)=CHEB, ...
        label_set = set(label_list)
    elif label_list == 'all':
        label_set = None
    elif isinstance(label_list, (list, tuple, np.ndarray)):
        label_set = set(label_list)
    else:
        raise TypeError("label_list should equal to 'all' or should be a list, tuple, or np.ndarray.")

    collect_range = None
    if photo_err_scaler is None:
        if isinstance(label_list, (list, tuple, np.ndarray)):
            if use_tqdm:
                print(f"Collecting grids with label = {label_list}...")
        else:
            if use_tqdm:
                print("Collecting all grids...")
    elif isinstance(photo_err_scaler, (int, float)):
        if abs_mags_observed is None or abs_err_mags_observed is None:
            raise ValueError(
                "abs_mags_observed, abs_err_mags_observed, and photo_err_scaler must either "
                "all be None or all be provided together."
            )
        collect_range = photo_err_scaler * np.asarray(abs_err_mags_observed)
        if use_tqdm:
            print(f"Collecting grids with range = {collect_range}")
    else:
        collect_range = np.asarray(photo_err_scaler)
        if use_tqdm:
            print(f"Collecting grids with range = {collect_range}")

    if collect_range is not None:
        if bands_list is None or abs_mags_observed is None:
            raise ValueError(
                "bands_list and abs_mags_observed must be provided when photo filtering is used."
            )
        abs_mags_observed = np.asarray(abs_mags_observed)
        band_columns_for_cut = [f"{band}_fSB" for band in bands_list]

    if isinstance(iso_list, str):
        iso_list = [iso_list]
    iterable = tqdm(iso_list) if use_tqdm else iso_list
    omega_pattern = re.compile(r"omega_ini([^_]+)_")

    df_final_list = []

    for path in iterable:
        match = omega_pattern.search(path)
        if match is None:
            raise ValueError(f"Cannot parse omega_ini from path: {path}")
        omega_ini = round(float(match.group(1)), 2)

        # Early skip before reading full table, if possible
        if omega_ini_set is not None and omega_ini not in omega_ini_set:
            continue

        df = read_isochrones(path, use_tqdm=False)
        df = add_iso_inc(df, omega_ini)

        # Early filters
        if label_set is not None:
            df = df[df["label"].isin(label_set)]
            if df.empty:
                continue

        if logAge_set is not None:
            df = df[df["logAge"].isin(logAge_set)]
            if df.empty:
                continue

        if omega_ini_set is not None:
            df = df[df["omega_ini"].isin(omega_ini_set)]
            if df.empty:
                continue

        if Inclination_set is not None:
            df = df[df["Inclination"].isin(Inclination_set)]
            if df.empty:
                continue

        # Photometric cut
        if collect_range is not None:
            mask = np.ones(len(df), dtype=bool)
            for j, col in enumerate(band_columns_for_cut):
                lower = abs_mags_observed[j] - collect_range[j]
                upper = abs_mags_observed[j] + collect_range[j]
                mask &= (df[col] > lower) & (df[col] < upper)

            df = df[mask]
            if df.empty:
                continue

        df_final_list.append(df)

    if len(df_final_list) == 0:
        return pd.DataFrame()

    df_final = pd.concat(df_final_list, ignore_index=True)
    return df_final





def _read_one_parsec_file_for_regularization(args):
    path_i, path_this, label_list, Inclination_list, logAge_list = args

    df_i = PARSECv2_collect_data(
        [path_this],
        label_list=label_list,
        Inclination_list=Inclination_list,
        logAge_list=logAge_list,
        use_tqdm=False,
    )

    df_i["path_index"] = path_i
    df_i["path"] = path_this

    return df_i


def _process_one_Mini_group(args):
    (
        loop_i,
        group_key,
        df_i,
        group_cols,
        compare_cols,
        interp_output_list,
        Mini_grid,
        dtype,
        grid_type,
        logAge_to_idx,
        MH_to_idx,
        omega_ini_to_idx,
        Inclination_to_idx,
        path_i,
        path_this,
    ) = args

    group_info = {col: group_key[k] for k, col in enumerate(group_cols)}

    result = {
        "status": "skipped",
        "problematic_records": [],
        "safe_duplicate_count": 0,
        "filled_group_count": 0,
        "skipped_group_count": 0,
        "grid_index": None,
        "y_grid": None,
        "grid_bounds": None,
    }

    df_i = df_i.sort_values("Mini").reset_index(drop=True)

    if len(df_i) < 2:
        result["skipped_group_count"] += 1
        return result

    interp_x = df_i["Mini"].values
    interp_y = df_i[interp_output_list].values

    dup_mask_all = pd.Series(interp_x).duplicated(keep=False).to_numpy()

    if dup_mask_all.any():
        local_Mini_min = np.nanmin(interp_x)
        local_Mini_max = np.nanmax(interp_x)
        local_range = local_Mini_max - local_Mini_min

        dup_groups = df_i.loc[dup_mask_all].groupby("Mini", sort=False)

        for mini_value, g in dup_groups:
            n_unique_rows = g[compare_cols].drop_duplicates().shape[0]

            if n_unique_rows == 1:
                result["safe_duplicate_count"] += len(g) - 1
            else:
                bad_pct = (
                    100 * (mini_value - local_Mini_min) / local_range
                    if local_range > 0 else np.nan
                )

                record = {
                    "path_index": path_i,
                    "path": path_this,
                    "loop_index_in_this_file": loop_i,
                    "Mini": mini_value,
                    "position_pct": bad_pct,
                    "n_duplicates": len(g),
                    "n_unique_rows_in_compare_cols": n_unique_rows,
                    "local_Mini_min": local_Mini_min,
                    "local_Mini_max": local_Mini_max,
                }

                for col in group_cols:
                    record[col] = group_info[col]

                result["problematic_records"].append(record)

        keep_mask = ~pd.Series(interp_x).duplicated(keep="first").to_numpy()
        df_i = df_i.loc[keep_mask].reset_index(drop=True)

        interp_x = df_i["Mini"].values
        interp_y = df_i[interp_output_list].values

    if len(interp_x) < 2:
        result["skipped_group_count"] += 1
        return result

    if not np.all(np.diff(interp_x) > 0):
        record = {
            "path_index": path_i,
            "path": path_this,
            "loop_index_in_this_file": loop_i,
            "reason": "Mini is not strictly increasing after duplicate handling",
        }

        for col in group_cols:
            record[col] = group_info[col]

        result["problematic_records"].append(record)
        result["skipped_group_count"] += 1
        return result

    local_Mini_min = np.nanmin(interp_x)
    local_Mini_max = np.nanmax(interp_x)

    y_grid = np.full(
        (len(Mini_grid), len(interp_output_list)),
        np.nan,
        dtype=dtype,
    )

    valid_mask = (
        (Mini_grid >= local_Mini_min) &
        (Mini_grid <= local_Mini_max)
    )

    for j, col in enumerate(interp_output_list):
        f = interp1d(
            interp_x,
            interp_y[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        y_grid[valid_mask, j] = f(Mini_grid[valid_mask]).astype(dtype)

    idx_logAge = logAge_to_idx[group_info["logAge"]]
    idx_MH = MH_to_idx[group_info["MH"]]
    idx_omega_ini = omega_ini_to_idx[group_info["omega_ini"]]

    if grid_type == "without_inc":
        grid_index = (idx_logAge, idx_MH, idx_omega_ini)
    else:
        inc_value = np.round(group_info["Inclination"], 3)

        if inc_value not in Inclination_to_idx:
            result["skipped_group_count"] += 1
            return result

        idx_inc = Inclination_to_idx[inc_value]
        grid_index = (idx_logAge, idx_MH, idx_omega_ini, idx_inc)

    result["status"] = "filled"
    result["filled_group_count"] = 1
    result["grid_index"] = grid_index
    result["y_grid"] = y_grid
    result["grid_bounds"] = np.array(
        [local_Mini_min, local_Mini_max],
        dtype=dtype,
    )

    return result


def regularization_on_Mini_from_files(
    path,
    save_path=".",
    save_name=None,
    Mini_setting=None,
    grid_type="with_inc",
    label_list=None,
    Inclination_list=None,
    logAge_list=None,
    print_path_list=False,
    save_float32=True,
    read_all_at_once=False,
    parallel_file_read=True,
    parallel_backend="process",
    max_workers=None,
):
    if not read_all_at_once and parallel_file_read:
        print("[info] parallel_read is ignored because read_all_at_once=False.")

    if label_list is None:
        label_list = [1.0]

    if Inclination_list is None:
        Inclination_list = np.array(
            np.deg2rad([0.0, 5.0, 10.0, 15, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 90.0])
        )
        Inclination_list = np.round(Inclination_list, 3)

    if save_path[-1] != "/":
        save_path += "/"
    os.makedirs(save_path, exist_ok=True)

    if save_name is None:
        save_name = f"regularized_Mini_grid_{grid_type}.npz"

    dtype = np.float32 if save_float32 else np.float64

    if isinstance(path, str):
        files = glob(path)
    elif isinstance(path, list):
        files = []
        for path_0 in path:
            files += glob(path_0)
    else:
        raise ValueError("path must be str or list.")

    path_list = sorted(files)

    if len(path_list) == 0:
        raise ValueError(f"No files found with path pattern: {path}")

    if print_path_list:
        print(path_list)

    if grid_type == "without_inc":
        in_cols = ["Mini", "logAge", "MH", "omega_ini"]
        interp_variable_list = ["Mini", "logAge", "MH", "omega_ini"]
        group_cols = ["logAge", "MH", "omega_ini"]
        interp_output_list = ["Mass", "omega", "logL", "Req", "Rpol"]

    elif grid_type == "with_inc":
        in_cols = ["Mini", "logAge", "MH", "omega_ini", "Inclination"]
        interp_variable_list = ["Mini", "logAge", "MH", "omega_ini", "Inclination"]
        group_cols = ["logAge", "MH", "omega_ini", "Inclination"]
        interp_output_list = [
            "Mass", "omega", "Req", "Rpol",
            "G", "G_BP", "G_RP",
            "B_T", "V_T",
            "J", "H", "Ks",
        ]

    else:
        raise ValueError('grid_type must be "without_inc" or "with_inc".')

    compare_cols = in_cols + interp_output_list

    df_all = None

    if read_all_at_once:
        if parallel_file_read:
            print("Reading all files at once in parallel. This is faster but uses more memory...")

            read_args = [
                (path_i, path_list[path_i], label_list, Inclination_list, logAge_list)
                for path_i in range(len(path_list))
            ]

            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(path_list))

            if parallel_backend == "process":
                Executor = ProcessPoolExecutor
            elif parallel_backend == "thread":
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")

            with Executor(max_workers=max_workers) as executor:
                df_list = list(
                    tqdm(
                        executor.map(_read_one_parsec_file_for_regularization, read_args),
                        total=len(read_args),
                    )
                )
        else:
            print("Reading all files at once. This is faster but uses more memory...")
            df_list = []

            for path_i in tqdm(range(len(path_list))):
                df_i = PARSECv2_collect_data(
                    [path_list[path_i]],
                    label_list=label_list,
                    Inclination_list=Inclination_list,
                    use_tqdm=False,
                )

                df_i["path_index"] = path_i
                df_i["path"] = path_list[path_i]
                df_list.append(df_i)

        df_all = pd.concat(df_list, ignore_index=True)
        del df_list

        if grid_type == "with_inc":
            df_all["Inclination"] = np.round(df_all["Inclination"].values, 3)

        print(f"df_all shape: {df_all.shape}")
        print(
            f"df_all memory: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB "
            f"({df_all.memory_usage(deep=True).sum() / 1024**3:.2f} GB)"
        )

    print("Collecting axis values...")

    Mini_min_list = []
    Mini_max_list = []
    logAge_list = []
    MH_list = []
    omega_ini_list = []
    Inclination_axis_list = []

    if read_all_at_once:
        df = df_all.copy()

        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in df_all: {missing_cols}")

        Mini_min_list.append(np.nanmin(df["Mini"].values))
        Mini_max_list.append(np.nanmax(df["Mini"].values))

        logAge_list.append(df["logAge"].dropna().unique())
        MH_list.append(df["MH"].dropna().unique())
        omega_ini_list.append(df["omega_ini"].dropna().unique())

        if grid_type == "with_inc":
            Inclination_axis_list.append(
                np.round(df["Inclination"].dropna().unique(), 3)
            )

        del df

    else:
        for path_i in tqdm(range(len(path_list))):
            df = PARSECv2_collect_data(
                [path_list[path_i]],
                label_list=label_list,
                Inclination_list=Inclination_list,
                use_tqdm=False,
            )

            if grid_type == "with_inc":
                df["Inclination"] = np.round(df["Inclination"].values, 3)

            missing_cols = [c for c in compare_cols if c not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"Missing columns in df from {path_list[path_i]}: {missing_cols}"
                )

            Mini_min_list.append(np.nanmin(df["Mini"].values))
            Mini_max_list.append(np.nanmax(df["Mini"].values))

            logAge_list.append(df["logAge"].dropna().unique())
            MH_list.append(df["MH"].dropna().unique())
            omega_ini_list.append(df["omega_ini"].dropna().unique())

            if grid_type == "with_inc":
                Inclination_axis_list.append(
                    np.round(df["Inclination"].dropna().unique(), 3)
                )

            del df

    if Mini_setting is None:
        Mini_min = np.nanmin(Mini_min_list)
        Mini_max = np.nanmax(Mini_max_list)
        Mini_step = 0.01
    else:
        if not isinstance(Mini_setting, (list, tuple, np.ndarray)) or len(Mini_setting) != 3:
            raise ValueError("Mini_setting must be [Mini_min, Mini_max, Mini_step].")
        Mini_min, Mini_max, Mini_step = Mini_setting

    Mini_min_round = np.floor(Mini_min * 100) / 100
    Mini_max_round = np.ceil(Mini_max * 100) / 100

    Mini_grid = (
        np.arange(
            int(np.round(Mini_min_round * 100)),
            int(np.round(Mini_max_round * 100)) + 1,
            int(np.round(Mini_step * 100)),
        ) / 100.0
    )

    logAge_grid = np.sort(np.unique(np.concatenate(logAge_list)))
    MH_grid = np.sort(np.unique(np.concatenate(MH_list)))
    omega_ini_grid = np.sort(np.unique(np.concatenate(omega_ini_list)))

    if grid_type == "with_inc":
        Inclination_grid = np.round(np.asarray(Inclination_list), 3)
        Inclination_grid = np.round(np.sort(Inclination_grid), 3)
    else:
        Inclination_grid = None

    logAge_to_idx = {v: i for i, v in enumerate(logAge_grid)}
    MH_to_idx = {v: i for i, v in enumerate(MH_grid)}
    omega_ini_to_idx = {v: i for i, v in enumerate(omega_ini_grid)}

    if grid_type == "with_inc":
        Inclination_to_idx = {v: i for i, v in enumerate(Inclination_grid)}
    else:
        Inclination_to_idx = None

    if grid_type == "without_inc":
        grid_outputs = np.full(
            (
                len(Mini_grid),
                len(logAge_grid),
                len(MH_grid),
                len(omega_ini_grid),
                len(interp_output_list),
            ),
            np.nan,
            dtype=dtype,
        )

        grid_bounds = np.full(
            (
                len(logAge_grid),
                len(MH_grid),
                len(omega_ini_grid),
                2,
            ),
            np.nan,
            dtype=dtype,
        )

    else:
        grid_outputs = np.full(
            (
                len(Mini_grid),
                len(logAge_grid),
                len(MH_grid),
                len(omega_ini_grid),
                len(Inclination_grid),
                len(interp_output_list),
            ),
            np.nan,
            dtype=dtype,
        )

        grid_bounds = np.full(
            (
                len(logAge_grid),
                len(MH_grid),
                len(omega_ini_grid),
                len(Inclination_grid),
                2,
            ),
            np.nan,
            dtype=dtype,
        )

    print(f"Mini_grid:  [min: {np.min(Mini_grid)}, max: {np.max(Mini_grid)}]")
    print(f"Mini_grid shape: {Mini_grid.shape}")
    print(f"logAge_grid shape: {logAge_grid.shape}")
    print(f"MH_grid shape: {MH_grid.shape}")
    print(f"omega_ini_grid shape: {omega_ini_grid.shape}")
    if grid_type == "with_inc":
        print(f"Inclination_grid shape: {Inclination_grid.shape}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_bounds shape: {grid_bounds.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(
        f"grid_outputs memory: {grid_outputs.nbytes / 1024**2:.2f} MB "
        f"({grid_outputs.nbytes / 1024**3:.2f} GB)"
    )

    print("Filling regular grids...")

    problematic_records = []
    safe_duplicate_count = 0
    filled_group_count = 0
    skipped_group_count = 0

    if read_all_at_once:
        file_iter = [(-1, "df_all")]
    else:
        file_iter = tqdm([(path_i, path_list[path_i]) for path_i in range(len(path_list))])

    for path_i, path_this in file_iter:
        if read_all_at_once:
            df = df_all.copy()
        else:
            df = PARSECv2_collect_data(
                [path_this],
                label_list=label_list,
                Inclination_list=Inclination_list,
                use_tqdm=False,
            )

            if grid_type == "with_inc":
                df["Inclination"] = np.round(df["Inclination"].values, 3)

            df["path_index"] = path_i
            df["path"] = path_this

        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in df from {path_this}: {missing_cols}")

        grouped = list(df.groupby(group_cols, sort=False))

        task_args = []

        for loop_i, (group_key, df_i) in enumerate(grouped):
            if len(group_cols) == 1:
                group_key = (group_key,)

            task_args.append(
                (
                    loop_i,
                    group_key,
                    df_i,
                    group_cols,
                    compare_cols,
                    interp_output_list,
                    Mini_grid,
                    dtype,
                    grid_type,
                    logAge_to_idx,
                    MH_to_idx,
                    omega_ini_to_idx,
                    Inclination_to_idx,
                    path_i,
                    path_this,
                )
            )

        use_process = False
        local_max_workers = 8

        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        with Executor(max_workers=local_max_workers) as executor:
            iterator = executor.map(_process_one_Mini_group, task_args)

            if read_all_at_once:
                iterator = tqdm(iterator, total=len(task_args))

            results = list(iterator)

        for res in results:
            problematic_records.extend(res["problematic_records"])
            safe_duplicate_count += res["safe_duplicate_count"]
            skipped_group_count += res["skipped_group_count"]

            if res["status"] != "filled":
                continue

            y_grid = res["y_grid"]
            bounds_i = res["grid_bounds"]
            grid_index = res["grid_index"]

            if grid_type == "without_inc":
                idx_logAge, idx_MH, idx_omega_ini = grid_index

                grid_outputs[:, idx_logAge, idx_MH, idx_omega_ini, :] = y_grid
                grid_bounds[idx_logAge, idx_MH, idx_omega_ini, :] = bounds_i

            else:
                idx_logAge, idx_MH, idx_omega_ini, idx_inc = grid_index

                grid_outputs[:, idx_logAge, idx_MH, idx_omega_ini, idx_inc, :] = y_grid
                grid_bounds[idx_logAge, idx_MH, idx_omega_ini, idx_inc, :] = bounds_i

            filled_group_count += res["filled_group_count"]

        del df

    if read_all_at_once:
        del df_all

    problematic_df = pd.DataFrame(problematic_records)

    save_file = save_path + save_name
    problematic_path = save_path + f"problematic_duplicated_{Path(save_name).stem}.csv"

    save_dict = dict(
        Mini_grid=Mini_grid,
        logAge_grid=logAge_grid,
        MH_grid=MH_grid,
        omega_ini_grid=omega_ini_grid,
        grid_outputs=grid_outputs,
        grid_bounds=grid_bounds,
        Mini_bounds_output_list=np.array(["Mini_min", "Mini_max"], dtype=object),
        in_cols=np.array(in_cols, dtype=object),
        interp_output_list=np.array(interp_output_list, dtype=object),
        interp_variable_list=np.array(interp_variable_list, dtype=object),
        group_cols=np.array(group_cols, dtype=object),
        grid_type=np.array(grid_type),
        label_list=np.array(label_list),
        save_float32=np.array(save_float32),
        read_all_at_once=np.array(read_all_at_once),
        grid_dtype=np.array(str(grid_outputs.dtype)),
        safe_duplicate_count=np.array(safe_duplicate_count),
        filled_group_count=np.array(filled_group_count),
        skipped_group_count=np.array(skipped_group_count),
    )

    if grid_type == "with_inc":
        save_dict["Inclination_grid"] = Inclination_grid

    np.savez_compressed(save_file, **save_dict)
    problematic_df.to_csv(problematic_path, index=False)

    total_bytes = (
        grid_outputs.nbytes
        + grid_bounds.nbytes
        + Mini_grid.nbytes
        + logAge_grid.nbytes
        + MH_grid.nbytes
        + omega_ini_grid.nbytes
    )

    if grid_type == "with_inc":
        total_bytes += Inclination_grid.nbytes

    print("Finished regularization.")
    print(f"grid_type: {grid_type}")
    print(f"read_all_at_once: {read_all_at_once}")
    print(f"parallel_file_read: {parallel_file_read}")
    print(f"parallel_backend: {parallel_backend}")
    print(f"max_workers: {max_workers}")
    print(f"save_float32: {save_float32}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(f"grid_bounds dtype: {grid_bounds.dtype}")
    print(f"filled_group_count: {filled_group_count}")
    print(f"skipped_group_count: {skipped_group_count}")
    print(f"safe_duplicate_count: {safe_duplicate_count}")
    print(f"problematic_duplicate_count: {len(problematic_df)}")
    print(f"problematic duplicated Mini records are saved at: {os.path.abspath(problematic_path)}")
    print(f"Grids are saved at: {os.path.abspath(save_file)}")
    print(
        f"total grid memory: {total_bytes / 1024**2:.2f} MB "
        f"({total_bytes / 1024**3:.2f} GB)"
    )

    return os.path.abspath(save_file), problematic_df


def load_regularized_Mini_interpolator(npz_path, force_float64=False):
    data = np.load(npz_path, allow_pickle=True)

    grid_type = str(data["grid_type"])

    Mini_grid = data["Mini_grid"]
    logAge_grid = data["logAge_grid"]
    MH_grid = data["MH_grid"]
    omega_ini_grid = data["omega_ini_grid"]

    grid_outputs = data["grid_outputs"]

    if "grid_bounds" not in data.files:
        raise ValueError(
            "This Mini grid does not contain grid_bounds. "
            "Please regenerate the Mini grid with the updated regularization_on_Mini_from_files()."
        )

    grid_bounds = data["grid_bounds"]

    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)
        grid_bounds = grid_bounds.astype(np.float64)

    interp_variable_list = data["interp_variable_list"].astype(str).tolist()
    interp_output_list = data["interp_output_list"].astype(str).tolist()

    if grid_type == "without_inc":
        points = (
            Mini_grid,
            logAge_grid,
            MH_grid,
            omega_ini_grid,
        )

        param_points = (
            logAge_grid,
            MH_grid,
            omega_ini_grid,
        )

    elif grid_type == "with_inc":
        Inclination_grid = data["Inclination_grid"]

        points = (
            Mini_grid,
            logAge_grid,
            MH_grid,
            omega_ini_grid,
            Inclination_grid,
        )

        param_points = (
            logAge_grid,
            MH_grid,
            omega_ini_grid,
            Inclination_grid,
        )

    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    interp_func = RegularGridInterpolator(
        points,
        grid_outputs,
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_u_bounds = RegularGridInterpolator(
        param_points,
        grid_bounds,
        bounds_error=False,
        fill_value=np.nan,
    )

    n_param = len(interp_variable_list)

    def interp_u(x, eps=1e-12, clip_u=False, return_bounds=False):
        x = np.asarray(x, dtype=np.float64)

        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")
            x = x[None, :]
            single_input = True

        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")

        Mini = x[:, 0]
        params = x[:, 1:]

        bounds = interp_u_bounds(params)

        Mini_min = bounds[:, 0]
        Mini_max = bounds[:, 1]
        width = Mini_max - Mini_min

        u = np.full_like(Mini, np.nan, dtype=np.float64)

        valid = np.isfinite(width) & (width > eps)
        u[valid] = (Mini[valid] - Mini_min[valid]) / width[valid]

        if clip_u:
            u = np.clip(u, 0.0, 1.0)

        if single_input:
            u = u[0]
            bounds = bounds[0]

        if return_bounds:
            return u, bounds
        return u

    def interp_u_to_outputs(x, eps=1e-12, clip_Mini=False, return_Mini=False, return_bounds=False):
        x = np.asarray(x, dtype=np.float64)

        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")
            x = x[None, :]
            single_input = True

        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")

        u = x[:, 0]
        params = x[:, 1:]

        bounds = interp_u_bounds(params)

        Mini_min = bounds[:, 0]
        Mini_max = bounds[:, 1]
        width = Mini_max - Mini_min

        Mini = np.full_like(u, np.nan, dtype=np.float64)

        valid = np.isfinite(width) & (width > eps)
        Mini[valid] = Mini_min[valid] + u[valid] * width[valid]

        if clip_Mini:
            Mini = np.clip(Mini, Mini_min, Mini_max)

        x_Mini = np.column_stack([Mini, params])

        y = interp_func(x_Mini)

        if single_input:
            y = y[0]
            Mini = Mini[0]
            bounds = bounds[0]

        if return_Mini and return_bounds:
            return y, Mini, bounds
        elif return_Mini:
            return y, Mini
        elif return_bounds:
            return y, bounds
        else:
            return y

    total_grid_bytes = (
        grid_outputs.nbytes
        + grid_bounds.nbytes
        + Mini_grid.nbytes
        + logAge_grid.nbytes
        + MH_grid.nbytes
        + omega_ini_grid.nbytes
    )

    if grid_type == "with_inc":
        total_grid_bytes += Inclination_grid.nbytes

    print("Loaded regularized Mini grid.")
    print(f"grid_type: {grid_type}")
    print(f"interp_variable_list: {interp_variable_list}")
    print(f"interp_output_list: {interp_output_list}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_bounds shape: {grid_bounds.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(f"grid_bounds dtype: {grid_bounds.dtype}")

    if "save_float32" in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")

    if "read_all_at_once" in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")

    print(
        f"total grid memory: {total_grid_bytes / 1024**2:.2f} MB "
        f"({total_grid_bytes / 1024**3:.2f} GB)"
    )

    return (
        interp_func,
        interp_u,
        interp_u_to_outputs,
        interp_u_bounds,
        interp_variable_list,
        interp_output_list,
        data,
    )



def _as_dir_string(path, create=True):
    """
    Convert a directory path to absolute string with trailing slash.
    Needed because regularization_on_Mini_from_files uses string concatenation.
    """
    path = Path(path).expanduser().resolve()

    if create:
        path.mkdir(parents=True, exist_ok=True)

    path_str = str(path)

    if not path_str.endswith(os.sep):
        path_str += os.sep

    return path_str


def _as_glob_pattern_string(path):
    """
    Convert glob pattern path to string.
    Do not mkdir/resolve the '*' part.
    """
    return str(Path(path).expanduser())


def _label_list_to_tag(label_list):
    if label_list == [1] or label_list == [1.0]:
        return "MS"

    if set(label_list) == {0, 1} or set(label_list) == {0.0, 1.0}:
        return "preMS_MS"

    return "labels_" + "_".join(str(x).replace(".", "p") for x in label_list)


def regularize_PARSECv2_Mini_grid(
    path=None,
    save_path=None,
    save_name=None,
    age_initial="1e6",
    age_end="2001e6",
    age_interval="10e6",
    age_type="linear",
    photometry_suffix="2mass & tycho2 & gaia dr3",
    grid_type="with_inc",
    label_list=None,
    print_path_list=False,
    save_float32=True,
    read_all_at_once=False,
    **kwargs,
):
    """
    Wrapper for regularization_on_Mini_from_files().

    Default input:
        DATA_DIR/isochrones/PARSEC_v2/
        1e6_2001e6yr_linear_isochrones_rearranged (2mass & tycho2 & gaia dr3)/*

    Default output:
        DATA_DIR/isochrones/regular_PARSECv2_Mini_grids/
    """

    if label_list is None:
        label_list = [1]

    if path is None:
        parsec_dir = DATA_DIR / "isochrones" / "PARSEC_v2"

        path = [
            str(
                parsec_dir
                / f"{age_initial}_{age_end}yr_{age_type}_isochrones_rearranged ({photometry_suffix})"
                / "*"
            )
        ]

    elif isinstance(path, (str, Path)):
        path = [_as_glob_pattern_string(path)]

    elif isinstance(path, (list, tuple)):
        path = [_as_glob_pattern_string(p) for p in path]

    else:
        raise ValueError("path must be None, str, Path, list, or tuple.")

    if save_path is None:
        save_path = DATA_DIR / "isochrones" / "regular_PARSECv2_Mini_grids"

    save_path = _as_dir_string(save_path, create=True)

    if save_name is None:
        label_tag = _label_list_to_tag(label_list)

        save_name = (
            f"regularized_Mini_grid_"
            f"{age_initial}_{age_end}yr_step{age_interval}_"
            f"{age_type}_isochrones_{label_tag}_{grid_type}.npz"
        )

    save_file, problematic_df = regularization_on_Mini_from_files(
        path=path,
        save_path=save_path,
        save_name=save_name,
        grid_type=grid_type,
        label_list=label_list,
        print_path_list=print_path_list,
        save_float32=save_float32,
        read_all_at_once=read_all_at_once,
        **kwargs,
    )

    return save_file, problematic_df


def load_default_regularized_PARSECv2_Mini_interpolator(
    npz_path=None,
    age_initial="1e6",
    age_end="2001e6",
    age_interval="10e6",
    age_type="linear",
    label_tag="MS",
    grid_type="with_inc",
    force_float64=False,
):
    """
    Load regularized Mini interpolator.

    If npz_path is None, use the default path under:
        DATA_DIR/isochrones/regular_PARSECv2_Mini_grids/
    """

    if npz_path is None:
        save_dir = DATA_DIR / "isochrones" / "regular_PARSECv2_Mini_grids"

        save_name = (
            f"regularized_Mini_grid_"
            f"{age_initial}_{age_end}yr_step{age_interval}_"
            f"{age_type}_isochrones_{label_tag}_{grid_type}.npz"
        )

        npz_path = save_dir / save_name

    npz_path = Path(npz_path).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"Regularized Mini grid not found: {npz_path}")

    return load_regularized_Mini_interpolator(
        npz_path=npz_path,
        force_float64=force_float64,
    )



def _get_u_grid_columns(grid_type):
    if grid_type == "without_inc":
        interp_variable_list = ["Mini", "logAge", "MH", "omega_ini"]
        interp_output_list = ["Mass", "omega", "logL", "Req", "Rpol"]
        other_interp_variable_list = ["logAge", "MH", "omega_ini"]

    elif grid_type == "with_inc":
        interp_variable_list = ["Mini", "logAge", "MH", "omega_ini", "Inclination"]
        interp_output_list = [
            "Mass", "omega", "Req", "Rpol",
            "G", "G_BP", "G_RP",
            "B_T", "V_T",
            "J", "H", "Ks",
        ]
        other_interp_variable_list = ["logAge", "MH", "omega_ini", "Inclination"]

    else:
        raise ValueError('grid_type must be "without_inc" or "with_inc".')

    interp_variable_list_regular = ["u"] + other_interp_variable_list

    return interp_variable_list, interp_output_list, other_interp_variable_list, interp_variable_list_regular


def _process_one_u_group(args):
    (
        group_key,
        g,
        other_interp_variable_list,
        interp_output_list,
        axis_to_idx,
        u_grid,
        dtype,
        min_width_threshold,
        interp_kind_1d,
        path_i,
        path_this,
    ) = args

    result = {
        "status": "skipped",
        "grid_index": None,
        "y_on_grid": None,
        "bounds": None,
        "bad_group_records": [],
        "duplicated_u_records": [],
        "filled_group_count": 0,
        "skipped_group_count": 0,
    }

    if len(other_interp_variable_list) == 1:
        group_key = (group_key,)

    group_dict = {
        c: group_key[k]
        for k, c in enumerate(other_interp_variable_list)
    }

    width_i = g["Mini_width_used"].iloc[0]

    if not np.isfinite(width_i) or width_i < min_width_threshold:
        rec = {
            "path_index": path_i,
            "path": path_this,
            "reason": "Mini_width_used too small",
            "Mini_width_used": width_i,
        }
        rec.update(group_dict)
        result["bad_group_records"].append(rec)
        result["skipped_group_count"] = 1
        return result

    idx = tuple(
        axis_to_idx[c][group_dict[c]]
        for c in other_interp_variable_list
    )

    mini_min_vals = g["Mini_min_used"].unique()
    mini_max_vals = g["Mini_max"].unique()

    if len(mini_min_vals) != 1 or len(mini_max_vals) != 1:
        rec = {
            "path_index": path_i,
            "path": path_this,
            "reason": "inconsistent Mini bounds",
        }
        rec.update(group_dict)
        result["bad_group_records"].append(rec)
        result["skipped_group_count"] = 1
        return result

    bounds = np.array([mini_min_vals[0], mini_max_vals[0]], dtype=dtype)

    g = g.sort_values("u").copy()

    dup_mask = g["u"].duplicated(keep=False).to_numpy()

    if dup_mask.any():
        dup_u_values = g.loc[dup_mask, "u"].unique()

        for dup_u in dup_u_values:
            g_dup = g[g["u"] == dup_u]
            rec = {
                "path_index": path_i,
                "path": path_this,
                "u": dup_u,
                "n_duplicates": len(g_dup),
            }
            rec.update(group_dict)
            result["duplicated_u_records"].append(rec)

    g2 = g.groupby("u", as_index=False)[interp_output_list].mean()

    u_vals = g2["u"].to_numpy(dtype=np.float64)
    y_vals = g2[interp_output_list].to_numpy(dtype=np.float64)

    if len(u_vals) < 2:
        rec = {
            "path_index": path_i,
            "path": path_this,
            "reason": "too few u points",
        }
        rec.update(group_dict)
        result["bad_group_records"].append(rec)
        result["skipped_group_count"] = 1
        return result

    if not np.all(np.diff(u_vals) > 0):
        rec = {
            "path_index": path_i,
            "path": path_this,
            "reason": "u not strictly increasing",
        }
        rec.update(group_dict)
        result["bad_group_records"].append(rec)
        result["skipped_group_count"] = 1
        return result

    try:
        f = interp1d(
            u_vals,
            y_vals,
            axis=0,
            kind=interp_kind_1d,
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )

        y_on_grid = f(u_grid).astype(dtype)

        result["status"] = "filled"
        result["grid_index"] = idx
        result["y_on_grid"] = y_on_grid
        result["bounds"] = bounds
        result["filled_group_count"] = 1
        return result

    except Exception as e:
        rec = {
            "path_index": path_i,
            "path": path_this,
            "reason": repr(e),
        }
        rec.update(group_dict)
        result["bad_group_records"].append(rec)
        result["skipped_group_count"] = 1
        return result

def regularization_on_u_from_files(
    path,
    save_path=".",
    save_name=None,
    grid_type="with_inc",
    label_list=None,
    Inclination_list=None,
    logAge_list=None,
    print_path_list=False,
    n_u=None,
    Mini_min_mode=None,
    min_width_threshold=0.01,
    interp_kind_1d="linear",
    save_float32=True,
    u_save_range=None,
    read_all_at_once=False,
    parallel_file_read=True,
    parallel_backend="process",      # "process" or "thread"
    parallel_interp=True,
    interp_backend="thread",         # recommend "thread" in Jupyter
    max_workers=None,
    interp_max_workers=8,
):
    if not read_all_at_once and parallel_file_read:
        print("[info] parallel_read is ignored because read_all_at_once=False.")

    if save_path[-1] != "/":
        save_path += "/"
    os.makedirs(save_path, exist_ok=True)

    if label_list is None:
        label_list = [1.0]

    if Inclination_list is None:
        Inclination_list = np.array(
            np.deg2rad([0.0, 5.0, 10.0, 15, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 90.0])
        )
        Inclination_list = np.round(Inclination_list, 3)

    if save_name is None:
        save_name = f"regularized_u_grid_{grid_type}.npz"

    dtype = np.float32 if save_float32 else np.float64

    if isinstance(path, str):
        path_list = sorted(glob(path))
    elif isinstance(path, list):
        files = []
        for path_0 in path:
            files += glob(path_0)
        path_list = sorted(files)
    else:
        raise ValueError("path must be str or list.")

    if len(path_list) == 0:
        raise ValueError(f"No files found with path pattern: {path}")

    if print_path_list:
        print(path_list)

    (
        interp_variable_list,
        interp_output_list,
        other_interp_variable_list,
        interp_variable_list_regular,
    ) = _get_u_grid_columns(grid_type)

    df_all = None

    # ============================================================
    # optional read all files
    # ============================================================
    if read_all_at_once:
        if parallel_file_read:
            print("Reading all files at once in parallel. This is faster but uses more memory...")

            read_args = [
                (path_i, path_list[path_i], label_list, Inclination_list, logAge_list)
                for path_i in range(len(path_list))
            ]

            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(path_list))

            if parallel_backend == "process":
                Executor = ProcessPoolExecutor
            elif parallel_backend == "thread":
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")

            with Executor(max_workers=max_workers) as executor:
                df_list = list(
                    tqdm(
                        executor.map(_read_one_parsec_file_for_regularization, read_args),
                        total=len(read_args),
                    )
                )

        else:
            print("Reading all files at once. This is faster but uses more memory...")

            df_list = []

            for path_i in tqdm(range(len(path_list))):
                df_i = PARSECv2_collect_data(
                    [path_list[path_i]],
                    label_list=label_list,
                    Inclination_list=Inclination_list,
                    use_tqdm=False,
                )

                df_i["path_index"] = path_i
                df_i["path"] = path_list[path_i]

                df_list.append(df_i)

        df_all = pd.concat(df_list, ignore_index=True)
        del df_list

        if grid_type == "with_inc":
            df_all["Inclination"] = np.round(df_all["Inclination"].values, 3)

        print(f"df_all shape: {df_all.shape}")
        print(
            f"df_all memory: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB "
            f"({df_all.memory_usage(deep=True).sum() / 1024**3:.2f} GB)"
        )

    # ============================================================
    # collect axis values and Mini bounds
    # ============================================================
    print("Collecting axis values and Mini bounds...")

    axis_values = {c: [] for c in other_interp_variable_list}
    bounds_records = []

    if read_all_at_once:
        df = df_all.copy()

        missing_cols = [c for c in interp_variable_list + interp_output_list if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in df_all: {missing_cols}")

        df = df[interp_variable_list + interp_output_list + ["path_index", "path"]].dropna().copy()

        for c in other_interp_variable_list:
            axis_values[c].append(df[c].dropna().unique())

        df_bounds = (
            df.groupby(other_interp_variable_list)["Mini"]
            .agg(Mini_min_orig="min", Mini_max="max")
            .reset_index()
        )

        df_bounds["path_index"] = -1
        df_bounds["path"] = "df_all"

        del df

    else:
        for path_i in tqdm(range(len(path_list))):
            df = PARSECv2_collect_data(
                [path_list[path_i]],
                label_list=label_list,
                Inclination_list=Inclination_list,
                use_tqdm=False,
            )

            if grid_type == "with_inc":
                df["Inclination"] = np.round(df["Inclination"].values, 3)

            missing_cols = [c for c in interp_variable_list + interp_output_list if c not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"Missing columns in df from {path_list[path_i]}: {missing_cols}"
                )

            df = df[interp_variable_list + interp_output_list].dropna().copy()

            for c in other_interp_variable_list:
                axis_values[c].append(df[c].dropna().unique())

            df_bounds_i = (
                df.groupby(other_interp_variable_list)["Mini"]
                .agg(Mini_min_orig="min", Mini_max="max")
                .reset_index()
            )

            df_bounds_i["path_index"] = path_i
            df_bounds_i["path"] = path_list[path_i]

            bounds_records.append(df_bounds_i)

            del df

        df_bounds = pd.concat(bounds_records, ignore_index=True)

    axis_grids = {}
    for c in other_interp_variable_list:
        axis_grids[c] = np.sort(np.unique(np.concatenate(axis_values[c])))

    if Mini_min_mode is None:
        df_bounds["Mini_min_used"] = df_bounds["Mini_min_orig"]
    else:
        df_bounds["Mini_min_used"] = float(Mini_min_mode)

    df_bounds["Mini_width_used"] = df_bounds["Mini_max"] - df_bounds["Mini_min_used"]

    bad_width_df = df_bounds[df_bounds["Mini_width_used"] < min_width_threshold].copy()

    if len(bad_width_df) > 0:
        bad_width_path = save_path + f"bad_u_width_records_{grid_type}.csv"
        bad_width_df.to_csv(bad_width_path, index=False)
        print(f"Warning: bad width records saved to: {os.path.abspath(bad_width_path)}")

    # ============================================================
    # u grid
    # ============================================================
    Mini_global_min = np.nanmin(df_bounds["Mini_min_orig"].to_numpy(dtype=np.float64))
    Mini_global_max = np.nanmax(df_bounds["Mini_max"].to_numpy(dtype=np.float64))

    if not np.isfinite(Mini_global_min) or not np.isfinite(Mini_global_max):
        raise ValueError("Cannot determine global Mini range from df_bounds.")

    if Mini_global_max <= Mini_global_min:
        raise ValueError(
            f"Invalid global Mini range: "
            f"Mini_global_min={Mini_global_min}, Mini_global_max={Mini_global_max}"
        )

    n_u_default = int(np.ceil((Mini_global_max - Mini_global_min) / 0.01)) + 1

    if n_u is None:
        n_u = n_u_default
    else:
        n_u = int(n_u)

        n_u_min_recommended = int(np.ceil(n_u_default * 0.5))

        if n_u < n_u_min_recommended:
            raise ValueError(
                f"n_u is too small. "
                f"n_u >= {n_u_min_recommended} is recommended. "
                f"Default value: {n_u_default}."
            )

    u_grid_full = np.linspace(0.0, 1.0, n_u, dtype=np.float64)

    if u_save_range is None:
        u_grid = u_grid_full
        u_save_min = 0.0
        u_save_max = 1.0
    else:
        if not isinstance(u_save_range, (list, tuple, np.ndarray)) or len(u_save_range) != 2:
            raise ValueError("u_save_range must be None or [u_min, u_max].")

        u_save_min = float(u_save_range[0])
        u_save_max = float(u_save_range[1])

        if not (0.0 <= u_save_min < u_save_max <= 1.0):
            raise ValueError("u_save_range must satisfy 0.0 <= u_min < u_max <= 1.0.")

        u_mask = (u_grid_full >= u_save_min) & (u_grid_full <= u_save_max)
        u_grid = u_grid_full[u_mask]

        if len(u_grid) < 2:
            raise ValueError(
                f"u_save_range={u_save_range} leaves fewer than 2 u-grid points. "
                "Increase n_u or use a wider u_save_range."
            )

    print(f"Mini global range: [{Mini_global_min}, {Mini_global_max}]")
    print(f"n_u_default for 0.01 Mini resolution: {n_u_default}")
    print(f"n_u used: {n_u}")

    # ============================================================
    # allocate dense grids
    # ============================================================
    param_sizes = [len(axis_grids[c]) for c in other_interp_variable_list]
    n_out = len(interp_output_list)

    grid_outputs = np.full(
        [len(u_grid)] + param_sizes + [n_out],
        np.nan,
        dtype=dtype,
    )

    grid_bounds = np.full(
        param_sizes + [2],
        np.nan,
        dtype=dtype,
    )

    axis_to_idx = {
        c: {v: i for i, v in enumerate(axis_grids[c])}
        for c in other_interp_variable_list
    }

    print(f"u_grid shape: {u_grid.shape}")
    for c in other_interp_variable_list:
        print(f"{c}_grid shape: {axis_grids[c].shape}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_bounds shape: {grid_bounds.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(
        f"grid_outputs memory: {grid_outputs.nbytes / 1024**2:.2f} MB "
        f"({grid_outputs.nbytes / 1024**3:.2f} GB)"
    )

    # ============================================================
    # fill grids
    # ============================================================
    print("Filling u-regular grids...")

    filled_group_count = 0
    skipped_group_count = 0
    bad_group_records = []
    duplicated_u_records = []

    if read_all_at_once:
        file_iter = [(-1, "df_all")]
    else:
        file_iter = tqdm([(path_i, path_list[path_i]) for path_i in range(len(path_list))])

    for path_i, path_this in file_iter:
        if read_all_at_once:
            df = df_all.copy()
        else:
            df = PARSECv2_collect_data(
                [path_this],
                label_list=label_list,
                Inclination_list=Inclination_list,
                use_tqdm=False,
            )

            if grid_type == "with_inc":
                df["Inclination"] = np.round(df["Inclination"].values, 3)

            df["path_index"] = path_i
            df["path"] = path_this

        df = df[interp_variable_list + interp_output_list + ["path_index", "path"]].dropna().copy()
        df = df.sort_values(other_interp_variable_list + ["Mini"]).reset_index(drop=True)

        df_bounds_i = (
            df.groupby(other_interp_variable_list)["Mini"]
            .agg(Mini_min_orig="min", Mini_max="max")
            .reset_index()
        )

        if Mini_min_mode is None:
            df_bounds_i["Mini_min_used"] = df_bounds_i["Mini_min_orig"]
        else:
            df_bounds_i["Mini_min_used"] = float(Mini_min_mode)

        df_bounds_i["Mini_width_used"] = (
            df_bounds_i["Mini_max"] - df_bounds_i["Mini_min_used"]
        )

        df = df.merge(
            df_bounds_i,
            on=other_interp_variable_list,
            how="left",
        )

        width = df["Mini_width_used"].to_numpy(dtype=np.float64)
        Mini_vals = df["Mini"].to_numpy(dtype=np.float64)
        Mini_min_used = df["Mini_min_used"].to_numpy(dtype=np.float64)

        u_raw = np.full_like(Mini_vals, np.nan, dtype=np.float64)

        valid_width = np.isfinite(width) & (width >= min_width_threshold)

        u_raw[valid_width] = (
                                     Mini_vals[valid_width] - Mini_min_used[valid_width]
                             ) / width[valid_width]

        df["u_raw"] = u_raw
        df["u"] = np.clip(u_raw, 0.0, 1.0)

        grouped = list(df.groupby(other_interp_variable_list, sort=False))

        task_args = [
            (
                group_key,
                g,
                other_interp_variable_list,
                interp_output_list,
                axis_to_idx,
                u_grid,
                dtype,
                min_width_threshold,
                interp_kind_1d,
                path_i,
                path_this,
            )
            for group_key, g in grouped
        ]

        if parallel_interp:
            if interp_backend == "process":
                Executor = ProcessPoolExecutor
            elif interp_backend == "thread":
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("interp_backend must be 'process' or 'thread'.")
            # if read_all_at_once:
            with Executor(max_workers=interp_max_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(_process_one_u_group, task_args),
                        total=len(task_args),
                    )
                )
        else:
            results = [
                _process_one_u_group(arg)
                for arg in tqdm(task_args)
            ]

        for res in results:
            bad_group_records.extend(res["bad_group_records"])
            duplicated_u_records.extend(res["duplicated_u_records"])
            skipped_group_count += res["skipped_group_count"]

            if res["status"] != "filled":
                continue

            idx = res["grid_index"]
            y_on_grid = res["y_on_grid"]
            bounds = res["bounds"]

            grid_bounds[idx + (slice(None),)] = bounds
            grid_outputs[(slice(None),) + idx + (slice(None),)] = y_on_grid

            filled_group_count += res["filled_group_count"]

        del df

    if read_all_at_once:
        del df_all

    bad_group_df = pd.DataFrame(bad_group_records)
    duplicated_u_df = pd.DataFrame(duplicated_u_records)

    bad_group_path = save_path + f"bad_records_{Path(save_name).stem}.csv"
    duplicated_u_path = save_path + f"duplicated_records_{Path(save_name).stem}.csv"

    bad_group_df.to_csv(bad_group_path, index=False)
    duplicated_u_df.to_csv(duplicated_u_path, index=False)

    # ============================================================
    # save
    # ============================================================
    save_file = save_path + save_name

    save_dict = dict(
        u_grid=u_grid,
        grid_outputs=grid_outputs,
        grid_bounds=grid_bounds,
        interp_output_list=np.array(interp_output_list, dtype=object),
        interp_variable_list=np.array(interp_variable_list, dtype=object),
        interp_variable_list_regular=np.array(interp_variable_list_regular, dtype=object),
        other_interp_variable_list=np.array(other_interp_variable_list, dtype=object),
        grid_type=np.array(grid_type),
        label_list=np.array(label_list),
        save_float32=np.array(save_float32),
        read_all_at_once=np.array(read_all_at_once),
        parallel_file_read=np.array(parallel_file_read),
        parallel_backend=np.array(parallel_backend),
        parallel_interp=np.array(parallel_interp),
        interp_backend=np.array(interp_backend),
        grid_dtype=np.array(str(grid_outputs.dtype)),
        filled_group_count=np.array(filled_group_count),
        skipped_group_count=np.array(skipped_group_count),
        u_save_range=np.array([u_save_min, u_save_max]),
        u_grid_was_cropped=np.array(u_save_range is not None),
        n_u_original=np.array(n_u),
        n_u_saved=np.array(len(u_grid)),
    )

    for c in other_interp_variable_list:
        save_dict[f"{c}_grid"] = axis_grids[c]

    np.savez_compressed(save_file, **save_dict)

    total_bytes = grid_outputs.nbytes + grid_bounds.nbytes + u_grid.nbytes
    for c in other_interp_variable_list:
        total_bytes += axis_grids[c].nbytes

    print("Finished u-grid regularization.")
    print(f"grid_type: {grid_type}")
    print(f"read_all_at_once: {read_all_at_once}")
    print(f"parallel_file_read: {parallel_file_read}")
    print(f"parallel_backend: {parallel_backend}")
    print(f"parallel_interp: {parallel_interp}")
    print(f"interp_backend: {interp_backend}")
    print(f"max_workers: {max_workers}")
    print(f"interp_max_workers: {interp_max_workers}")
    print(f"save_float32: {save_float32}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(f"u_grid shape: {u_grid.shape}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_bounds shape: {grid_bounds.shape}")
    print(f"filled_group_count: {filled_group_count}")
    print(f"skipped_group_count: {skipped_group_count}")
    print(f"bad_group_count: {len(bad_group_df)}")
    print(f"duplicated_u_count: {len(duplicated_u_df)}")
    print(f"bad group records saved at: {os.path.abspath(bad_group_path)}")
    print(f"duplicated u records saved at: {os.path.abspath(duplicated_u_path)}")
    print(f"u_save_range: [{u_save_min}, {u_save_max}]")
    print(f"n_u_original: {n_u}")
    print(f"n_u_saved: {len(u_grid)}")
    print(f"Grids are saved at: {os.path.abspath(save_file)}")
    print(
        f"total grid memory: {total_bytes / 1024**2:.2f} MB "
        f"({total_bytes / 1024**3:.2f} GB)"
    )

    return os.path.abspath(save_file), bad_group_df, duplicated_u_df


def load_regularized_u_interpolator(npz_path, force_float64=False):
    data = np.load(npz_path, allow_pickle=True)

    grid_type = str(data["grid_type"])

    u_grid = data["u_grid"]
    grid_outputs = data["grid_outputs"]
    grid_bounds = data["grid_bounds"]

    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)
        grid_bounds = grid_bounds.astype(np.float64)

    interp_output_list = data["interp_output_list"].astype(str).tolist()
    interp_variable_list = data["interp_variable_list"].astype(str).tolist()
    interp_variable_list_regular = data["interp_variable_list_regular"].astype(str).tolist()
    other_interp_variable_list = data["other_interp_variable_list"].astype(str).tolist()

    param_grids = []
    for c in other_interp_variable_list:
        param_grids.append(data[f"{c}_grid"])

    interp_u_bounds = RegularGridInterpolator(
        tuple(param_grids),
        grid_bounds,
        bounds_error=False,
        fill_value=np.nan,
    )

    interp_u_to_outputs = RegularGridInterpolator(
        tuple([u_grid] + param_grids),
        grid_outputs,
        bounds_error=False,
        fill_value=np.nan,
    )

    n_param = len(interp_variable_list)

    def interp_u(x, eps=1e-12, clip_u=False, return_bounds=False):
        x = np.asarray(x, dtype=np.float64)

        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")
            x = x[None, :]
            single_input = True

        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")

        Mini = x[:, 0]
        p_bounds = x[:, 1:]

        bounds = interp_u_bounds(p_bounds)

        Mini_min_used = bounds[:, 0]
        Mini_max = bounds[:, 1]
        width = Mini_max - Mini_min_used

        u = np.full_like(Mini, np.nan, dtype=np.float64)

        valid = np.isfinite(width) & (width > eps)
        u[valid] = (Mini[valid] - Mini_min_used[valid]) / width[valid]

        if clip_u:
            u = np.clip(u, 0.0, 1.0)

        if single_input:
            u = u[0]
            bounds = bounds[0]

        if return_bounds:
            return u, bounds
        return u

    def interp_func(
        x,
        eps=1e-12,
        clip_u=False,
        return_u=False,
        return_bounds=False,
    ):
        x = np.asarray(x, dtype=np.float64)

        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}), while x.shape[0]={x.shape[0]}")
            x = x[None, :]
            single_input = True

        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f"Input must have shape ({n_param},) or (N, {n_param}).")

        u, bounds = interp_u(
            x,
            eps=eps,
            clip_u=clip_u,
            return_bounds=True,
        )

        if single_input:
            u_eval = np.array([u], dtype=np.float64)
            x_eval = x
        else:
            u_eval = u
            x_eval = x

        p_eval = np.column_stack([u_eval, x_eval[:, 1:]])

        y = interp_u_to_outputs(p_eval)

        if single_input:
            y = y[0]

        if return_u and return_bounds:
            return y, u, bounds
        elif return_u:
            return y, u
        elif return_bounds:
            return y, bounds
        else:
            return y

    total_bytes = grid_outputs.nbytes + grid_bounds.nbytes + u_grid.nbytes
    for g in param_grids:
        total_bytes += g.nbytes

    print("Loaded u-regularized grid.")
    print(f"grid_type: {grid_type}")
    print(f"interp_variable_list: {interp_variable_list}")
    print(f"interp_variable_list_regular: {interp_variable_list_regular}")
    print(f"interp_output_list: {interp_output_list}")
    print(f"u_grid shape: {u_grid.shape}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_bounds shape: {grid_bounds.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(f"grid_bounds dtype: {grid_bounds.dtype}")

    if "read_all_at_once" in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")

    if "save_float32" in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")

    if "u_save_range" in data.files:
        print(f"u_save_range: {data['u_save_range'].tolist()}")

    if "u_grid_was_cropped" in data.files:
        print(f"u_grid_was_cropped: {bool(data['u_grid_was_cropped'])}")

    if "n_u_original" in data.files and "n_u_saved" in data.files:
        print(f"n_u_original: {int(data['n_u_original'])}")
        print(f"n_u_saved: {int(data['n_u_saved'])}")

    print(
        f"total grid memory: {total_bytes / 1024**2:.2f} MB "
        f"({total_bytes / 1024**3:.2f} GB)"
    )

    return (
        interp_func,
        interp_u,
        interp_u_to_outputs,
        interp_u_bounds,
        interp_variable_list,
        interp_output_list,
        data,
    )



def regularize_PARSECv2_u_grid(
    path=None,
    save_path=None,
    save_name=None,
    age_initial="1e6",
    age_end="2001e6",
    age_interval="10e6",
    age_type="linear",
    photometry_suffix="2mass & tycho2 & gaia dr3",
    grid_type="with_inc",
    label_list=None,
    min_width_threshold=0.01,
    save_float32=True,
    read_all_at_once=False,
    **kwargs,
):
    """
    Wrapper for regularization_on_u_from_files().

    Default input:
        DATA_DIR/isochrones/PARSEC_v2/
        1e6_2001e6yr_linear_isochrones_rearranged (2mass & tycho2 & gaia dr3)/*

    Default output:
        DATA_DIR/isochrones/regular_PARSECv2_u_grids/

    Returns
    -------
    save_file, bad_group_df, duplicated_u_df
    """

    if label_list is None:
        label_list = [1]

    if path is None:
        parsec_dir = DATA_DIR / "isochrones" / "PARSEC_v2"

        path = [
            str(
                parsec_dir
                / f"{age_initial}_{age_end}yr_{age_type}_isochrones_rearranged ({photometry_suffix})"
                / "*"
            )
        ]

    elif isinstance(path, (str, Path)):
        path = [_as_glob_pattern_string(path)]

    elif isinstance(path, (list, tuple)):
        path = [_as_glob_pattern_string(p) for p in path]

    else:
        raise ValueError("path must be None, str, Path, list, or tuple.")

    if save_path is None:
        save_path = DATA_DIR / "isochrones" / "regular_PARSECv2_u_grids"

    save_path = _as_dir_string(save_path, create=True)

    if save_name is None:
        label_tag = _label_list_to_tag(label_list)

        save_name = (
            f"regularized_u_grid_"
            f"{age_initial}_{age_end}yr_step{age_interval}_"
            f"{age_type}_isochrones_{label_tag}_{grid_type}.npz"
        )

    save_file, bad_group_df, duplicated_u_df = regularization_on_u_from_files(
        path=path,
        save_path=save_path,
        save_name=save_name,
        grid_type=grid_type,
        label_list=label_list,
        min_width_threshold=min_width_threshold,
        save_float32=save_float32,
        read_all_at_once=read_all_at_once,
        **kwargs,
    )

    return save_file, bad_group_df, duplicated_u_df


def load_default_regularized_PARSECv2_u_interpolator(
    npz_path=None,
    age_initial="1e6",
    age_end="2001e6",
    age_interval="10e6",
    age_type="linear",
    label_tag="MS",
    grid_type="with_inc",
    force_float64=False,
):
    """
    Load default PARSEC v2 u-regularized interpolator.

    Default file:
        DATA_DIR/isochrones/regular_PARSECv2_u_grids/
        regularized_u_grid_1e6_2001e6yr_step10e6_linear_isochrones_MS_with_inc.npz
    """

    if npz_path is None:
        save_dir = DATA_DIR / "isochrones" / "regular_PARSECv2_u_grids"

        save_name = (
            f"regularized_u_grid_"
            f"{age_initial}_{age_end}yr_step{age_interval}_"
            f"{age_type}_isochrones_{label_tag}_{grid_type}.npz"
        )

        npz_path = save_dir / save_name

    npz_path = Path(npz_path).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"Regularized PARSEC v2 u grid not found: {npz_path}")

    return load_regularized_u_interpolator(
        npz_path=npz_path,
        force_float64=force_float64,
    )




def MIST_read_isos(file_path, add_iso_index=True):
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    blocks = []
    current_block = None

    # Split by each small isochrone block
    for line in lines:
        s = line.strip()

        if s.startswith("# number of EEPs, cols"):
            if current_block is not None and len(current_block) > 0:
                blocks.append(current_block)
            current_block = [line]
        else:
            if current_block is not None:
                current_block.append(line)

    if current_block is not None and len(current_block) > 0:
        blocks.append(current_block)

    dfs = []

    for i, block in enumerate(blocks):
        columns = None
        data = []
        data_started = False

        for line in block:
            s = line.strip()

            if not s:
                continue

            # Column name line
            if s.startswith("# EEP"):
                columns = s[1:].strip().split()
                data_started = True
                continue

            if not data_started:
                continue

            if s.startswith("#"):
                continue

            parts = s.split()
            if parts:
                data.append(parts)

        if columns is None or len(data) == 0:
            continue

        df = pd.DataFrame(data, columns=columns)

        # Convert columns to numeric when possible
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

        if add_iso_index:
            df["iso_index"] = i

        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all



def MIST_read_phot(file_path, omega_col="omega_ini", strict_omega=False):
    """
    Read a MIST .iso.cmd photometry file and return a combined DataFrame.

    The function also parses vvcrit from the filename, e.g.
    MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd
    and adds it as the last column named omega_ini.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the MIST .iso.cmd file.

    omega_col : str
        Name of the column used to store vvcrit / omega_ini.

    strict_omega : bool
        If True, raise ValueError when vvcrit cannot be parsed.
        If False, set omega_ini to np.nan when vvcrit cannot be parsed.

    Returns
    -------
    df_all : pandas.DataFrame
        Combined DataFrame from all MIST blocks.
    """
    file_path = Path(file_path)

    # Parse vvcrit value from filename or full path
    match = re.search(r"vvcrit([0-9]*\.?[0-9]+)", str(file_path))
    if match is not None:
        omega_ini = float(match.group(1))
    else:
        if strict_omega:
            raise ValueError(f"Cannot parse vvcrit from file path: {file_path}")
        omega_ini = np.nan

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    blocks = []
    current_block = []

    # Split file into blocks
    for line in lines:
        if line.strip().startswith("# MIST version number"):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        else:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    dfs = []

    # Parse each block
    for block in blocks:
        columns = None
        data = []
        data_started = False

        for line in block:
            s = line.strip()

            if not s:
                continue

            # Column header line
            if s.startswith("# EEP"):
                columns = s[1:].strip().split()
                data_started = True
                continue

            if not data_started:
                continue

            if s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) == 0:
                continue

            # Only keep rows with the expected number of columns
            if columns is not None and len(parts) == len(columns):
                data.append(parts)

        if columns is None or len(data) == 0:
            continue

        df = pd.DataFrame(data, columns=columns)

        # Convert all possible columns to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        dfs.append(df)

    if len(dfs) == 0:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    # Add omega_ini as the last column
    df_all[omega_col] = omega_ini

    return df_all

def MIST_merge_isos_phot(df_isos, df_phot, check_order=True):
    """
    Merge df_isos and df_phot by their common columns, using df_isos as the base.

    Parameters
    ----------
    df_isos : pd.DataFrame
    df_phot : pd.DataFrame
    check_order : bool, default=True
        If True, require the common-column values to be identical row by row.
        If False, merge by common columns regardless of row order.

    Returns
    -------
    df_merged : pd.DataFrame
        df_isos plus extra columns from df_phot appended on the right.
    common_cols : list
        Common columns shared by df_isos and df_phot.
    phot_only_cols : list
        Columns only in df_phot and appended to the result.
    """
    common_cols = [c for c in df_isos.columns if c in df_phot.columns]
    phot_only_cols = [c for c in df_phot.columns if c not in common_cols]

    if len(common_cols) == 0:
        raise ValueError("df_isos and df_phot have no common columns.")

    if check_order:
        if len(df_isos) != len(df_phot):
            raise ValueError(
                f"Row count mismatch: len(df_isos)={len(df_isos)}, len(df_phot)={len(df_phot)}"
            )

        is_same = df_isos[common_cols].equals(df_phot[common_cols])
        if not is_same:
            # help locate first mismatch
            diff_mask = (df_isos[common_cols] != df_phot[common_cols])
            diff_mask = diff_mask.fillna(False)
            bad_rows = diff_mask.any(axis=1)

            first_bad_idx = bad_rows.idxmax() if bad_rows.any() else None
            raise ValueError(
                f"Common columns are not identical row by row. "
                f"First mismatch at row index: {first_bad_idx}"
            )

        df_merged = pd.concat(
            [df_isos.reset_index(drop=True), df_phot[phot_only_cols].reset_index(drop=True)],
            axis=1
        )
    else:
        # merge by common columns, order not required
        if df_phot.duplicated(subset=common_cols).any():
            raise ValueError("df_phot has duplicated rows in common columns; cannot safely merge.")
        if df_isos.duplicated(subset=common_cols).any():
            raise ValueError("df_isos has duplicated rows in common columns; cannot safely merge.")

        df_merged = df_isos.merge(
            df_phot[common_cols + phot_only_cols],
            on=common_cols,
            how="left",
            validate="one_to_one"
        )

    return df_merged#, common_cols, phot_only_cols




# ============================================================
# 1. Parse MIST file names
# ============================================================

def parse_mist_filename(file_path):
    """
    Parse a MIST filename.

    Example:
    MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd
    MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_full.iso

    Returns
    -------
    info : dict
        {
            "path": Path,
            "feh_tag": str,
            "afe_tag": str,
            "vvcrit": float,
            "kind": "phot" or "full",
            "key": tuple,
        }
    """
    file_path = Path(file_path)
    name = file_path.name

    feh_match = re.search(r"feh_([mp][0-9]+(?:\.[0-9]+)?)", name)
    afe_match = re.search(r"afe_([mp][0-9]+(?:\.[0-9]+)?)", name)
    vvcrit_match = re.search(r"vvcrit([0-9]+(?:\.[0-9]+)?)", name)

    if feh_match is None:
        raise ValueError(f"Cannot parse feh tag from filename: {name}")
    if afe_match is None:
        raise ValueError(f"Cannot parse afe tag from filename: {name}")
    if vvcrit_match is None:
        raise ValueError(f"Cannot parse vvcrit from filename: {name}")

    feh_tag = feh_match.group(1)
    afe_tag = afe_match.group(1)
    vvcrit = float(vvcrit_match.group(1))

    if name.endswith("_UBVRIplus.iso.cmd"):
        kind = "phot"
    elif name.endswith("_full.iso"):
        kind = "full"
    else:
        kind = "unknown"

    key = (feh_tag, afe_tag, vvcrit)

    return {
        "path": file_path,
        "feh_tag": feh_tag,
        "afe_tag": afe_tag,
        "vvcrit": vvcrit,
        "kind": kind,
        "key": key,
    }


def find_mist_file_pairs(mist_root_or_folders, verbose=True):
    """
    Find and pair MIST full isochrone files and photometry files.

    Parameters
    ----------
    mist_root_or_folders : str, Path, or list
        Can be:
        1. A root folder containing subfolders.
        2. A list of folders.
        3. A single folder.

    Returns
    -------
    pairs : list of dict
        Each dict contains:
        {
            "key": (feh_tag, afe_tag, vvcrit),
            "full_file": Path,
            "phot_file": Path,
        }
    """
    if isinstance(mist_root_or_folders, (str, Path)):
        mist_root_or_folders = [mist_root_or_folders]

    all_full_files = []
    all_phot_files = []

    for folder in mist_root_or_folders:
        folder = Path(folder)

        if not folder.exists():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        all_full_files += list(folder.rglob("*_full.iso"))
        all_phot_files += list(folder.rglob("*_UBVRIplus.iso.cmd"))

    if len(all_full_files) == 0:
        raise ValueError("No *_full.iso files found.")
    if len(all_phot_files) == 0:
        raise ValueError("No *_UBVRIplus.iso.cmd files found.")

    full_dict = {}
    phot_dict = {}

    for file_path in all_full_files:
        info = parse_mist_filename(file_path)
        if info["kind"] != "full":
            continue
        key = info["key"]

        if key in full_dict:
            raise ValueError(f"Duplicated full file key: {key}")
        full_dict[key] = info["path"]

    for file_path in all_phot_files:
        info = parse_mist_filename(file_path)
        if info["kind"] != "phot":
            continue
        key = info["key"]

        if key in phot_dict:
            raise ValueError(f"Duplicated phot file key: {key}")
        phot_dict[key] = info["path"]

    common_keys = sorted(set(full_dict.keys()) & set(phot_dict.keys()))
    missing_phot = sorted(set(full_dict.keys()) - set(phot_dict.keys()))
    missing_full = sorted(set(phot_dict.keys()) - set(full_dict.keys()))

    if verbose:
        print(f"Found full files: {len(full_dict)}")
        print(f"Found phot files: {len(phot_dict)}")
        print(f"Matched pairs: {len(common_keys)}")
        print(f"Missing phot files: {len(missing_phot)}")
        print(f"Missing full files: {len(missing_full)}")

    if len(common_keys) == 0:
        raise ValueError("No matched MIST full/phot file pairs found.")

    pairs = []
    for key in common_keys:
        pairs.append(
            {
                "key": key,
                "full_file": full_dict[key],
                "phot_file": phot_dict[key],
            }
        )

    return pairs


# ============================================================
# 2. Read one MIST pair and standardize columns
# ============================================================

def standardize_mist_dataframe(df_mist):
    """
    Convert raw merged MIST DataFrame into the standard grid format.

    Keeps only:
        Mini, logAge, MH, omega_ini, Mass, omega, logL, Req

    Returns
    -------
    df_new : pd.DataFrame
    """
    col_map = {
        "initial_mass": "Mini",
        "log10_isochrone_age_yr": "logAge",
        "star_mass": "Mass",
        "surf_avg_omega_div_omega_crit": "omega",
        "[Fe/H]_init": "MH",
        "[Fe/H]": "MH_current",
        "log_L": "logL",
    }

    required_cols = list(col_map.keys()) + ["log_R", "omega_ini"]

    missing_cols = [c for c in required_cols if c not in df_mist.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns in merged MIST DataFrame: {missing_cols}")

    df_new = df_mist[required_cols].copy()
    df_new = df_new.rename(columns=col_map)

    df_new["Req"] = 10.0 ** df_new["log_R"]
    df_new = df_new.drop(columns=["log_R"])

    keep_cols = [
        "Mini",
        "logAge",
        "MH",
        "omega_ini",
        "Mass",
        "omega",
        "logL",
        "Req",
    ]

    df_new = df_new[keep_cols].copy()

    for col in keep_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

    df_new = df_new.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols)
    df_new = df_new.reset_index(drop=True)

    return df_new


def _read_one_mist_pair_for_regularization(args):
    """
    Worker function for reading one MIST full/phot pair.
    """
    pair_i, full_file, phot_file, check_order = args

    df_isos = MIST_read_isos(full_file)
    df_phot = MIST_read_phot(phot_file)
    df_mist = MIST_merge_isos_phot(df_isos, df_phot, check_order=check_order)

    df_new = standardize_mist_dataframe(df_mist)

    df_new["pair_index"] = pair_i
    df_new["full_file"] = str(full_file)
    df_new["phot_file"] = str(phot_file)

    return df_new


# ============================================================
# 3. Process one Mini group
# ============================================================

def _process_one_MIST_Mini_group(args):
    """
    Regularize one group along Mini.

    One group is fixed:
        logAge, MH, omega_ini

    The function interpolates:
        Mass, omega, logL, Req

    onto the global Mini_grid.
    """
    (
        loop_i,
        group_key,
        df_i,
        group_cols,
        compare_cols,
        interp_output_list,
        Mini_grid,
        dtype,
        logAge_to_idx,
        MH_to_idx,
        omega_ini_to_idx,
        pair_i,
        source_info,
    ) = args

    group_info = {col: group_key[k] for k, col in enumerate(group_cols)}

    result = {
        "status": "skipped",
        "problematic_records": [],
        "safe_duplicate_count": 0,
        "filled_group_count": 0,
        "skipped_group_count": 0,
        "grid_index": None,
        "y_grid": None,
    }

    df_i = df_i.sort_values("Mini").reset_index(drop=True)

    if len(df_i) < 2:
        result["skipped_group_count"] += 1
        return result

    interp_x = df_i["Mini"].values
    interp_y = df_i[interp_output_list].values

    # ----------------------------
    # Handle duplicated Mini values
    # ----------------------------
    dup_mask_all = pd.Series(interp_x).duplicated(keep=False).to_numpy()

    if dup_mask_all.any():
        local_Mini_min = np.nanmin(interp_x)
        local_Mini_max = np.nanmax(interp_x)
        local_range = local_Mini_max - local_Mini_min

        dup_groups = df_i.loc[dup_mask_all].groupby("Mini", sort=False)

        for mini_value, g in dup_groups:
            n_unique_rows = g[compare_cols].drop_duplicates().shape[0]

            if n_unique_rows == 1:
                result["safe_duplicate_count"] += len(g) - 1
            else:
                bad_pct = (
                    100.0 * (mini_value - local_Mini_min) / local_range
                    if local_range > 0 else np.nan
                )

                record = {
                    "pair_index": pair_i,
                    "source_info": source_info,
                    "loop_index_in_this_pair": loop_i,
                    "Mini": mini_value,
                    "position_pct": bad_pct,
                    "n_duplicates": len(g),
                    "n_unique_rows_in_compare_cols": n_unique_rows,
                    "local_Mini_min": local_Mini_min,
                    "local_Mini_max": local_Mini_max,
                }

                for col in group_cols:
                    record[col] = group_info[col]

                result["problematic_records"].append(record)

        # Keep first duplicated Mini
        keep_mask = ~pd.Series(interp_x).duplicated(keep="first").to_numpy()
        df_i = df_i.loc[keep_mask].reset_index(drop=True)

        interp_x = df_i["Mini"].values
        interp_y = df_i[interp_output_list].values

    if len(interp_x) < 2:
        result["skipped_group_count"] += 1
        return result

    if not np.all(np.diff(interp_x) > 0):
        record = {
            "pair_index": pair_i,
            "source_info": source_info,
            "loop_index_in_this_pair": loop_i,
            "reason": "Mini is not strictly increasing after duplicate handling",
        }

        for col in group_cols:
            record[col] = group_info[col]

        result["problematic_records"].append(record)
        result["skipped_group_count"] += 1
        return result

    local_Mini_min = np.nanmin(interp_x)
    local_Mini_max = np.nanmax(interp_x)

    y_grid = np.full(
        (len(Mini_grid), len(interp_output_list)),
        np.nan,
        dtype=dtype,
    )

    valid_mask = (
        (Mini_grid >= local_Mini_min) &
        (Mini_grid <= local_Mini_max)
    )

    if np.sum(valid_mask) == 0:
        result["skipped_group_count"] += 1
        return result

    for j, col in enumerate(interp_output_list):
        f = interp1d(
            interp_x,
            interp_y[:, j],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )

        y_grid[valid_mask, j] = f(Mini_grid[valid_mask]).astype(dtype)

    idx_logAge = logAge_to_idx[group_info["logAge"]]
    idx_MH = MH_to_idx[group_info["MH"]]
    idx_omega_ini = omega_ini_to_idx[group_info["omega_ini"]]

    grid_index = (idx_logAge, idx_MH, idx_omega_ini)

    result["status"] = "filled"
    result["filled_group_count"] = 1
    result["grid_index"] = grid_index
    result["y_grid"] = y_grid

    return result


# ============================================================
# 4. Main regularization function
# ============================================================

def regularization_on_Mini_from_MIST_folders(
    mist_root_or_folders,
    save_path=".",
    save_name="regularized_MIST_Mini_grid.npz",
    Mini_setting=(1.0, 5.0, 0.01),
    save_float32=True,
    read_all_at_once=False,
    parallel_file_read=True,
    parallel_backend="process",
    max_workers=None,
    check_order=True,
    local_max_workers=8,
    print_pairs=False,
):
    """
    Build a regular Mini grid from MIST files.

    Parameters
    ----------
    mist_root_or_folders : str, Path, or list
        Root folder or list of folders containing MIST full_isos and UBVRIplus files.

    save_path : str or Path
        Output folder.

    save_name : str
        Output npz filename.

    Mini_setting : tuple
        (Mini_min, Mini_max, Mini_step). Default is (1.0, 5.0, 0.01).

    save_float32 : bool
        If True, save grid_outputs as float32. Default True.

    read_all_at_once : bool
        If True, read all matched pairs into one large DataFrame first.
        If False, read one pair at a time.

    parallel_file_read : bool
        If True and read_all_at_once=True, read pairs in parallel.

    parallel_backend : {"process", "thread"}
        Backend for parallel reading.

    max_workers : int or None
        Number of workers for parallel reading.

    check_order : bool
        Passed to MIST_merge_isos_phot.

    local_max_workers : int
        Number of threads for processing groups within each file/pair.

    print_pairs : bool
        If True, print matched file pairs.

    Returns
    -------
    save_file : str
        Saved npz path.

    problematic_df : pd.DataFrame
        Problematic duplicated Mini records.
    """
    if not read_all_at_once and parallel_file_read:
        print("[info] parallel_read is ignored because read_all_at_once=False.")

    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    dtype = np.float32 if save_float32 else np.float64

    interp_variable_list = ["Mini", "logAge", "MH", "omega_ini"]
    group_cols = ["logAge", "MH", "omega_ini"]
    interp_output_list = ["Mass", "omega", "logL", "Req"]

    in_cols = interp_variable_list
    compare_cols = in_cols + interp_output_list

    # ----------------------------
    # Find and pair files
    # ----------------------------
    pairs = find_mist_file_pairs(mist_root_or_folders, verbose=True)

    if print_pairs:
        for i, pair in enumerate(pairs):
            print(f"[{i}] key={pair['key']}")
            print(f"    full: {pair['full_file']}")
            print(f"    phot: {pair['phot_file']}")

    # ----------------------------
    # Mini grid
    # ----------------------------
    if Mini_setting is None:
        Mini_setting = (1.0, 5.0, 0.01)

    if not isinstance(Mini_setting, (list, tuple, np.ndarray)) or len(Mini_setting) != 3:
        raise ValueError("Mini_setting must be (Mini_min, Mini_max, Mini_step).")

    Mini_min, Mini_max, Mini_step = Mini_setting

    if Mini_step <= 0:
        raise ValueError("Mini_step must be positive.")

    n_Mini = int(np.floor((Mini_max - Mini_min) / Mini_step + 0.5)) + 1
    Mini_grid = Mini_min + np.arange(n_Mini) * Mini_step

    # Avoid tiny floating point artifacts
    Mini_grid = np.round(Mini_grid, 10)

    # Make sure the last point does not exceed Mini_max too much
    Mini_grid = Mini_grid[Mini_grid <= Mini_max + 1e-10]

    df_all = None

    # ----------------------------
    # Option A: read all pairs first
    # ----------------------------
    if read_all_at_once:
        print("Reading all MIST pairs at once...")

        read_args = [
            (
                pair_i,
                pairs[pair_i]["full_file"],
                pairs[pair_i]["phot_file"],
                check_order,
            )
            for pair_i in range(len(pairs))
        ]

        if parallel_file_read:
            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(read_args))

            if parallel_backend == "process":
                Executor = ProcessPoolExecutor
            elif parallel_backend == "thread":
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")

            with Executor(max_workers=max_workers) as executor:
                df_list = list(
                    tqdm(
                        executor.map(_read_one_mist_pair_for_regularization, read_args),
                        total=len(read_args),
                    )
                )
        else:
            df_list = []
            for args in tqdm(read_args):
                df_list.append(_read_one_mist_pair_for_regularization(args))

        df_all = pd.concat(df_list, ignore_index=True)
        del df_list

        print(f"df_all shape: {df_all.shape}")
        print(
            f"df_all memory: {df_all.memory_usage(deep=True).sum() / 1024**2:.2f} MB "
            f"({df_all.memory_usage(deep=True).sum() / 1024**3:.2f} GB)"
        )

    # ----------------------------
    # Collect axis values
    # ----------------------------
    print("Collecting axis values...")

    logAge_list = []
    MH_list = []
    omega_ini_list = []

    if read_all_at_once:
        df = df_all

        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in df_all: {missing_cols}")

        logAge_list.append(df["logAge"].dropna().unique())
        MH_list.append(df["MH"].dropna().unique())
        omega_ini_list.append(df["omega_ini"].dropna().unique())

    else:
        for pair_i, pair in enumerate(tqdm(pairs)):
            df = _read_one_mist_pair_for_regularization(
                (
                    pair_i,
                    pair["full_file"],
                    pair["phot_file"],
                    check_order,
                )
            )

            missing_cols = [c for c in compare_cols if c not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(
                    f"Missing columns in pair {pair_i}: {missing_cols}"
                )

            logAge_list.append(df["logAge"].dropna().unique())
            MH_list.append(df["MH"].dropna().unique())
            omega_ini_list.append(df["omega_ini"].dropna().unique())

            del df

    logAge_grid = np.sort(np.unique(np.concatenate(logAge_list)))
    MH_grid = np.sort(np.unique(np.concatenate(MH_list)))
    omega_ini_grid = np.sort(np.unique(np.concatenate(omega_ini_list)))

    logAge_to_idx = {v: i for i, v in enumerate(logAge_grid)}
    MH_to_idx = {v: i for i, v in enumerate(MH_grid)}
    omega_ini_to_idx = {v: i for i, v in enumerate(omega_ini_grid)}

    # ----------------------------
    # Initialize output grid
    # ----------------------------
    grid_outputs = np.full(
        (
            len(Mini_grid),
            len(logAge_grid),
            len(MH_grid),
            len(omega_ini_grid),
            len(interp_output_list),
        ),
        np.nan,
        dtype=dtype,
    )

    print(f"Mini_grid: [min: {np.min(Mini_grid)}, max: {np.max(Mini_grid)}]")
    print(f"Mini_grid shape: {Mini_grid.shape}")
    print(f"logAge_grid shape: {logAge_grid.shape}")
    print(f"MH_grid shape: {MH_grid.shape}")
    print(f"omega_ini_grid shape: {omega_ini_grid.shape}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(
        f"grid_outputs memory: {grid_outputs.nbytes / 1024**2:.2f} MB "
        f"({grid_outputs.nbytes / 1024**3:.2f} GB)"
    )

    # ----------------------------
    # Fill regular grids
    # ----------------------------
    print("Filling regular grids...")

    problematic_records = []
    safe_duplicate_count = 0
    filled_group_count = 0
    skipped_group_count = 0

    if read_all_at_once:
        file_iter = [(-1, None)]
    else:
        file_iter = tqdm([(pair_i, pairs[pair_i]) for pair_i in range(len(pairs))])

    for pair_i, pair in file_iter:
        if read_all_at_once:
            df = df_all
            source_info = "df_all"
        else:
            df = _read_one_mist_pair_for_regularization(
                (
                    pair_i,
                    pair["full_file"],
                    pair["phot_file"],
                    check_order,
                )
            )
            source_info = f"full={pair['full_file']}; phot={pair['phot_file']}"

        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns during filling: {missing_cols}")

        grouped = list(df.groupby(group_cols, sort=False))

        task_args = []

        for loop_i, (group_key, df_i) in enumerate(grouped):
            if len(group_cols) == 1:
                group_key = (group_key,)

            task_args.append(
                (
                    loop_i,
                    group_key,
                    df_i,
                    group_cols,
                    compare_cols,
                    interp_output_list,
                    Mini_grid,
                    dtype,
                    logAge_to_idx,
                    MH_to_idx,
                    omega_ini_to_idx,
                    pair_i,
                    source_info,
                )
            )

        with ThreadPoolExecutor(max_workers=local_max_workers) as executor:
            iterator = executor.map(_process_one_MIST_Mini_group, task_args)

            if read_all_at_once:
                iterator = tqdm(iterator, total=len(task_args))

            results = list(iterator)

        for res in results:
            problematic_records.extend(res["problematic_records"])
            safe_duplicate_count += res["safe_duplicate_count"]
            skipped_group_count += res["skipped_group_count"]

            if res["status"] != "filled":
                continue

            y_grid = res["y_grid"]
            idx_logAge, idx_MH, idx_omega_ini = res["grid_index"]

            grid_outputs[:, idx_logAge, idx_MH, idx_omega_ini, :] = y_grid

            filled_group_count += res["filled_group_count"]

        if not read_all_at_once:
            del df

    if read_all_at_once:
        del df_all

    # ----------------------------
    # Save
    # ----------------------------
    problematic_df = pd.DataFrame(problematic_records)

    save_file = save_path / save_name
    problematic_path = save_path / "problematic_duplicated_Mini_records_MIST.csv"

    save_dict = dict(
        Mini_grid=Mini_grid,
        logAge_grid=logAge_grid,
        MH_grid=MH_grid,
        omega_ini_grid=omega_ini_grid,
        grid_outputs=grid_outputs,
        in_cols=np.array(in_cols, dtype=object),
        interp_variable_list=np.array(interp_variable_list, dtype=object),
        interp_output_list=np.array(interp_output_list, dtype=object),
        group_cols=np.array(group_cols, dtype=object),
        grid_type=np.array("MIST"),
        save_float32=np.array(save_float32),
        read_all_at_once=np.array(read_all_at_once),
        grid_dtype=np.array(str(grid_outputs.dtype)),
        safe_duplicate_count=np.array(safe_duplicate_count),
        filled_group_count=np.array(filled_group_count),
        skipped_group_count=np.array(skipped_group_count),
    )

    np.savez_compressed(save_file, **save_dict)
    problematic_df.to_csv(problematic_path, index=False)

    total_bytes = (
        grid_outputs.nbytes
        + Mini_grid.nbytes
        + logAge_grid.nbytes
        + MH_grid.nbytes
        + omega_ini_grid.nbytes
    )

    print("Finished MIST regularization.")
    print(f"save_float32: {save_float32}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")
    print(f"filled_group_count: {filled_group_count}")
    print(f"skipped_group_count: {skipped_group_count}")
    print(f"safe_duplicate_count: {safe_duplicate_count}")
    print(f"problematic_duplicate_count: {len(problematic_df)}")
    print(f"problematic duplicated Mini records are saved at: {os.path.abspath(problematic_path)}")
    print(f"Grids are saved at: {os.path.abspath(save_file)}")
    print(
        f"total grid memory: {total_bytes / 1024**2:.2f} MB "
        f"({total_bytes / 1024**3:.2f} GB)"
    )

    return os.path.abspath(save_file), problematic_df


# ============================================================
# 5. Loader
# ============================================================

def load_regularized_MIST_interpolator(npz_path, force_float64=False):
    """
    Load regularized MIST Mini grid and return a RegularGridInterpolator.

    Input order:
        Mini, logAge, MH, omega_ini

    Output order:
        Mass, omega, logL, Req
    """
    data = np.load(npz_path, allow_pickle=True)

    grid_type = str(data["grid_type"])
    if grid_type != "MIST":
        raise ValueError(f"This does not look like a MIST grid. grid_type={grid_type}")

    Mini_grid = data["Mini_grid"]
    logAge_grid = data["logAge_grid"]
    MH_grid = data["MH_grid"]
    omega_ini_grid = data["omega_ini_grid"]

    grid_outputs = data["grid_outputs"]

    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)

    interp_variable_list = data["interp_variable_list"].astype(str).tolist()
    interp_output_list = data["interp_output_list"].astype(str).tolist()

    points = (
        Mini_grid,
        logAge_grid,
        MH_grid,
        omega_ini_grid,
    )

    interp_func = RegularGridInterpolator(
        points,
        grid_outputs,
        bounds_error=False,
        fill_value=np.nan,
    )

    total_grid_bytes = (
        grid_outputs.nbytes
        + Mini_grid.nbytes
        + logAge_grid.nbytes
        + MH_grid.nbytes
        + omega_ini_grid.nbytes
    )

    print("Loaded regularized MIST Mini grid.")
    print(f"grid_type: {grid_type}")
    print(f"interp_variable_list: {interp_variable_list}")
    print(f"interp_output_list: {interp_output_list}")
    print(f"grid_outputs shape: {grid_outputs.shape}")
    print(f"grid_outputs dtype: {grid_outputs.dtype}")

    if "save_float32" in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")

    if "read_all_at_once" in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")

    print(
        f"total grid memory: {total_grid_bytes / 1024**2:.2f} MB "
        f"({total_grid_bytes / 1024**3:.2f} GB)"
    )

    return (
        interp_func,
        interp_variable_list,
        interp_output_list,
        data,
    )

def regularize_MIST_Mini_grid(
    mist_root_or_folders=None,
    save_path=None,
    save_name=None,
    Mini_setting=(1.0, 5.0, 0.001),
    save_float32=True,
    read_all_at_once=False,
    parallel_file_read=True,
    parallel_backend="process",
    max_workers=None,
    check_order=True,
    local_max_workers=8,
    print_pairs=False,
):
    """
    Wrapper for regularization_on_Mini_from_MIST_folders().

    MIST does not provide inclination grids, so only METHOD PARS is available.

    Default input:
        DATA_DIR/isochrones/MIST

    Default output:
        DATA_DIR/isochrones/regular_MIST_Mini_grids/regularized_MIST_Mini_grid.npz

    Returns
    -------
    save_file : str
    problematic_df : pandas.DataFrame
    """

    if mist_root_or_folders is None:
        mist_root_or_folders = DATA_DIR / "isochrones" / "MIST"

    if save_path is None:
        save_path = DATA_DIR / "isochrones" / "regular_MIST_Mini_grids"

    if save_name is None:
        save_name = "regularized_MIST_Mini_grid.npz"

    mist_root_or_folders = Path(mist_root_or_folders).expanduser().resolve()
    save_path = Path(save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    save_file, problematic_df = regularization_on_Mini_from_MIST_folders(
        mist_root_or_folders=mist_root_or_folders,
        save_path=save_path,
        save_name=save_name,
        Mini_setting=Mini_setting,
        save_float32=save_float32,
        read_all_at_once=read_all_at_once,
        parallel_file_read=parallel_file_read,
        parallel_backend=parallel_backend,
        max_workers=max_workers,
        check_order=check_order,
        local_max_workers=local_max_workers,
        print_pairs=print_pairs,
    )

    return save_file, problematic_df

from pathlib import Path

from belfast.paths import DATA_DIR


def load_default_regularized_MIST_Mini_interpolator(
    npz_path=None,
    save_path=None,
    save_name=None,
    force_float64=False,
):
    """
    Load the default regularized MIST Mini interpolator.

    Default file:
        DATA_DIR/isochrones/regular_MIST_Mini_grids/regularized_MIST_Mini_grid.npz

    Returns
    -------
    interp_func
    interp_variable_list
    interp_output_list
    data
    """

    if npz_path is None:
        if save_path is None:
            save_path = DATA_DIR / "isochrones" / "regular_MIST_Mini_grids"

        if save_name is None:
            save_name = "regularized_MIST_Mini_grid.npz"

        npz_path = Path(save_path).expanduser().resolve() / save_name

    else:
        npz_path = Path(npz_path).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(f"Regularized MIST Mini grid not found: {npz_path}")

    return load_regularized_MIST_interpolator(
        npz_path=npz_path,
        force_float64=force_float64,
    )