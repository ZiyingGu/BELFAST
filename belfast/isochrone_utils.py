import gzip
import os
import re
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from io import StringIO
from pathlib import Path
from datetime import datetime
import astropy.units as u
import astropy.constants as ct
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d, RegularGridInterpolator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from belfast.paths import DATA_DIR, ISOCHRONE_DIR, KLD_DIR

def _find_PARSECv2_rearranged_paths(
    parsec_dir=None,
    photometry_suffix='2mass_tycho2_gaia_dr3',
):
    """Find all rearranged PARSEC v2 isochrone folders."""
    if parsec_dir is None:
        parsec_dir = ISOCHRONE_DIR / 'PARSEC_v2'
    parsec_dir = Path(parsec_dir).expanduser().resolve()
    folders = [
        p for p in parsec_dir.glob(
            f'*_isochrones_rearranged_{photometry_suffix}'
        )
        if p.is_dir()
    ]
    if not folders:
        raise FileNotFoundError(
            f'No rearranged PARSEC v2 folders found in:\n{parsec_dir}'
        )
    # Sort by starting age, e.g. 1e6 before 21e6
    def start_age(folder):
        try:
            return float(folder.name.split('_', 1)[0])
        except ValueError:
            return np.inf
    folders.sort(key=start_age)
    return [str(folder / '*') for folder in folders]

def _open_text_maybe_gz(path):
    path = Path(path)
    if path.suffix == '.gz':
        return gzip.open(path, 'rt')
    else:
        return open(path, 'r')

def _smart_open(path, mode='wt'):
    path = str(path)
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    else:
        return open(path, mode)

def _assert_same_omega(omega_list, file_list=None, atol=1e-12):
    """
    Check that all omega_ini values exist and are identical.
    """
    if any((omega is None for omega in omega_list)):
        missing = []
        for i, omega in enumerate(omega_list):
            if omega is None:
                if file_list is None:
                    missing.append(str(i))
                else:
                    missing.append(str(file_list[i]))
        raise ValueError('omega_i was not found in the following file(s):\n' + '\n'.join(missing))
    omega_ref = omega_list[0]
    for i, omega in enumerate(omega_list[1:], start=1):
        if not np.isclose(omega_ref, omega, rtol=0.0, atol=atol):
            if file_list is None:
                file_info = ''
            else:
                file_info = f'\nReference file: {file_list[0]}\nCurrent file:   {file_list[i]}'
            raise ValueError(f'omega_i mismatch!\nomega_i[0] = {omega_ref}\nomega_i[{i}] = {omega}{file_info}')
    return omega_ref

def read_isochrones(iso_file, show_progress=True, return_omega_ini=False, return_metadata=False):
    """
    Reads isochrone data from a .dat or .dat.gz file.

    Args:
        iso_file (str or os.PathLike): Path to the isochrone file.
        show_progress (bool): Whether to display a tqdm progress bar.
        return_omega_ini (bool): Whether to return the initial rotation rate
            separately. This option is kept for backward compatibility.
        return_metadata (bool): Whether to return metadata instead of only
            the isochrone data. The metadata dictionary contains the following
            keys:
            - "path": Path to the isochrone file.
            - "omega_ini": Initial rotation rate, if available.
            - "header_comments": Comments extracted from the file header.

    Returns:
        list[pd.DataFrame]: A list of isochrone data tables. If
            ``return_metadata`` is True, returns a tuple of
            ``(results, metadata)``. Otherwise, if ``return_omega_ini`` is
            True, returns ``(results, omega_ini)``.
    """
    iso_file = Path(iso_file)
    omega_ini = None
    header_comments = []
    with _open_text_maybe_gz(iso_file) as f:
        total_lines = sum((1 for _ in f))
    results = []
    data_list = []
    columns = None
    n_isochrones = 0
    reading_file_header = True
    with _open_text_maybe_gz(iso_file) as f:
        if show_progress:
            iterable = tqdm(f, desc=f'Reading {iso_file.name}', total=total_lines)
        else:
            iterable = f
        for line in iterable:
            stripped = line.rstrip('\n')
            if reading_file_header and stripped.startswith('#'):
                if 'Zini' in stripped:
                    reading_file_header = False
                else:
                    header_comments.append(stripped)
            if omega_ini is None and 'omega_i' in line:
                match = re.search('omega_i\\s*=\\s*([-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?)', line)
                if match:
                    omega_ini = float(match.group(1))
            if 'isochrone terminated' in line:
                if columns is not None and data_list:
                    df = pd.DataFrame(data_list, columns=columns).astype('float64')
                    results.append(df)
                break
            if '#' in line and 'Zini' not in line:
                continue
            elif 'Zini' in line:
                if columns is not None and data_list:
                    df = pd.DataFrame(data_list, columns=columns).astype('float64')
                    results.append(df)
                n_isochrones += 1
                data_list = []
                columns = re.split('\\s+', line.strip('#').strip())
            else:
                if columns is None:
                    continue
                data_row = re.split('\\s+', line.strip())
                if len(data_row) == len(columns):
                    data_list.append(data_row)
    if show_progress:
        print('The number of isochrones:', n_isochrones)
    metadata = {'path': str(iso_file), 'omega_ini': omega_ini, 'header_comments': header_comments}
    if return_metadata:
        return (results, metadata)
    elif return_omega_ini:
        return (results, omega_ini)
    else:
        return results

def rearrange_isochrones(iso_paths_list, selected_bands_list, saved_folder_dir, name_suffix=None, n_common=35):
    """
    Rearranges and merges multiple sets of isochrone files.

    Args:
        iso_paths_list (list[str]): Glob patterns for input isochrone series.
        selected_bands_list (list[list[str]]): Bands to select from each
            input series.
        saved_folder_dir (str or os.PathLike): Output directory.
        name_suffix (str, optional): Suffix added to the output directory
            and file names.
        n_common (int): Number of leading columns that must be identical.

    Raises:
        ValueError: If the input series have inconsistent file or block
            counts, or if fewer than two series are provided.
        AssertionError: If the first `n_common` columns differ.
        KeyError: If a selected column is missing.

    Notes:
        Source metadata and comments are preserved, and output columns are
        formatted with fixed widths.
    """
    if len(iso_paths_list) < 2:
        raise ValueError('At least two isochrone series are required.')
    if len(iso_paths_list) != len(selected_bands_list):
        raise ValueError('The number of iso_paths_list entries must equal the number of selected_bands_list entries.')
    saved_folder_dir = str(saved_folder_dir)
    if name_suffix is not None:
        clean_suffix = name_suffix.replace(' ', '_')
        if not saved_folder_dir.endswith('_' + clean_suffix):
            saved_folder_dir += '_' + clean_suffix
    else:
        clean_suffix = None
    os.makedirs(saved_folder_dir, exist_ok=True)
    iso_file_list = []
    for iso_path in iso_paths_list:
        files = glob(iso_path)
        files.sort()
        iso_file_list.append(files)
    file_counts = [len(files) for files in iso_file_list]
    if len(set(file_counts)) != 1:
        raise ValueError(f'All isochrone series must contain the same number of files.\nFound file counts: {file_counts}')
    if file_counts[0] == 0:
        raise ValueError('No isochrone files were found.')
    selected_columns_list = []
    for selected_bands in selected_bands_list:
        selected_columns = []
        for band in selected_bands:
            selected_columns += [band + '_fSB', band + '_f0', band + '_fk']
            selected_columns += [band + '_i' + f'{inc:02d}' for inc in np.linspace(0, 90, 19, dtype=int)]
        selected_columns_list.append(selected_columns)
    for iso_file_index in range(len(iso_file_list[0])):
        paired_files = [file_series[iso_file_index] for file_series in iso_file_list]
        filename = os.path.basename(paired_files[0])
        if filename.endswith('.dat.gz'):
            file_name_body = filename[:-len('.dat.gz')]
            file_name_ext = '.dat.gz'
        elif filename.endswith('.dat'):
            file_name_body = filename[:-len('.dat')]
            file_name_ext = '.dat'
        else:
            file_name_body, file_name_ext = os.path.splitext(filename)
        file_name_body = re.sub('[\\s_]*\\([^)]*\\)$', '', file_name_body).strip()
        photometric_suffixes = ['_2mass_tycho2_gaia_dr3', '_gaia_dr3_2mass_tycho2', '_2mass_tycho2_gaia_edr3', '_gaia_edr3_2mass_tycho2', '_2mass_tycho2', '_gaia_dr3', '_gaia_edr3', '_gaia_dr2']
        suffix_removed = True
        while suffix_removed:
            suffix_removed = False
            for suffix in photometric_suffixes:
                if file_name_body.endswith(suffix):
                    file_name_body = file_name_body[:-len(suffix)]
                    suffix_removed = True
                    break
        if clean_suffix is not None:
            file_name_body += '_' + clean_suffix
        final_file_name = file_name_body + file_name_ext
        out_path = os.path.join(saved_folder_dir, final_file_name)
        print()
        print(f'Now rearranging: {out_path}')
        isos_list = []
        metadata_list = []
        for path in paired_files:
            isos, metadata = read_isochrones(path, show_progress=False, return_metadata=True)
            isos_list.append(isos)
            metadata_list.append(metadata)
        omega_list = [metadata['omega_ini'] for metadata in metadata_list]
        omega_ini = _assert_same_omega(omega_list, file_list=paired_files)
        print(f'    omega_i = {omega_ini:g} ✓')
        block_counts = [len(isos) for isos in isos_list]
        if len(set(block_counts)) != 1:
            raise ValueError(f'Number of isochrone blocks differs between paired files.\nFiles:\n' + '\n'.join(paired_files) + f'\nBlock counts: {block_counts}')
        df_merged_list = []
        for block_index, df_tuple in enumerate(zip(*isos_list)):
            reference_common = df_tuple[0].iloc[:, :n_common]
            for series_index in range(1, len(df_tuple)):
                try:
                    pd.testing.assert_frame_equal(reference_common, df_tuple[series_index].iloc[:, :n_common])
                except AssertionError as e:
                    raise AssertionError(f'First {n_common} columns are not identical.\nBlock index: {block_index}\nReference file:\n    {paired_files[0]}\nCompared file:\n    {paired_files[series_index]}') from e
            common_part = df_tuple[0].iloc[:, :n_common].copy()
            df_part_list = [common_part]
            for series_index in range(len(df_tuple)):
                selected_columns = selected_columns_list[series_index]
                missing_columns = [col for col in selected_columns if col not in df_tuple[series_index].columns]
                if missing_columns:
                    raise KeyError(f'Missing columns in:\n{paired_files[series_index]}\n\n{missing_columns}')
                df_part_list.append(df_tuple[series_index][selected_columns].copy())
            df_merged = pd.concat(df_part_list, axis=1)
            df_merged = df_merged.copy()
            duplicated_columns = df_merged.columns[df_merged.columns.duplicated()].unique().tolist()
            if duplicated_columns:
                raise ValueError(f'Duplicated columns after merging:\n{duplicated_columns}')
            df_merged_list.append(df_merged)
        formatted_blocks = []
        for df in df_merged_list:
            formatted_data = {col: df[col].map(lambda x: 'nan' if pd.isna(x) else f'{x:.6g}') for col in df.columns}
            df_str = pd.DataFrame(formatted_data, index=df.index)
            formatted_blocks.append(df_str)
        all_columns = list(df_merged_list[0].columns)
        column_widths = {}
        for col in all_columns:
            max_width = len(col)
            for df_str in formatted_blocks:
                if len(df_str) == 0:
                    continue
                value_width = df_str[col].str.len().max()
                if pd.notna(value_width):
                    max_width = max(max_width, int(value_width))
            column_widths[col] = max_width
        with _smart_open(out_path, 'wt') as f:
            now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            f.write(f'# File rearranged at {now_utc}\n')
            f.write(f'# omega_i = {omega_ini:g}\n')
            f.write('#\n')
            for series_index, metadata in enumerate(metadata_list):
                f.write('# ============================================================\n')
                f.write(f'# Source {series_index + 1}\n')
                f.write(f'# Source file: {paired_files[series_index]}\n')
                f.write('# Selected bands: ' + ', '.join(selected_bands_list[series_index]) + '\n')
                f.write(f"# Source omega_i = {metadata['omega_ini']:g}\n")
                f.write('# ------------------------------------------------------------\n')
                for comment in metadata['header_comments']:
                    if 'isochrone terminated' in comment:
                        continue
                    f.write(comment.rstrip() + '\n')
                f.write('# ============================================================\n')
                f.write('#\n')
            for df_str in formatted_blocks:
                header_line = '# ' + ' '.join((f'{col:<{column_widths[col]}}' for col in all_columns))
                f.write(header_line.rstrip() + '\n')
                for row in df_str.itertuples(index=False, name=None):
                    line = ' '.join((f'{value:>{column_widths[col]}}' for col, value in zip(all_columns, row)))
                    f.write(line.rstrip() + '\n')
            f.write('#isochrone terminated\n')
        print('    merged successfully ✓')
    print()
    print('Done! All isochrone files were rearranged.')

def check_rearranged_isochrones(rearranged_iso_path, original_iso_path_list, selected_bands_list, n_common=35, verbose=True):
    """
    Validates rearranged isochrone files against the original files.

    Args:
        rearranged_iso_path (str): Glob pattern for rearranged isochrone files.
        original_iso_path_list (list[str]): Glob patterns for original
            isochrone series.
        selected_bands_list (list[list[str]]): Bands to validate for each
            original series.
        n_common (int): Number of leading columns to validate.
        verbose (bool): Whether to print detailed progress information.

    Raises:
        ValueError: If files are missing or file counts do not match.
        AssertionError: If omega_i, isochrone blocks, common columns, or
            selected photometric columns do not match.
    """
    print('Starting rearranged isochrone checks...')
    selected_columns_list = []
    for selected_bands in selected_bands_list:
        selected_columns = []
        for band in selected_bands:
            selected_columns += [band + '_fSB', band + '_f0', band + '_fk']
            selected_columns += [band + '_i' + f'{inc:02d}' for inc in np.linspace(0, 90, 19, dtype=int)]
        selected_columns_list.append(selected_columns)
    rearranged_iso = glob(rearranged_iso_path)
    rearranged_iso.sort()
    if len(rearranged_iso) == 0:
        raise ValueError('No rearranged files found.')
    original_iso_list = []
    for pattern in original_iso_path_list:
        files = glob(pattern)
        files.sort()
        original_iso_list.append(files)
    counts = [len(rearranged_iso)] + [len(files) for files in original_iso_list]
    if len(set(counts)) != 1:
        raise ValueError(f'File counts do not match.\nRearranged: {len(rearranged_iso)}\nOriginal series: {[len(x) for x in original_iso_list]}')
    for iso_file_index in range(len(rearranged_iso)):
        rearranged_file = rearranged_iso[iso_file_index]
        original_files = [original_iso_list[series_index][iso_file_index] for series_index in range(len(original_iso_path_list))]
        if verbose:
            print('\n========================================')
            print(f'Checking:\n    {rearranged_file}')
        rearranged_isos, rearranged_metadata = read_isochrones(rearranged_file, show_progress=False, return_metadata=True)
        original_isos_list = []
        original_metadata_list = []
        for original_file in original_files:
            if verbose:
                print(f'    original: {original_file}')
            isos, metadata = read_isochrones(original_file, show_progress=False, return_metadata=True)
            original_isos_list.append(isos)
            original_metadata_list.append(metadata)
        original_omega_list = [metadata['omega_ini'] for metadata in original_metadata_list]
        omega_ini = _assert_same_omega(original_omega_list, file_list=original_files)
        rearranged_omega = rearranged_metadata['omega_ini']
        if rearranged_omega is None:
            raise AssertionError(f'omega_i is missing from the rearranged file:\n{rearranged_file}')
        if not np.isclose(rearranged_omega, omega_ini, rtol=0.0, atol=1e-12):
            raise AssertionError(f'omega_i mismatch between rearranged and original files!\nRearranged: {rearranged_omega}\nOriginal:   {omega_ini}\nFile: {rearranged_file}')
        if verbose:
            print(f'    omega_i = {omega_ini:g} ✓')
        block_counts = [len(rearranged_isos)] + [len(isos) for isos in original_isos_list]
        if len(set(block_counts)) != 1:
            raise AssertionError(f'Number of isochrone blocks does not match.\nCounts: {block_counts}\nFile: {rearranged_file}')
        for block_index in range(len(rearranged_isos)):
            rearranged_df = rearranged_isos[block_index]
            for series_index in range(len(original_isos_list)):
                original_df = original_isos_list[series_index][block_index]
                try:
                    pd.testing.assert_frame_equal(rearranged_df.iloc[:, :n_common], original_df.iloc[:, :n_common], rtol=5e-6, atol=1e-8)
                    pd.testing.assert_frame_equal(rearranged_df[selected_columns_list[series_index]], original_df[selected_columns_list[series_index]], rtol=5e-6, atol=1e-8)
                except AssertionError as e:
                    raise AssertionError(f'NOT identical!\nBlock: {block_index}\nRearranged: {rearranged_file}\nOriginal: {original_files[series_index]}') from e
        if verbose:
            print('    common columns ✓')
            print('    selected bands ✓')
            print('    Identical!')
    print('\nDone! All rearranged isochrone files were correctly combined!')

def restructure_inclination(df):
    """
    Restructures inclination-dependent columns into a long-format table.

    Args:
        df (pandas.DataFrame): Input wide-format dataframe with inclination
            columns such as ``G_RP_i00`` and ``G_RP_i05``.

    Returns:
        pandas.DataFrame: Restructured dataframe with ``Inclination_deg`` and
            ``Inclination`` columns and one column per passband.
    """
    df = df.copy()
    inc_cols = [c for c in df.columns if re.search('_i\\d\\d$', c)]
    bands = sorted(set((c.rsplit('_i', 1)[0] for c in inc_cols)))
    inc_vals = sorted(set((int(c.split('_i')[1]) for c in inc_cols)))
    base_cols = [c for c in df.columns if c not in inc_cols]
    rows = []
    for inc in inc_vals:
        tmp = df[base_cols].copy()
        tmp['Inclination_deg'] = inc
        tmp['Inclination'] = np.round(np.deg2rad(inc), 3)
        for band in bands:
            col = f'{band}_i{inc:02d}'
            if col in df.columns:
                tmp[band] = df[col]
        rows.append(tmp)
    new_df = pd.concat(rows, ignore_index=True)
    return new_df

def add_iso_inc_rot_age(df_iso_list, omega_ini):
    """
    Adds inclination, rotation, and age-related quantities to isochrones.

    Args:
        df_iso_list (list[pandas.DataFrame]): List of isochrone dataframes.
        omega_ini (float): Initial rotation rate.

    Returns:
        pandas.DataFrame: Combined dataframe with inclination, v sin(i),
            omega_ini, omega_k, and age columns.
    """
    df_inc_list = []
    for df_i in df_iso_list:
        df_inc = restructure_inclination(df_i)
        idx_vtaneq = df_inc.columns.get_loc('vtaneq')
        vsini = df_inc['vtaneq'].values * np.sin(df_inc['Inclination'].values)
        df_inc.insert(idx_vtaneq + 1, 'vsini', vsini)
        true_angular_velocity = df_inc['vtaneq'].values * 1000 * u.m / u.s / (ct.R_sun * df_inc['Req'].values)
        keplerian_angular_velocity = (df_inc['Mass'].values * ct.GM_sun / (ct.R_sun * df_inc['Req'].values) ** 3) ** 0.5
        omega_k = true_angular_velocity / keplerian_angular_velocity
        idx_omega = df_inc.columns.get_loc('omega')
        df_inc.insert(idx_omega, 'omega_ini', omega_ini)
        idx_omega = df_inc.columns.get_loc('omega')
        df_inc.insert(idx_omega + 1, 'omega_k', omega_k)
        idx_logAge = df_inc.columns.get_loc('logAge')
        age = np.round(10.0 ** (df_inc['logAge'].to_numpy() - 6.0), 8)
        df_inc.insert(idx_logAge + 1, 'Age', age)
        df_inc_list.append(df_inc)
    df_inc_final = pd.concat(df_inc_list, ignore_index=True)
    return df_inc_final

def PARSECv2_collect_data(iso_list, bands_list=None, abs_mags_observed=None, abs_err_mags_observed=None,
                          photo_err_scaler=None, show_progress=True, label_list=None, logAge_list=None,
                          omega_ini_list=None, Inclination_list=None):
    """
    Collects and filters PARSEC v2 isochrone data.

    Args:
        iso_list (str or os.PathLike or list): Isochrone file path(s).
        bands_list (list[str], optional): Photometric bands used for
            photometric filtering.
        abs_mags_observed (array-like, optional): Observed absolute
            magnitudes.
        abs_err_mags_observed (array-like, optional): Uncertainties of the
            observed absolute magnitudes.
        photo_err_scaler (float or array-like, optional): Scale factor or
            direct range for photometric filtering.
        show_progress (bool): Whether to display progress information.
        label_list (list or 'all', optional): Evolutionary labels to retain.
        logAge_list (list, optional): ``logAge`` values to retain.
        omega_ini_list (list, optional): Initial rotation rates to retain.
        Inclination_list (list, optional): Inclination values to retain.

    Returns:
        pandas.DataFrame: Filtered and concatenated isochrone data.

    Raises:
        TypeError: If a filter argument has an invalid type.
        ValueError: If required photometric inputs are missing or
            ``omega_ini`` cannot be determined.
    """
    def _validate_optional_list(name, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, np.ndarray)):
            if show_progress:
                print(f'Collecting grids with {name} = {value}')
            return set(value)
        raise TypeError(f'{name} should be a list, tuple, or np.ndarray.')

    logAge_set = _validate_optional_list('logAge', logAge_list)
    omega_ini_set = _validate_optional_list('omega_ini', omega_ini_list)
    Inclination_set = _validate_optional_list('Inclination', Inclination_list)
    if label_list is None:
        label_list = [1.0]
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
            if show_progress:
                print(f'Collecting grids with label = {label_list}...')
        elif show_progress:
            print('Collecting all grids...')
    elif isinstance(photo_err_scaler, (int, float)):
        if abs_mags_observed is None or abs_err_mags_observed is None:
            raise ValueError('abs_mags_observed, abs_err_mags_observed, and photo_err_scaler must either all be None or all be provided together.')
        collect_range = photo_err_scaler * np.asarray(abs_err_mags_observed)
        if show_progress:
            print(f'Collecting grids with range = {collect_range}')
    else:
        collect_range = np.asarray(photo_err_scaler)
        if show_progress:
            print(f'Collecting grids with range = {collect_range}')
    if collect_range is not None:
        if bands_list is None or abs_mags_observed is None:
            raise ValueError('bands_list and abs_mags_observed must be provided when photo filtering is used.')
        abs_mags_observed = np.asarray(abs_mags_observed)
        band_columns_for_cut = [f'{band}_fSB' for band in bands_list]
    if isinstance(iso_list, (str, os.PathLike)):
        iso_list = [iso_list]
    iterable = tqdm(iso_list) if show_progress else iso_list
    df_final_list = []
    for path in iterable:
        df, omega_ini = read_isochrones(path, show_progress=False, return_omega_ini=True)
        if omega_ini is None:
            omega_pattern = re.compile('omega_ini([^_]+)_')
            match = omega_pattern.search(str(path))
            if match is None:
                raise ValueError(f'Cannot parse omega_ini from path: {path}')
            omega_ini = round(float(match.group(1)), 2)
        if omega_ini_set is not None and omega_ini not in omega_ini_set:
            continue
        df = add_iso_inc_rot_age(df, omega_ini)
        if label_set is not None:
            df = df[df['label'].isin(label_set)]
            if df.empty:
                continue
        if logAge_set is not None:
            df = df[df['logAge'].isin(logAge_set)]
            if df.empty:
                continue
        if omega_ini_set is not None:
            df = df[df['omega_ini'].isin(omega_ini_set)]
            if df.empty:
                continue
        if Inclination_set is not None:
            df = df[df['Inclination'].isin(Inclination_set)]
            if df.empty:
                continue
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
    df_i = PARSECv2_collect_data([path_this], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
    df_i['path_index'] = path_i
    df_i['path'] = path_this
    return df_i

def _process_one_Mini_group(args):
    loop_i, group_key, df_i, group_cols, compare_cols, interp_output_list, Mini_grid, dtype, grid_type, Age_to_idx, MH_to_idx, omega_ini_to_idx, Inclination_to_idx, path_i, path_this = args
    group_info = {col: group_key[k] for k, col in enumerate(group_cols)}
    result = {'status': 'skipped', 'problematic_records': [], 'safe_duplicate_count': 0, 'filled_group_count': 0, 'skipped_group_count': 0, 'grid_index': None, 'y_grid': None, 'grid_bounds': None}
    df_i = df_i.sort_values('Mini').reset_index(drop=True)
    if len(df_i) < 2:
        result['skipped_group_count'] += 1
        return result
    interp_x = df_i['Mini'].values
    interp_y = df_i[interp_output_list].values
    dup_mask_all = pd.Series(interp_x).duplicated(keep=False).to_numpy()
    if dup_mask_all.any():
        local_Mini_min = np.nanmin(interp_x)
        local_Mini_max = np.nanmax(interp_x)
        local_range = local_Mini_max - local_Mini_min
        dup_groups = df_i.loc[dup_mask_all].groupby('Mini', sort=False)
        for mini_value, g in dup_groups:
            n_unique_rows = g[compare_cols].drop_duplicates().shape[0]
            if n_unique_rows == 1:
                result['safe_duplicate_count'] += len(g) - 1
            else:
                bad_pct = 100 * (mini_value - local_Mini_min) / local_range if local_range > 0 else np.nan
                record = {'path_index': path_i, 'path': path_this, 'loop_index_in_this_file': loop_i, 'Mini': mini_value, 'position_pct': bad_pct, 'n_duplicates': len(g), 'n_unique_rows_in_compare_cols': n_unique_rows, 'local_Mini_min': local_Mini_min, 'local_Mini_max': local_Mini_max}
                for col in group_cols:
                    record[col] = group_info[col]
                result['problematic_records'].append(record)
        keep_mask = ~pd.Series(interp_x).duplicated(keep='first').to_numpy()
        df_i = df_i.loc[keep_mask].reset_index(drop=True)
        interp_x = df_i['Mini'].values
        interp_y = df_i[interp_output_list].values
    if len(interp_x) < 2:
        result['skipped_group_count'] += 1
        return result
    if not np.all(np.diff(interp_x) > 0):
        record = {'path_index': path_i, 'path': path_this, 'loop_index_in_this_file': loop_i, 'reason': 'Mini is not strictly increasing after duplicate handling'}
        for col in group_cols:
            record[col] = group_info[col]
        result['problematic_records'].append(record)
        result['skipped_group_count'] += 1
        return result
    local_Mini_min = np.nanmin(interp_x)
    local_Mini_max = np.nanmax(interp_x)
    y_grid = np.full((len(Mini_grid), len(interp_output_list)), np.nan, dtype=dtype)
    valid_mask = (Mini_grid >= local_Mini_min) & (Mini_grid <= local_Mini_max)
    for j, col in enumerate(interp_output_list):
        f = interp1d(interp_x, interp_y[:, j], kind='linear', bounds_error=False, fill_value=np.nan)
        y_grid[valid_mask, j] = f(Mini_grid[valid_mask]).astype(dtype)
    age_value = group_info['Age']
    if age_value not in Age_to_idx:
        result['skipped_group_count'] += 1
        result['problematic_records'].append({'path_index': path_i, 'path': path_this, 'loop_index_in_this_file': loop_i, 'reason': 'Canonical Age is not present in Age_grid', 'Age': age_value})
        return result
    idx_Age = Age_to_idx[age_value]
    idx_MH = MH_to_idx[group_info['MH']]
    idx_omega_ini = omega_ini_to_idx[group_info['omega_ini']]
    if grid_type == 'without_inc':
        grid_index = (idx_Age, idx_MH, idx_omega_ini)
    else:
        inc_value = np.round(group_info['Inclination'], 3)
        if inc_value not in Inclination_to_idx:
            result['skipped_group_count'] += 1
            return result
        idx_inc = Inclination_to_idx[inc_value]
        grid_index = (idx_Age, idx_MH, idx_omega_ini, idx_inc)
    result['status'] = 'filled'
    result['filled_group_count'] = 1
    result['grid_index'] = grid_index
    result['y_grid'] = y_grid
    result['grid_bounds'] = np.array([local_Mini_min, local_Mini_max], dtype=dtype)
    return result

def _age_min_physical_step(age, age_break=20.0, age_step_low=1.0, age_step_high=10.0):
    age = np.asarray(age, dtype=np.float64)
    return np.where(age <= age_break, age_step_low, age_step_high)

def _age_jitter_tolerance(age, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    physical_step = _age_min_physical_step(age, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high)
    return np.maximum(age_jitter_abs, physical_step * age_jitter_fraction)

def _ages_are_close(age1, age2, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    scale_age = max(float(age1), float(age2))
    tol = float(_age_jitter_tolerance(scale_age, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs))
    return abs(float(age1) - float(age2)) <= tol

def _collapse_close_ages(ages, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    ages = np.asarray(ages, dtype=np.float64)
    ages = ages[np.isfinite(ages)]
    if len(ages) == 0:
        return (np.array([], dtype=np.float64), {})
    ages = np.sort(np.unique(ages))
    clusters = []
    current_cluster = [ages[0]]
    for age in ages[1:]:
        previous = current_cluster[-1]
        if _ages_are_close(previous, age, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs):
            current_cluster.append(age)
        else:
            clusters.append(current_cluster)
            current_cluster = [age]
    clusters.append(current_cluster)
    cleaned_ages = []
    raw_to_clean = {}
    for cluster in clusters:
        cluster = np.asarray(cluster, dtype=np.float64)
        representative = float(np.median(cluster))
        cleaned_ages.append(representative)
        for raw_age in cluster:
            raw_to_clean[float(raw_age)] = representative
    return (np.asarray(cleaned_ages, dtype=np.float64), raw_to_clean)

def _estimate_age_group_spacing(ages, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    ages_clean, _ = _collapse_close_ages(ages, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
    if len(ages_clean) < 2:
        return np.inf
    diffs = np.diff(ages_clean)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return np.inf
    return float(np.median(diffs))


def _build_merged_age_grid(age_groups, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06, verbose=True):
    group_infos = []
    for group_index, ages_raw in enumerate(age_groups):
        ages_raw = np.asarray(ages_raw, dtype=np.float64)
        ages_raw = ages_raw[np.isfinite(ages_raw)]
        ages_clean, raw_to_clean = _collapse_close_ages(ages_raw, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
        spacing = _estimate_age_group_spacing(ages_clean, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
        if len(ages_clean) > 0:
            age_min = float(np.min(ages_clean))
            age_max = float(np.max(ages_clean))
        else:
            age_min = np.nan
            age_max = np.nan
        group_infos.append({'group_index': group_index, 'ages_raw': ages_raw, 'ages_clean': ages_clean, 'raw_to_clean': raw_to_clean, 'spacing': spacing, 'age_min': age_min, 'age_max': age_max})
    candidates = []
    for info in group_infos:
        group_index = info['group_index']
        spacing = info['spacing']
        for age in info['ages_clean']:
            age = float(age)
            covering_groups = []
            for other in group_infos:
                if len(other['ages_clean']) == 0:
                    continue
                tol = float(_age_jitter_tolerance(age, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs))
                if age >= other['age_min'] - tol and age <= other['age_max'] + tol:
                    covering_groups.append(other)
            if len(covering_groups) == 0:
                best_spacing = spacing
            else:
                best_spacing = min((x['spacing'] for x in covering_groups))
            keep = np.isclose(spacing, best_spacing, rtol=1e-08, atol=1e-12)
            candidates.append({'age': age, 'group_index': group_index, 'spacing': spacing, 'keep': bool(keep)})
    retained_candidates = [c for c in candidates if c['keep']]
    retained_candidates.sort(key=lambda x: x['age'])
    final_clusters = []
    for candidate in retained_candidates:
        if len(final_clusters) == 0:
            final_clusters.append([candidate])
            continue
        previous_age = final_clusters[-1][-1]['age']
        if _ages_are_close(previous_age, candidate['age'], age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs):
            final_clusters[-1].append(candidate)
        else:
            final_clusters.append([candidate])
    final_ages = []
    candidate_to_final = {}
    for cluster in final_clusters:
        best_candidate = min(cluster, key=lambda x: (x['spacing'], x['group_index']))
        final_age = float(best_candidate['age'])
        final_ages.append(final_age)
        for candidate in cluster:
            candidate_to_final[candidate['group_index'], candidate['age']] = final_age
    Age_grid = np.asarray(final_ages, dtype=np.float64)
    Age_grid = np.sort(np.unique(Age_grid))
    retained_keys = {(c['group_index'], c['age']) for c in retained_candidates}
    age_selection = []
    for info in group_infos:
        group_index = info['group_index']
        clean_to_final = {}
        for clean_age in info['ages_clean']:
            clean_age = float(clean_age)
            key = (group_index, clean_age)
            if key in retained_keys:
                clean_to_final[clean_age] = candidate_to_final[key]
            else:
                clean_to_final[clean_age] = None
        raw_to_final = {}
        for raw_age, clean_age in info['raw_to_clean'].items():
            raw_to_final[float(raw_age)] = clean_to_final[float(clean_age)]
        age_selection.append({'group_index': group_index, 'raw_to_final': raw_to_final, 'clean_to_final': clean_to_final, 'spacing': info['spacing'], 'age_min': info['age_min'], 'age_max': info['age_max']})
    records = []
    for info, selection in zip(group_infos, age_selection):
        n_raw = len(info['ages_raw'])
        n_clean = len(info['ages_clean'])
        n_retained = sum((x is not None for x in selection['clean_to_final'].values()))
        records.append({'group_index': info['group_index'], 'age_min': info['age_min'], 'age_max': info['age_max'], 'estimated_spacing': info['spacing'], 'n_raw_age': n_raw, 'n_clean_age': n_clean, 'n_retained_age': n_retained, 'n_discarded_age': n_clean - n_retained})
    summary_df = pd.DataFrame(records)
    if verbose:
        print()
        print('Age group summary:')
        print(summary_df.to_string(index=False))
        print()
        print(f'Final Age_grid: {len(Age_grid)} points')
        if len(Age_grid) > 0:
            print(f'Age range: {Age_grid.min():g} - {Age_grid.max():g} Myr')
    return (Age_grid, age_selection, summary_df)

def _apply_age_selection_to_one_df(df, source_index, age_selection, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    if len(df) == 0:
        return df.copy()
    selection = age_selection[source_index]
    raw_to_final = selection['raw_to_final']
    known_raw_ages = np.asarray(list(raw_to_final.keys()), dtype=np.float64)
    known_final_ages = [raw_to_final[x] for x in known_raw_ages]
    age_values = df['Age'].to_numpy(dtype=np.float64)
    keep = np.zeros(len(df), dtype=bool)
    mapped = np.full(len(df), np.nan, dtype=np.float64)
    unique_ages = np.unique(age_values[np.isfinite(age_values)])
    unique_mapping = {}
    for age in unique_ages:
        age_float = float(age)
        if age_float in raw_to_final:
            final_age = raw_to_final[age_float]
        elif len(known_raw_ages) == 0:
            final_age = None
        else:
            nearest_idx = int(np.argmin(np.abs(known_raw_ages - age)))
            nearest_raw = known_raw_ages[nearest_idx]
            if _ages_are_close(age, nearest_raw, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs):
                final_age = known_final_ages[nearest_idx]
            else:
                final_age = None
        unique_mapping[age_float] = final_age
    for age, final_age in unique_mapping.items():
        mask = age_values == age
        if final_age is not None:
            keep[mask] = True
            mapped[mask] = final_age
    df_out = df.loc[keep].copy().reset_index(drop=True)
    df_out['Age'] = mapped[keep]
    return df_out

def _apply_age_selection_to_df_all(df_all, age_selection, **kwargs):
    parts = []
    source_indices = np.sort(df_all['path_index'].unique())
    for source_index in source_indices:
        source_index = int(source_index)
        df_i = df_all.loc[df_all['path_index'] == source_index].copy()
        df_i = _apply_age_selection_to_one_df(df_i, source_index, age_selection, **kwargs)
        if len(df_i) > 0:
            parts.append(df_i)
    if len(parts) == 0:
        return df_all.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)

def regularization_on_Mini_from_files(
        path, save_path='.', save_name=None, Mini_setting=None, grid_type='with_inc', label_list=None, Inclination_list=None,
        logAge_list=None, print_path_list=False, save_float32=True, read_all_at_once=False, parallel_file_read=True, parallel_backend='process',
        max_workers=None, merge_age_groups=True, age_break=20.0, age_step_low=1.0, age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    """
    Regularizes PARSEC isochrones along Mini and constructs a regular grid.

    Args:
        path (str or os.PathLike or list): Input isochrone file path(s).
        save_path (str or os.PathLike): Output directory.
        save_name (str, optional): Output filename.
        Mini_setting (list, optional): ``[Mini_min, Mini_max, Mini_step]``.
        grid_type (str): Grid type, either ``"with_inc"`` or ``"without_inc"``.
        label_list (list, optional): Evolutionary labels to retain.
        Inclination_list (array-like, optional): Inclination values to retain.
        logAge_list (list, optional): ``logAge`` values to retain.
        print_path_list (bool): Whether to print the input file list.
        save_float32 (bool): Whether to store grid data as float32.
        read_all_at_once (bool): Whether to load all input data at once.
        parallel_file_read (bool): Whether to read files in parallel.
        parallel_backend (str): Backend for parallel file reading.
        max_workers (int, optional): Maximum number of parallel workers.
        merge_age_groups (bool): Whether to merge Age sampling groups.
        age_break (float): Age separating the low- and high-resolution grids.
        age_step_low (float): Age step below ``age_break``.
        age_step_high (float): Age step above ``age_break``.
        age_jitter_fraction (float): Relative tolerance for Age matching.
        age_jitter_abs (float): Absolute tolerance for Age matching.

    Returns:
        tuple: Absolute output file path and a dataframe of problematic
            duplicated Mini records.
    """
    if not read_all_at_once and parallel_file_read:
        print('[info] parallel_file_read is ignored because read_all_at_once=False.')
    if label_list is None:
        label_list = [1.0]
    if Inclination_list is None:
        Inclination_list = np.array(np.deg2rad([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 90.0]))
    Inclination_list = np.round(Inclination_list, 3)
    save_path = str(save_path)
    if not save_path.endswith('/'):
        save_path += '/'
    os.makedirs(save_path, exist_ok=True)
    if save_name is None:
        save_name = f'regularized_Mini_grid_{grid_type}.npz'
    dtype = np.float32 if save_float32 else np.float64
    if isinstance(path, (str, os.PathLike)):
        files = glob(path)
    elif isinstance(path, list):
        files = []
        for path_0 in path:
            files += glob(path_0)
    else:
        raise ValueError('path must be str or list.')
    path_list = sorted(files)
    if len(path_list) == 0:
        raise ValueError(f'No files found with path pattern: {path}')
    if print_path_list:
        print(path_list)
    if grid_type == 'without_inc':
        in_cols = ['Mini', 'Age', 'MH', 'omega_ini']
        interp_variable_list = ['Mini', 'Age', 'MH', 'omega_ini']
        group_cols = ['Age', 'MH', 'omega_ini']
        interp_output_list = ['Mass', 'omega', 'logL', 'logTe', 'Req', 'Rpol']
    elif grid_type == 'with_inc':
        in_cols = ['Mini', 'Age', 'MH', 'omega_ini', 'Inclination']
        interp_variable_list = ['Mini', 'Age', 'MH', 'omega_ini', 'Inclination']
        group_cols = ['Age', 'MH', 'omega_ini', 'Inclination']
        interp_output_list = ['Mass', 'omega', 'logL', 'logTe', 'Req', 'Rpol', 'G', 'G_BP', 'G_RP', 'B_T', 'V_T', 'J', 'H', 'Ks']
    else:
        raise ValueError('grid_type must be "without_inc" or "with_inc".')
    compare_cols = in_cols + interp_output_list
    df_all = None
    if read_all_at_once:
        if parallel_file_read:
            print('Reading all files at once in parallel...')
            read_args = [(path_i, path_list[path_i], label_list, Inclination_list, logAge_list) for path_i in range(len(path_list))]
            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(path_list))
            if parallel_backend == 'process':
                Executor = ProcessPoolExecutor
            elif parallel_backend == 'thread':
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")
            with Executor(max_workers=max_workers) as executor:
                df_list = list(tqdm(executor.map(_read_one_parsec_file_for_regularization, read_args), total=len(read_args)))
        else:
            print('Reading all files at once...')
            df_list = []
            for path_i in tqdm(range(len(path_list))):
                df_i = PARSECv2_collect_data([path_list[path_i]], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
                df_i['path_index'] = path_i
                df_i['path'] = path_list[path_i]
                df_list.append(df_i)
        df_all = pd.concat(df_list, ignore_index=True)
        del df_list
        if grid_type == 'with_inc':
            df_all['Inclination'] = np.round(df_all['Inclination'].values, 3)
        print(f'df_all shape: {df_all.shape}')
        memory_bytes = df_all.memory_usage(deep=True).sum()
        print(f'df_all memory: {memory_bytes / 1024 ** 2:.2f} MB ({memory_bytes / 1024 ** 3:.2f} GB)')
    print('Collecting axis values...')
    Mini_min_list = []
    Mini_max_list = []
    Age_groups = [None for _ in path_list]
    MH_list = []
    omega_ini_list = []
    Inclination_axis_list = []
    if read_all_at_once:
        missing_cols = [c for c in compare_cols if c not in df_all.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in df_all: {missing_cols}')
        Mini_min_list.append(np.nanmin(df_all['Mini'].values))
        Mini_max_list.append(np.nanmax(df_all['Mini'].values))
        for path_i in range(len(path_list)):
            df_i = df_all.loc[df_all['path_index'] == path_i]
            Age_groups[path_i] = df_i['Age'].dropna().unique()
        MH_list.append(df_all['MH'].dropna().unique())
        omega_ini_list.append(df_all['omega_ini'].dropna().unique())
        if grid_type == 'with_inc':
            Inclination_axis_list.append(np.round(df_all['Inclination'].dropna().unique(), 3))
    else:
        for path_i in tqdm(range(len(path_list))):
            df = PARSECv2_collect_data([path_list[path_i]], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
            if grid_type == 'with_inc':
                df['Inclination'] = np.round(df['Inclination'].values, 3)
            missing_cols = [c for c in compare_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f'Missing columns in df from {path_list[path_i]}: {missing_cols}')
            Mini_min_list.append(np.nanmin(df['Mini'].values))
            Mini_max_list.append(np.nanmax(df['Mini'].values))
            Age_groups[path_i] = df['Age'].dropna().unique()
            MH_list.append(df['MH'].dropna().unique())
            omega_ini_list.append(df['omega_ini'].dropna().unique())
            if grid_type == 'with_inc':
                Inclination_axis_list.append(np.round(df['Inclination'].dropna().unique(), 3))
            del df
    if Mini_setting is None:
        Mini_min = np.nanmin(Mini_min_list)
        Mini_max = np.nanmax(Mini_max_list)
        Mini_step = 0.01
    else:
        if not isinstance(Mini_setting, (list, tuple, np.ndarray)) or len(Mini_setting) != 3:
            raise ValueError('Mini_setting must be [Mini_min, Mini_max, Mini_step].')
        Mini_min, Mini_max, Mini_step = Mini_setting
    if Mini_step <= 0:
        raise ValueError('Mini_step must be positive.')
    n_Mini = int(np.floor((Mini_max - Mini_min) / Mini_step + 0.5)) + 1
    Mini_grid = Mini_min + np.arange(n_Mini) * Mini_step
    Mini_grid = np.round(Mini_grid, 10)
    Mini_grid = Mini_grid[Mini_grid <= Mini_max + 1e-10]
    if merge_age_groups:
        Age_grid, age_selection, age_group_summary = _build_merged_age_grid(
            Age_groups, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs, verbose=True)
    else:
        all_ages = np.concatenate(Age_groups)
        Age_grid, _ = _collapse_close_ages(all_ages, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
        age_selection = []
        for group_index, ages in enumerate(Age_groups):
            raw_to_final = {}
            for age in np.unique(ages):
                idx = np.argmin(np.abs(Age_grid - age))
                raw_to_final[float(age)] = float(Age_grid[idx])
            age_selection.append({'group_index': group_index, 'raw_to_final': raw_to_final})
        age_group_summary = pd.DataFrame()
    MH_grid = np.sort(np.unique(np.concatenate(MH_list)))
    omega_ini_grid = np.sort(np.unique(np.concatenate(omega_ini_list)))
    if grid_type == 'with_inc':
        Inclination_grid = np.round(np.asarray(Inclination_list), 3)
        Inclination_grid = np.round(np.sort(Inclination_grid), 3)
    else:
        Inclination_grid = None
    age_selection_kwargs = dict(age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
    if read_all_at_once:
        old_n_rows = len(df_all)
        df_all = _apply_age_selection_to_df_all(df_all, age_selection, **age_selection_kwargs)
        print(f'Age-group filtering: {old_n_rows:,} -> {len(df_all):,} rows')
    Age_to_idx = {float(v): i for i, v in enumerate(Age_grid)}
    MH_to_idx = {v: i for i, v in enumerate(MH_grid)}
    omega_ini_to_idx = {v: i for i, v in enumerate(omega_ini_grid)}
    if grid_type == 'with_inc':
        Inclination_to_idx = {v: i for i, v in enumerate(Inclination_grid)}
    else:
        Inclination_to_idx = None
    if grid_type == 'without_inc':
        grid_outputs = np.full((len(Mini_grid), len(Age_grid), len(MH_grid), len(omega_ini_grid), len(interp_output_list)), np.nan, dtype=dtype)
        grid_bounds = np.full((len(Age_grid), len(MH_grid), len(omega_ini_grid), 2), np.nan, dtype=dtype)
    else:
        grid_outputs = np.full((len(Mini_grid), len(Age_grid), len(MH_grid), len(omega_ini_grid), len(Inclination_grid), len(interp_output_list)), np.nan, dtype=dtype)
        grid_bounds = np.full((len(Age_grid), len(MH_grid), len(omega_ini_grid), len(Inclination_grid), 2), np.nan, dtype=dtype)
    print(f'Mini_grid: [min: {np.min(Mini_grid)}, max: {np.max(Mini_grid)}]')
    print(f'Mini_grid shape: {Mini_grid.shape}')
    print(f'Age_grid shape: {Age_grid.shape}')
    print(f'MH_grid shape: {MH_grid.shape}')
    print(f'omega_ini_grid shape: {omega_ini_grid.shape}')
    if grid_type == 'with_inc':
        print(f'Inclination_grid shape: {Inclination_grid.shape}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_bounds shape: {grid_bounds.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_outputs memory: {grid_outputs.nbytes / 1024 ** 2:.2f} MB ({grid_outputs.nbytes / 1024 ** 3:.2f} GB)')
    print('Filling regular grids...')
    problematic_records = []
    safe_duplicate_count = 0
    filled_group_count = 0
    skipped_group_count = 0
    if read_all_at_once:
        file_iter = [(-1, 'df_all')]
    else:
        file_iter = tqdm([(path_i, path_list[path_i]) for path_i in range(len(path_list))])
    for path_i, path_this in file_iter:
        if read_all_at_once:
            df = df_all.copy()
        else:
            df = PARSECv2_collect_data([path_this], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
            if grid_type == 'with_inc':
                df['Inclination'] = np.round(df['Inclination'].values, 3)
            df['path_index'] = path_i
            df['path'] = path_this
            df = _apply_age_selection_to_one_df(df, path_i, age_selection, **age_selection_kwargs)
        if len(df) == 0:
            if not read_all_at_once:
                continue
        missing_cols = [c for c in compare_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in df from {path_this}: {missing_cols}')
        grouped = list(df.groupby(group_cols, sort=False))
        task_args = []
        for loop_i, (group_key, df_i) in enumerate(grouped):
            if len(group_cols) == 1:
                group_key = (group_key,)
            task_args.append((loop_i, group_key, df_i, group_cols, compare_cols, interp_output_list, Mini_grid, dtype, grid_type, Age_to_idx, MH_to_idx, omega_ini_to_idx, Inclination_to_idx, path_i, path_this))
        use_process = False
        local_max_workers = 8
        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        with Executor(max_workers=local_max_workers) as executor:
            iterator = executor.map(_process_one_Mini_group, task_args)
            if read_all_at_once:
                iterator = tqdm(iterator, total=len(task_args))
            results = list(iterator)
        for res in results:
            problematic_records.extend(res['problematic_records'])
            safe_duplicate_count += res['safe_duplicate_count']
            skipped_group_count += res['skipped_group_count']
            if res['status'] != 'filled':
                continue
            y_grid = res['y_grid']
            bounds_i = res['grid_bounds']
            grid_index = res['grid_index']
            if grid_type == 'without_inc':
                idx_Age, idx_MH, idx_omega_ini = grid_index
                grid_outputs[:, idx_Age, idx_MH, idx_omega_ini, :] = y_grid
                grid_bounds[idx_Age, idx_MH, idx_omega_ini, :] = bounds_i
            else:
                idx_Age, idx_MH, idx_omega_ini, idx_inc = grid_index
                grid_outputs[:, idx_Age, idx_MH, idx_omega_ini, idx_inc, :] = y_grid
                grid_bounds[idx_Age, idx_MH, idx_omega_ini, idx_inc, :] = bounds_i
            filled_group_count += res['filled_group_count']
        del df
    if read_all_at_once:
        del df_all
    problematic_df = pd.DataFrame(problematic_records)
    save_file = save_path + save_name
    problematic_path = save_path + 'problematic_duplicated_' + f'{Path(save_name).stem}.csv'
    age_summary_path = save_path + 'age_group_summary_' + f'{Path(save_name).stem}.csv'
    save_dict = dict(
        Mini_grid=Mini_grid, Age_grid=Age_grid, MH_grid=MH_grid, omega_ini_grid=omega_ini_grid, grid_outputs=grid_outputs,
        grid_bounds=grid_bounds, Mini_bounds_output_list=np.array(['Mini_min', 'Mini_max'], dtype=object),
        in_cols=np.array(in_cols, dtype=object), interp_output_list=np.array(interp_output_list, dtype=object),
        interp_variable_list=np.array(interp_variable_list, dtype=object), group_cols=np.array(group_cols, dtype=object),
        grid_type=np.array(grid_type), label_list=np.array(label_list), age_unit=np.array('Myr'), save_float32=np.array(save_float32),
        read_all_at_once=np.array(read_all_at_once), grid_dtype=np.array(str(grid_outputs.dtype)), safe_duplicate_count=np.array(safe_duplicate_count),
        filled_group_count=np.array(filled_group_count), skipped_group_count=np.array(skipped_group_count), merge_age_groups=np.array(merge_age_groups),
        age_break=np.array(age_break), age_step_low=np.array(age_step_low), age_step_high=np.array(age_step_high),
        age_jitter_fraction=np.array(age_jitter_fraction), age_jitter_abs=np.array(age_jitter_abs))
    if grid_type == 'with_inc':
        save_dict['Inclination_grid'] = Inclination_grid
    np.savez_compressed(save_file, **save_dict)
    problematic_df.to_csv(problematic_path, index=False)
    if len(age_group_summary) > 0:
        age_group_summary = age_group_summary.copy()
        age_group_summary['path'] = path_list
        age_group_summary.to_csv(age_summary_path, index=False)
    total_bytes = grid_outputs.nbytes + grid_bounds.nbytes + Mini_grid.nbytes + Age_grid.nbytes + MH_grid.nbytes + omega_ini_grid.nbytes
    if grid_type == 'with_inc':
        total_bytes += Inclination_grid.nbytes
    print()
    print('Finished regularization.')
    print(f'grid_type: {grid_type}')
    print(f'read_all_at_once: {read_all_at_once}')
    print(f'parallel_file_read: {parallel_file_read}')
    print(f'parallel_backend: {parallel_backend}')
    print(f'max_workers: {max_workers}')
    print(f'save_float32: {save_float32}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_bounds dtype: {grid_bounds.dtype}')
    print(f'filled_group_count: {filled_group_count}')
    print(f'skipped_group_count: {skipped_group_count}')
    print(f'safe_duplicate_count: {safe_duplicate_count}')
    print(f'problematic_duplicate_count: {len(problematic_df)}')
    print(f'Final Age_grid contains {len(Age_grid)} points.')
    print(f'problematic duplicated Mini records are saved at:\n{os.path.abspath(problematic_path)}')
    if len(age_group_summary) > 0:
        print(f'Age-group summary is saved at:\n{os.path.abspath(age_summary_path)}')
    print(f'Grids are saved at:\n{os.path.abspath(save_file)}')
    print(f'total grid memory: {total_bytes / 1024 ** 2:.2f} MB ({total_bytes / 1024 ** 3:.2f} GB)')
    return (os.path.abspath(save_file), problematic_df)

class MapCoordinatesRegularGridInterpolator:
    """
    Linear interpolator for regular grids using scipy.ndimage.map_coordinates.

    Supports uniform and non-uniform grid axes, scalar and batch inputs, and
    multiple output dimensions. Returns fill_value for out-of-bounds inputs.

    Args:
        points (tuple of ndarray): Physical coordinates for each grid axis.
        values (ndarray): Grid values with shape
            ``(*grid_shape, n_outputs)``.
        fill_value (float): Value returned for invalid inputs.
        output_dtype (numpy.dtype): Output data type.
    """

    def __init__(self, points, values, fill_value=np.nan, output_dtype=np.float64):
        self.points = tuple((np.asarray(axis, dtype=np.float64) for axis in points))
        self.ndim = len(self.points)
        self.fill_value = fill_value
        self.output_dtype = np.dtype(output_dtype)
        values = np.asarray(values)
        expected_spatial_shape = tuple((len(axis) for axis in self.points))
        if values.ndim != self.ndim + 1:
            raise ValueError(
                f'values must have one additional output dimension.\n'
                f'Expected ndim: {self.ndim + 1}\n'
                f'Received ndim: {values.ndim}\n'
                f'Received shape: {values.shape}'
            )
        if values.shape[:self.ndim] != expected_spatial_shape:
            raise ValueError(
                f'The leading dimensions of values must match the shape of points.\n'
                f'Expected spatial shape: {expected_spatial_shape}\n'
                f'Received spatial shape: {values.shape[:self.ndim]}'
            )
        self.n_outputs = values.shape[-1]
        self.values = np.ascontiguousarray(np.moveaxis(values, -1, 0))
        self.axis_is_uniform = []
        self.axis_inverse_step = []
        for axis_i, axis in enumerate(self.points):
            if axis.ndim != 1:
                raise ValueError(f'Axis {axis_i} must be one-dimensional.')
            if len(axis) == 0:
                raise ValueError(f'Axis {axis_i} is empty.')
            if len(axis) > 1 and (not np.all(np.diff(axis) > 0)):
                raise ValueError(f'Axis {axis_i} must be strictly increasing.')
            if len(axis) == 1:
                is_uniform = True
                inverse_step = np.nan
            else:
                differences = np.diff(axis)
                is_uniform = np.allclose(differences, differences[0], rtol=1e-10, atol=1e-12)
                if is_uniform:
                    inverse_step = 1.0 / differences[0]
                else:
                    inverse_step = np.nan
            self.axis_is_uniform.append(is_uniform)
            self.axis_inverse_step.append(inverse_step)
        self.axis_is_uniform = tuple(self.axis_is_uniform)
        self.axis_inverse_step = tuple(self.axis_inverse_step)
        self.output_coordinates = np.arange(self.n_outputs, dtype=np.float64)

    def _physical_to_fractional_index(self, values, axis_index):
        """
        Convert physical coordinates to fractional array indices.

        Args:
            values (array-like): Physical coordinates.
            axis_index (int): Index of the grid axis.

        Returns:
            tuple: Fractional indices and a boolean validity mask.
        """
        values = np.asarray(values, dtype=np.float64)
        axis = self.points[axis_index]
        valid = np.isfinite(values) & (values >= axis[0]) & (values <= axis[-1])
        fractional_indices = np.zeros_like(values, dtype=np.float64)
        if len(axis) == 1:
            valid &= np.isclose(values, axis[0], rtol=0.0, atol=1e-12)
            fractional_indices[:] = 0.0
            return (fractional_indices, valid)
        if self.axis_is_uniform[axis_index]:
            fractional_indices[valid] = (values[valid] - axis[0]) * self.axis_inverse_step[axis_index]
        else:
            valid_values = values[valid]
            upper_indices = np.searchsorted(axis, valid_values, side='right')
            upper_indices = np.clip(upper_indices, 1, len(axis) - 1)
            lower_indices = upper_indices - 1
            lower_values = axis[lower_indices]
            upper_values = axis[upper_indices]
            fractions = (valid_values - lower_values) / (upper_values - lower_values)
            fractional_indices[valid] = lower_indices + fractions
        fractional_indices[valid] = np.clip(fractional_indices[valid], 0.0, float(len(axis) - 1))
        return (fractional_indices, valid)

    def _physical_to_fractional_index_scalar(self, value, axis_index):
        """
        Convert a scalar physical coordinate to a fractional index.

        Args:
            value (float): Physical coordinate.
            axis_index (int): Index of the grid axis.

        Returns:
            float: Fractional index, or np.nan if invalid.
        """
        axis = self.points[axis_index]
        if not np.isfinite(value) or value < axis[0] or value > axis[-1]:
            return np.nan
        if len(axis) == 1:
            if np.isclose(value, axis[0], rtol=0.0, atol=1e-12):
                return 0.0
            return np.nan
        if self.axis_is_uniform[axis_index]:
            index = (value - axis[0]) * self.axis_inverse_step[axis_index]
        else:
            upper_index = np.searchsorted(axis, value, side='right')
            if upper_index >= len(axis):
                return float(len(axis) - 1)
            lower_index = upper_index - 1
            lower_value = axis[lower_index]
            upper_value = axis[upper_index]
            fraction = (value - lower_value) / (upper_value - lower_value)
            index = lower_index + fraction
        return min(max(index, 0.0), float(len(axis) - 1))

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            if x.shape[0] != self.ndim:
                raise ValueError(f'Input must have shape ({self.ndim},) or (N, {self.ndim}). Received shape: {x.shape}')
            indices = np.empty(self.ndim, dtype=np.float64)
            for axis_index in range(self.ndim):
                index_i = self._physical_to_fractional_index_scalar(x[axis_index], axis_index)
                if not np.isfinite(index_i):
                    return np.full(self.n_outputs, self.fill_value, dtype=self.output_dtype)
                indices[axis_index] = index_i
            coordinates = np.empty((self.ndim + 1, self.n_outputs), dtype=np.float64)
            coordinates[0] = self.output_coordinates
            for axis_index in range(self.ndim):
                coordinates[axis_index + 1].fill(indices[axis_index])
            return map_coordinates(input=self.values, coordinates=coordinates, order=1, mode='nearest', prefilter=False, output=self.output_dtype)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(f'Input must have shape ({self.ndim},) or (N, {self.ndim}). Received shape: {x.shape}')
        n_points = x.shape[0]
        fractional_indices = np.empty((self.ndim, n_points), dtype=np.float64)
        valid = np.ones(n_points, dtype=bool)
        for axis_index in range(self.ndim):
            index_i, valid_i = self._physical_to_fractional_index(x[:, axis_index], axis_index)
            fractional_indices[axis_index] = index_i
            valid &= valid_i
        result = np.full((n_points, self.n_outputs), self.fill_value, dtype=self.output_dtype)
        if not np.any(valid):
            return result
        valid_indices = fractional_indices[:, valid]
        n_valid = np.count_nonzero(valid)
        coordinates = np.empty((self.ndim + 1, self.n_outputs, n_valid), dtype=np.float64)
        coordinates[0] = self.output_coordinates[:, None]
        for axis_index in range(self.ndim):
            coordinates[axis_index + 1] = valid_indices[axis_index][None, :]
        interpolated = map_coordinates(input=self.values, coordinates=coordinates, order=1, mode='nearest', prefilter=False, output=self.output_dtype)
        result[valid] = interpolated.T
        return result

def load_regularized_Mini_interpolator(npz_path, force_float64=False):
    """
    Load a regularized Mini grid using map_coordinates.

    Args:
        npz_path (str or os.PathLike): Path to the regularized Mini grid.
        force_float64 (bool): Whether to convert grid arrays to float64.

    Returns:
        tuple: Interpolators, conversion functions, metadata lists, and data.
            Returns ``(interp_func, interp_u, interp_u_to_outputs,
            interp_u_bounds, interp_variable_list, interp_output_list, data)``.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_type = str(data['grid_type'])
    Mini_grid = data['Mini_grid']
    Age_grid = data['Age_grid']
    MH_grid = data['MH_grid']
    omega_ini_grid = data['omega_ini_grid']
    grid_outputs = data['grid_outputs']
    if 'grid_bounds' not in data.files:
        raise ValueError('This Mini grid does not contain grid_bounds. Please regenerate the Mini grid with the updated regularization_on_Mini_from_files().')
    grid_bounds = data['grid_bounds']
    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)
        grid_bounds = grid_bounds.astype(np.float64)
    interp_variable_list = data['interp_variable_list'].astype(str).tolist()
    interp_output_list = data['interp_output_list'].astype(str).tolist()
    if 'Age' not in interp_variable_list:
        raise ValueError(f'This grid does not use Age as its age coordinate. interp_variable_list={interp_variable_list}')
    if grid_type == 'without_inc':
        points = (Mini_grid, Age_grid, MH_grid, omega_ini_grid)
        param_points = (Age_grid, MH_grid, omega_ini_grid)
        Inclination_grid = None
    elif grid_type == 'with_inc':
        Inclination_grid = data['Inclination_grid']
        points = (Mini_grid, Age_grid, MH_grid, omega_ini_grid, Inclination_grid)
        param_points = (Age_grid, MH_grid, omega_ini_grid, Inclination_grid)
    else:
        raise ValueError(f'Unknown grid_type: {grid_type}')
    interp_func = MapCoordinatesRegularGridInterpolator(points=points, values=grid_outputs, fill_value=np.nan, output_dtype=np.float64)
    interp_u_bounds = MapCoordinatesRegularGridInterpolator(points=param_points, values=grid_bounds, fill_value=np.nan, output_dtype=np.float64)
    n_param = len(interp_variable_list)

    def interp_u(x, eps=1e-12, clip_u=False, return_bounds=False):
        """
        Convert Mini to normalized evolutionary phase u.

        Args:
            x (array-like): Input parameters with shape ``(n_param,)`` or
                ``(N, n_param)``.
            eps (float): Minimum allowed Mini range.
            clip_u (bool): Whether to clip u to [0, 1].
            return_bounds (bool): Whether to return the Mini bounds.

        Returns:
            float or ndarray: Normalized evolutionary phase.
            Optionally returns the corresponding Mini bounds.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}). Received shape: {x.shape}')
            Mini = x[0]
            params = x[1:]
            bounds = interp_u_bounds(params)
            Mini_min = bounds[0]
            Mini_max = bounds[1]
            width = Mini_max - Mini_min
            if not np.isfinite(Mini) or not np.isfinite(Mini_min) or (not np.isfinite(Mini_max)) or (not np.isfinite(width)) or (width <= eps):
                u = np.nan
            else:
                u = (Mini - Mini_min) / width
                if clip_u:
                    u = np.clip(u, 0.0, 1.0)
            if return_bounds:
                return (u, bounds)
            return u
        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}). Received shape: {x.shape}')
        Mini = x[:, 0]
        params = x[:, 1:]
        bounds = interp_u_bounds(params)
        Mini_min = bounds[:, 0]
        Mini_max = bounds[:, 1]
        width = Mini_max - Mini_min
        u = np.full(Mini.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(Mini) & np.isfinite(Mini_min) & np.isfinite(Mini_max) & np.isfinite(width) & (width > eps)
        u[valid] = (Mini[valid] - Mini_min[valid]) / width[valid]
        if clip_u:
            u[valid] = np.clip(u[valid], 0.0, 1.0)
        if return_bounds:
            return (u, bounds)
        return u

    def interp_u_to_outputs(x, eps=1e-12, clip_Mini=False, return_Mini=False, return_bounds=False):
        x = np.asarray(x, dtype=np.float64)
        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}).')
            x = x[None, :]
            single_input = True
        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}).')
        u = x[:, 0]
        params = x[:, 1:]
        bounds = interp_u_bounds(params)
        Mini_min = bounds[:, 0]
        Mini_max = bounds[:, 1]
        width = Mini_max - Mini_min
        Mini = np.full_like(u, np.nan, dtype=np.float64)
        valid = np.isfinite(Mini_min) & np.isfinite(Mini_max) & np.isfinite(width) & (width > eps)
        Mini[valid] = Mini_min[valid] + u[valid] * width[valid]
        if clip_Mini:
            Mini[valid] = np.clip(Mini[valid], Mini_min[valid], Mini_max[valid])
        x_Mini = np.column_stack([Mini, params])
        y = interp_func(x_Mini)
        if single_input:
            y = y[0]
            Mini = Mini[0]
            bounds = bounds[0]
        if return_Mini and return_bounds:
            return (y, Mini, bounds)
        if return_Mini:
            return (y, Mini)
        if return_bounds:
            return (y, bounds)
        return y
    total_grid_bytes = grid_outputs.nbytes + grid_bounds.nbytes + Mini_grid.nbytes + Age_grid.nbytes + MH_grid.nbytes + omega_ini_grid.nbytes
    if grid_type == 'with_inc':
        total_grid_bytes += Inclination_grid.nbytes
    print(f"Age_grid: [{np.min(Age_grid)}, {np.max(Age_grid)}] {str(data['age_unit'])}")
    print(f'Age_grid shape: {Age_grid.shape}')
    print('Loaded regularized Mini grid using ndimage.map_coordinates.')
    print(f'grid_type: {grid_type}')
    if 'label_list' in data.files:
        print(f"label_list: {data['label_list'].tolist()}")
    print(f'interp_variable_list: {interp_variable_list}')
    print(f'interp_output_list: {interp_output_list}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_bounds shape: {grid_bounds.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_bounds dtype: {grid_bounds.dtype}')
    if 'save_float32' in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")
    if 'read_all_at_once' in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")
    print(f'total grid memory: {total_grid_bytes / 1024 ** 2:.2f} MB ({total_grid_bytes / 1024 ** 3:.2f} GB)')
    print('Axis spacing:')
    for name, is_uniform in zip(interp_variable_list, interp_func.axis_is_uniform):
        spacing_type = 'uniform' if is_uniform else 'non-uniform'
        print(f'  {name:15s}: {spacing_type}')
    return (interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variable_list, interp_output_list, data)

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

def _label_list_to_suffix(label_list):
    """Convert labels to a compact filename suffix, e.g. [0, 1] -> ``label01``."""
    if isinstance(label_list, str):
        if label_list == 'all':
            return 'labelall'
        raise ValueError("label_list must be a sequence of labels or 'all'.")
    labels = []
    for value in label_list:
        value_float = float(value)
        labels.append(int(value_float) if value_float.is_integer() else value_float)
    labels = sorted(set(labels))
    return 'label' + ''.join(str(x).replace('.', 'p') for x in labels)

def regularize_PARSECv2_Mini_grid(path=None, save_path=None, save_name=None,
                                  age_initial='1e6', age_end='2001e6', age_interval='10e6', age_type='linear',
                                  photometry_suffix='2mass_tycho2_gaia_dr3', grid_type='with_inc', label_list=None,
                                  print_path_list=False, save_float32=True, read_all_at_once=False, **kwargs):
    """
    Regularize PARSEC v2 isochrones along Mini.

    Uses the default PARSEC v2 isochrone directory when no input path is
    provided. Saves the regularized grid to the default output directory
    unless another location is specified.

    Args:
        path (str, os.PathLike, list, or tuple, optional): Input file path(s).
            Defaults to the PARSEC v2 rearranged isochrone directory.
        save_path (str or os.PathLike, optional): Output directory.
        save_name (str, optional): Output filename.
        age_initial (str): Initial isochrone age in years.
        age_end (str): Final isochrone age in years.
        age_interval (str): Age sampling interval in years.
        age_type (str): Age sampling scheme.
        photometry_suffix (str): Photometry configuration used to locate
            the default input files.
        grid_type (str): Grid type, either ``"with_inc"`` or
            ``"without_inc"``.
        label_list (list, optional): Evolutionary labels to retain.
        print_path_list (bool): Whether to print the input file list.
        save_float32 (bool): Whether to store grid data as float32.
        read_all_at_once (bool): Whether to load all input data at once.
        **kwargs: Additional arguments passed to
            ``regularization_on_Mini_from_files()``.

    Returns:
        tuple: Output file path and a dataframe of problematic records.
    """
    if label_list is None:
        label_list = [1]
    if path is None:
        path = _find_PARSECv2_rearranged_paths(
            photometry_suffix=photometry_suffix,
        )
    elif isinstance(path, (str, os.PathLike)):
        path = [_as_glob_pattern_string(path)]
    elif isinstance(path, (list, tuple)):
        path = [_as_glob_pattern_string(p) for p in path]
    else:
        raise ValueError('path must be None, str, Path, list, or tuple.')
    if save_path is None:
        save_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_Mini_grids'
    save_path = _as_dir_string(save_path, create=True)
    if save_name is None:
        label_suffix = _label_list_to_suffix(label_list)
        save_name = f'regularized_Mini_grid_{age_initial}_{age_end}yr_step{age_interval}_{age_type}_isochrones_{label_suffix}_{grid_type}.npz'
    save_file, problematic_df = regularization_on_Mini_from_files(
        path=path, save_path=save_path, save_name=save_name, grid_type=grid_type, label_list=label_list,
        print_path_list=print_path_list, save_float32=save_float32, read_all_at_once=read_all_at_once, **kwargs)
    return (save_file, problematic_df)

def load_default_regularized_PARSECv2_Mini_interpolator(
        npz_path=None, age_initial='1e6', age_end='2001e6', age_interval='10e6', age_type='linear',
        label_list=None, grid_type=None, force_float64=False
):
    """
    Load the default regularized PARSEC v2 Mini interpolator.

    Constructs the default grid filename from the supplied parameters when
    ``npz_path`` is None. Otherwise, loads the specified file directly and
    reads its grid type from the NPZ metadata.

    Args:
        npz_path (str or os.PathLike, optional): Path to the grid file.
        age_initial (str): Initial isochrone age in years.
        age_end (str): Final isochrone age in years.
        age_interval (str): Age sampling interval in years.
        age_type (str): Age sampling scheme.
        label_list (list, optional): Evolutionary labels used to construct
            the default filename. Defaults to ``[1]``.
        grid_type (str, optional): Grid type used to construct the default
            filename. Defaults to ``"with_inc"``.
        force_float64 (bool): Whether to convert grid arrays to float64.

    Returns:
        tuple: Interpolators, conversion functions, metadata lists, and data,
            as returned by ``load_regularized_Mini_interpolator()``.

    Raises:
        FileNotFoundError: If the specified grid file does not exist.
    """
    if npz_path is None:
        if label_list is None:
            label_list = [1]
        if grid_type is None:
            grid_type = 'with_inc'
        save_dir = DATA_DIR / 'isochrones' / 'regular_PARSECv2_Mini_grids'
        label_suffix = _label_list_to_suffix(label_list)
        save_name = f'regularized_Mini_grid_{age_initial}_{age_end}yr_step{age_interval}_{age_type}_isochrones_{label_suffix}_{grid_type}.npz'
        npz_path = save_dir / save_name
    npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f'Regularized Mini grid not found: {npz_path}')
    return load_regularized_Mini_interpolator(npz_path=npz_path, force_float64=force_float64)

def _get_u_grid_columns(grid_type):
    if grid_type == 'without_inc':
        interp_variable_list = ['Mini', 'Age', 'MH', 'omega_ini']
        interp_output_list = ['Mass', 'omega', 'logL', 'logTe', 'Req', 'Rpol']
        other_interp_variable_list = ['Age', 'MH', 'omega_ini']
    elif grid_type == 'with_inc':
        interp_variable_list = ['Mini', 'Age', 'MH', 'omega_ini', 'Inclination']
        interp_output_list = ['Mass', 'omega', 'logL', 'logTe', 'Req', 'Rpol', 'G', 'G_BP', 'G_RP', 'B_T', 'V_T', 'J', 'H', 'Ks']
        other_interp_variable_list = ['Age', 'MH', 'omega_ini', 'Inclination']
    else:
        raise ValueError('grid_type must be "without_inc" or "with_inc".')
    interp_variable_list_regular = ['u'] + other_interp_variable_list
    return (interp_variable_list, interp_output_list, other_interp_variable_list, interp_variable_list_regular)

def _process_one_u_group(args):
    group_key, g, other_interp_variable_list, interp_output_list, axis_to_idx, u_grid, dtype, min_width_threshold, interp_kind_1d, path_i, path_this = args
    result = {'status': 'skipped', 'grid_index': None, 'y_on_grid': None, 'bounds': None, 'bad_group_records': [], 'duplicated_u_records': [], 'filled_group_count': 0, 'skipped_group_count': 0}
    if len(other_interp_variable_list) == 1:
        group_key = (group_key,)
    group_dict = {c: group_key[k] for k, c in enumerate(other_interp_variable_list)}
    width_i = g['Mini_width_used'].iloc[0]
    if not np.isfinite(width_i) or width_i < min_width_threshold:
        rec = {'path_index': path_i, 'path': path_this, 'reason': 'Mini_width_used too small', 'Mini_width_used': width_i}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result
    try:
        idx = tuple((axis_to_idx[c][group_dict[c]] for c in other_interp_variable_list))
    except KeyError as e:
        rec = {'path_index': path_i, 'path': path_this, 'reason': f'axis value not found: {e}'}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result
    mini_min_vals = g['Mini_min_used'].unique()
    mini_max_vals = g['Mini_max'].unique()
    if len(mini_min_vals) != 1 or len(mini_max_vals) != 1:
        rec = {'path_index': path_i, 'path': path_this, 'reason': 'inconsistent Mini bounds'}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result
    bounds = np.array([mini_min_vals[0], mini_max_vals[0]], dtype=dtype)
    g = g.sort_values('u').copy()
    dup_mask = g['u'].duplicated(keep=False).to_numpy()
    if dup_mask.any():
        dup_u_values = g.loc[dup_mask, 'u'].unique()
        for dup_u in dup_u_values:
            g_dup = g[g['u'] == dup_u]
            rec = {'path_index': path_i, 'path': path_this, 'u': dup_u, 'n_duplicates': len(g_dup)}
            rec.update(group_dict)
            result['duplicated_u_records'].append(rec)
    g2 = g.groupby('u', as_index=False)[interp_output_list].mean()
    u_vals = g2['u'].to_numpy(dtype=np.float64)
    y_vals = g2[interp_output_list].to_numpy(dtype=np.float64)
    if len(u_vals) < 2:
        rec = {'path_index': path_i, 'path': path_this, 'reason': 'too few u points'}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result
    if not np.all(np.diff(u_vals) > 0):
        rec = {'path_index': path_i, 'path': path_this, 'reason': 'u not strictly increasing'}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result
    try:
        f = interp1d(u_vals, y_vals, axis=0, kind=interp_kind_1d, bounds_error=False, fill_value=np.nan, assume_sorted=True)
        y_on_grid = f(u_grid).astype(dtype)
        result['status'] = 'filled'
        result['grid_index'] = idx
        result['y_on_grid'] = y_on_grid
        result['bounds'] = bounds
        result['filled_group_count'] = 1
        return result
    except Exception as e:
        rec = {'path_index': path_i, 'path': path_this, 'reason': repr(e)}
        rec.update(group_dict)
        result['bad_group_records'].append(rec)
        result['skipped_group_count'] = 1
        return result

def regularization_on_u_from_files(
        path, save_path='.', save_name=None, grid_type='with_inc', label_list=None, Inclination_list=None, logAge_list=None,
        print_path_list=False, n_u=None, Mini_min_mode=None, min_width_threshold=0.01, interp_kind_1d='linear', save_float32=True,
        u_save_range=None, read_all_at_once=False, parallel_file_read=True, parallel_backend='process', parallel_interp=True,
        interp_backend='thread', max_workers=None, interp_max_workers=8, merge_age_groups=True, age_break=20.0, age_step_low=1.0,
        age_step_high=10.0, age_jitter_fraction=0.01, age_jitter_abs=1e-06):
    """
    Build and save a u-regularized PARSEC grid.

    Each input file defines an Age sampling group. In overlapping Age ranges,
    the denser sampling is retained. Nearly identical floating-point Age
    values are merged using the specified tolerances.

    Args:
        path (str, os.PathLike, list, or tuple): Input file path(s) or glob
            pattern(s).
        save_path (str or os.PathLike): Output directory.
        save_name (str, optional): Output filename.
        grid_type (str): Grid type, either ``"with_inc"`` or ``"without_inc"``.
        label_list (list, optional): Evolutionary labels to retain.
        Inclination_list (array-like, optional): Inclination values to retain.
        logAge_list (array-like, optional): Logarithmic age values to retain.
        print_path_list (bool): Whether to print the input file list.
        n_u (int, optional): Number of points in the full u grid. Defaults
            to a resolution of approximately 0.01 in Mini.
        Mini_min_mode (float, optional): Fixed lower Mini bound. Defaults
            to the minimum Mini value of each parameter group.
        min_width_threshold (float): Minimum valid Mini range.
        interp_kind_1d (str): One-dimensional interpolation method.
        save_float32 (bool): Whether to store grid arrays as float32.
        u_save_range (array-like, optional): Saved u range ``[u_min, u_max]``.
            Defaults to the full range [0, 1].
        read_all_at_once (bool): Whether to load all input data at once.
        parallel_file_read (bool): Whether to read input files in parallel.
        parallel_backend (str): Backend for parallel file reading,
            either ``"process"`` or ``"thread"``.
        parallel_interp (bool): Whether to interpolate parameter groups
            in parallel.
        interp_backend (str): Backend for parallel interpolation,
            either ``"process"`` or ``"thread"``.
        max_workers (int, optional): Maximum number of file-reading workers.
        interp_max_workers (int): Maximum number of interpolation workers.
        merge_age_groups (bool): Whether to merge and regularize Age groups.
        age_break (float): Age separating the low- and high-resolution grids.
        age_step_low (float): Age step below ``age_break``.
        age_step_high (float): Age step above ``age_break``.
        age_jitter_fraction (float): Relative tolerance for matching Age
            values.
        age_jitter_abs (float): Absolute tolerance for matching Age values.

    Returns:
        tuple: Absolute output file path, a dataframe of problematic groups,
            and a dataframe of duplicated u records.
    """
    if not read_all_at_once and parallel_file_read:
        print('[info] parallel_file_read is ignored because read_all_at_once=False.')
    save_path = str(save_path)
    if not save_path.endswith(os.sep):
        save_path += os.sep
    os.makedirs(save_path, exist_ok=True)
    if label_list is None:
        label_list = [1.0]
    if Inclination_list is None:
        Inclination_list = np.array(np.deg2rad([0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 90.0]))
    Inclination_list = np.round(Inclination_list, 3)
    if save_name is None:
        save_name = f'regularized_u_grid_{grid_type}.npz'
    dtype = np.float32 if save_float32 else np.float64
    if isinstance(path, (str, os.PathLike)):
        path_list = sorted(glob(str(path)))
    elif isinstance(path, (list, tuple)):
        files = []
        for path_0 in path:
            files += glob(str(path_0))
        path_list = sorted(files)
    else:
        raise ValueError('path must be str, Path, list, or tuple.')
    if len(path_list) == 0:
        raise ValueError(f'No files found with path pattern: {path}')
    if print_path_list:
        print(path_list)
    interp_variable_list, interp_output_list, other_interp_variable_list, interp_variable_list_regular = _get_u_grid_columns(grid_type)
    if 'Age' not in other_interp_variable_list:
        raise ValueError('This u-grid does not use Age as its age coordinate.')
    required_cols = interp_variable_list + interp_output_list
    df_all = None
    if read_all_at_once:
        if parallel_file_read:
            print('Reading all files at once in parallel...')
            read_args = [(path_i, path_list[path_i], label_list, Inclination_list, logAge_list) for path_i in range(len(path_list))]
            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(path_list))
            if parallel_backend == 'process':
                Executor = ProcessPoolExecutor
            elif parallel_backend == 'thread':
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")
            with Executor(max_workers=max_workers) as executor:
                df_list = list(tqdm(executor.map(_read_one_parsec_file_for_regularization, read_args), total=len(read_args)))
        else:
            print('Reading all files at once...')
            df_list = []
            for path_i in tqdm(range(len(path_list))):
                df_i = PARSECv2_collect_data([path_list[path_i]], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
                df_i['path_index'] = path_i
                df_i['path'] = path_list[path_i]
                df_list.append(df_i)
        df_all = pd.concat(df_list, ignore_index=True)
        del df_list
        if grid_type == 'with_inc':
            df_all['Inclination'] = np.round(df_all['Inclination'].values, 3)
        memory_bytes = df_all.memory_usage(deep=True).sum()
        print(f'df_all shape: {df_all.shape}')
        print(f'df_all memory: {memory_bytes / 1024 ** 2:.2f} MB ({memory_bytes / 1024 ** 3:.2f} GB)')
    print('Collecting axis values and Mini bounds...')
    Age_groups = [None for _ in path_list]
    axis_values = {c: [] for c in other_interp_variable_list if c != 'Age'}
    if read_all_at_once:
        missing_cols = [c for c in required_cols if c not in df_all.columns]
        if missing_cols:
            raise ValueError(f'Missing columns in df_all: {missing_cols}')
        for path_i in range(len(path_list)):
            df_i = df_all.loc[df_all['path_index'] == path_i]
            Age_groups[path_i] = df_i['Age'].dropna().unique()
        for c in axis_values:
            axis_values[c].append(df_all[c].dropna().unique())
    else:
        for path_i in tqdm(range(len(path_list))):
            df = PARSECv2_collect_data([path_list[path_i]], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
            if grid_type == 'with_inc':
                df['Inclination'] = np.round(df['Inclination'].values, 3)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f'Missing columns in df from {path_list[path_i]}: {missing_cols}')
            Age_groups[path_i] = df['Age'].dropna().unique()
            for c in axis_values:
                axis_values[c].append(df[c].dropna().unique())
            del df
    if merge_age_groups:
        Age_grid, age_selection, age_group_summary = _build_merged_age_grid(Age_groups, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs, verbose=True)
    else:
        all_ages = np.concatenate(Age_groups)
        Age_grid, _ = _collapse_close_ages(all_ages, age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
        age_selection = []
        for group_index, ages in enumerate(Age_groups):
            raw_to_final = {}
            for age in np.unique(ages):
                idx = int(np.argmin(np.abs(Age_grid - age)))
                raw_to_final[float(age)] = float(Age_grid[idx])
            age_selection.append({'group_index': group_index, 'raw_to_final': raw_to_final})
        age_group_summary = pd.DataFrame()
    axis_grids = {'Age': Age_grid}
    for c, value_list in axis_values.items():
        axis_grids[c] = np.sort(np.unique(np.concatenate(value_list)))
    if grid_type == 'with_inc' and 'Inclination' in axis_grids:
        axis_grids['Inclination'] = np.round(np.sort(np.asarray(Inclination_list, dtype=np.float64)), 3)
    age_selection_kwargs = dict(age_break=age_break, age_step_low=age_step_low, age_step_high=age_step_high, age_jitter_fraction=age_jitter_fraction, age_jitter_abs=age_jitter_abs)
    if read_all_at_once:
        old_rows = len(df_all)
        df_all = _apply_age_selection_to_df_all(df_all, age_selection, **age_selection_kwargs)
        print(f'Age-group filtering: {old_rows:,} -> {len(df_all):,} rows')
    bounds_records = []
    if read_all_at_once:
        df_bounds_source = df_all[required_cols + ['path_index', 'path']].dropna().copy()
        df_bounds = df_bounds_source.groupby(other_interp_variable_list)['Mini'].agg(Mini_min_orig='min', Mini_max='max').reset_index()
        df_bounds['path_index'] = -1
        df_bounds['path'] = 'df_all'
        del df_bounds_source
    else:
        for path_i in tqdm(range(len(path_list)), desc='Collecting Mini bounds'):
            df = PARSECv2_collect_data([path_list[path_i]], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
            if grid_type == 'with_inc':
                df['Inclination'] = np.round(df['Inclination'].values, 3)
            df['path_index'] = path_i
            df['path'] = path_list[path_i]
            df = _apply_age_selection_to_one_df(df, path_i, age_selection, **age_selection_kwargs)
            if len(df) == 0:
                continue
            df = df[required_cols].dropna().copy()
            if len(df) == 0:
                continue
            df_bounds_i = df.groupby(other_interp_variable_list)['Mini'].agg(Mini_min_orig='min', Mini_max='max').reset_index()
            df_bounds_i['path_index'] = path_i
            df_bounds_i['path'] = path_list[path_i]
            bounds_records.append(df_bounds_i)
            del df
        if len(bounds_records) == 0:
            raise ValueError('No valid Mini bounds remain after Age-group filtering.')
        df_bounds = pd.concat(bounds_records, ignore_index=True)
    if Mini_min_mode is None:
        df_bounds['Mini_min_used'] = df_bounds['Mini_min_orig']
    else:
        warnings.warn(
            "Mini_min_mode is experimental and not recommended for general use. "
            "It may introduce artifacts near the lower mass boundary and affect "
            "interpolation accuracy. Use Mini_min_mode=None (default) unless "
            "you have validated the resulting grid.",
            UserWarning,
            stacklevel=2,
        )
        df_bounds['Mini_min_used'] = float(Mini_min_mode)
    df_bounds['Mini_width_used'] = df_bounds['Mini_max'] - df_bounds['Mini_min_used']
    bad_width_df = df_bounds.loc[df_bounds['Mini_width_used'] < min_width_threshold].copy()
    if len(bad_width_df) > 0:
        bad_width_path = save_path + f'bad_u_width_records_' + f'{Path(save_name).stem}.csv'
        bad_width_df.to_csv(bad_width_path, index=False)
        print(f'Warning: bad width records saved to: {os.path.abspath(bad_width_path)}')
    Mini_global_min = np.nanmin(df_bounds['Mini_min_orig'].to_numpy(dtype=np.float64))
    Mini_global_max = np.nanmax(df_bounds['Mini_max'].to_numpy(dtype=np.float64))
    if not np.isfinite(Mini_global_min) or not np.isfinite(Mini_global_max):
        raise ValueError('Cannot determine global Mini range.')
    if Mini_global_max <= Mini_global_min:
        raise ValueError(f'Invalid global Mini range: {Mini_global_min}, {Mini_global_max}')
    n_u_default = int(np.ceil((Mini_global_max - Mini_global_min) / 0.01)) + 1
    if n_u is None:
        n_u = n_u_default
    else:
        n_u = int(n_u)
        n_u_min_recommended = int(np.ceil(n_u_default * 0.5))
        if n_u < n_u_min_recommended:
            raise ValueError(f'n_u is too small. n_u >= {n_u_min_recommended} is recommended. Default: {n_u_default}.')
    u_grid_full = np.linspace(0.0, 1.0, n_u, dtype=np.float64)
    if u_save_range is None:
        u_grid = u_grid_full
        u_save_min = 0.0
        u_save_max = 1.0
    else:
        if not isinstance(u_save_range, (list, tuple, np.ndarray)) or len(u_save_range) != 2:
            raise ValueError('u_save_range must be None or [u_min, u_max].')
        u_save_min = float(u_save_range[0])
        u_save_max = float(u_save_range[1])
        if not 0.0 <= u_save_min < u_save_max <= 1.0:
            raise ValueError('u_save_range must satisfy 0 <= u_min < u_max <= 1.')
        u_mask = (u_grid_full >= u_save_min) & (u_grid_full <= u_save_max)
        u_grid = u_grid_full[u_mask]
        if len(u_grid) < 2:
            raise ValueError(f'u_save_range={u_save_range} leaves fewer than 2 points.')
    print(f'Mini global range: [{Mini_global_min}, {Mini_global_max}]')
    print(f'n_u_default for 0.01 Mini resolution: {n_u_default}')
    print(f'n_u used: {n_u}')
    param_sizes = [len(axis_grids[c]) for c in other_interp_variable_list]
    n_out = len(interp_output_list)
    grid_outputs = np.full([len(u_grid)] + param_sizes + [n_out], np.nan, dtype=dtype)
    grid_bounds = np.full(param_sizes + [2], np.nan, dtype=dtype)
    axis_to_idx = {c: {v: i for i, v in enumerate(axis_grids[c])} for c in other_interp_variable_list}
    print(f'u_grid shape: {u_grid.shape}')
    for c in other_interp_variable_list:
        print(f'{c}_grid shape: {axis_grids[c].shape}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_bounds shape: {grid_bounds.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_outputs memory: {grid_outputs.nbytes / 1024 ** 2:.2f} MB ({grid_outputs.nbytes / 1024 ** 3:.2f} GB)')
    print('Filling u-regular grids...')
    filled_group_count = 0
    skipped_group_count = 0
    bad_group_records = []
    duplicated_u_records = []
    if read_all_at_once:
        file_iter = [(-1, 'df_all')]
    else:
        file_iter = tqdm([(path_i, path_list[path_i]) for path_i in range(len(path_list))])
    for path_i, path_this in file_iter:
        if read_all_at_once:
            df = df_all.copy()
        else:
            df = PARSECv2_collect_data([path_this], label_list=label_list, Inclination_list=Inclination_list, logAge_list=logAge_list, show_progress=False)
            if grid_type == 'with_inc':
                df['Inclination'] = np.round(df['Inclination'].values, 3)
            df['path_index'] = path_i
            df['path'] = path_this
            df = _apply_age_selection_to_one_df(df, path_i, age_selection, **age_selection_kwargs)
        if len(df) == 0:
            continue
        df = df[required_cols + ['path_index', 'path']].dropna().copy()
        if len(df) == 0:
            continue
        df = df.sort_values(other_interp_variable_list + ['Mini']).reset_index(drop=True)
        df_bounds_i = df.groupby(other_interp_variable_list)['Mini'].agg(Mini_min_orig='min', Mini_max='max').reset_index()
        if Mini_min_mode is None:
            df_bounds_i['Mini_min_used'] = df_bounds_i['Mini_min_orig']
        else:
            warnings.warn(
                "Mini_min_mode is experimental and not recommended for general use. "
                "It may introduce artifacts near the lower mass boundary and affect "
                "interpolation accuracy. Use Mini_min_mode=None (default) unless "
                "you have validated the resulting grid.",
                UserWarning,
                stacklevel=2,
            )
            df_bounds_i['Mini_min_used'] = float(Mini_min_mode)
        df_bounds_i['Mini_width_used'] = df_bounds_i['Mini_max'] - df_bounds_i['Mini_min_used']
        df = df.merge(df_bounds_i, on=other_interp_variable_list, how='left')
        width = df['Mini_width_used'].to_numpy(dtype=np.float64)
        Mini_vals = df['Mini'].to_numpy(dtype=np.float64)
        Mini_min_used = df['Mini_min_used'].to_numpy(dtype=np.float64)
        u_raw = np.full_like(Mini_vals, np.nan, dtype=np.float64)
        valid_width = np.isfinite(width) & (width >= min_width_threshold)
        u_raw[valid_width] = (Mini_vals[valid_width] - Mini_min_used[valid_width]) / width[valid_width]
        df['u_raw'] = u_raw
        df['u'] = np.clip(u_raw, 0.0, 1.0)
        grouped = list(df.groupby(other_interp_variable_list, sort=False))
        task_args = [(group_key, g, other_interp_variable_list, interp_output_list, axis_to_idx, u_grid, dtype, min_width_threshold, interp_kind_1d, path_i, path_this) for group_key, g in grouped]
        if parallel_interp:
            if interp_backend == 'process':
                Executor = ProcessPoolExecutor
            elif interp_backend == 'thread':
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("interp_backend must be 'process' or 'thread'.")
            with Executor(max_workers=interp_max_workers) as executor:
                results = list(tqdm(executor.map(_process_one_u_group, task_args), total=len(task_args)))
        else:
            results = [_process_one_u_group(arg) for arg in tqdm(task_args)]
        for res in results:
            bad_group_records.extend(res['bad_group_records'])
            duplicated_u_records.extend(res['duplicated_u_records'])
            skipped_group_count += res['skipped_group_count']
            if res['status'] != 'filled':
                continue
            idx = res['grid_index']
            y_on_grid = res['y_on_grid']
            bounds = res['bounds']
            grid_bounds[idx + (slice(None),)] = bounds
            grid_outputs[(slice(None),) + idx + (slice(None),)] = y_on_grid
            filled_group_count += res['filled_group_count']
        del df
    if read_all_at_once:
        del df_all
    bad_group_df = pd.DataFrame(bad_group_records)
    duplicated_u_df = pd.DataFrame(duplicated_u_records)
    bad_group_path = save_path + f'bad_records_' + f'{Path(save_name).stem}.csv'
    duplicated_u_path = save_path + f'duplicated_records_' + f'{Path(save_name).stem}.csv'
    age_summary_path = save_path + f'age_group_summary_' + f'{Path(save_name).stem}.csv'
    bad_group_df.to_csv(bad_group_path, index=False)
    duplicated_u_df.to_csv(duplicated_u_path, index=False)
    if len(age_group_summary) > 0:
        age_group_summary = age_group_summary.copy()
        age_group_summary['path'] = path_list
        age_group_summary.to_csv(age_summary_path, index=False)
    save_file = save_path + save_name
    save_dict = dict(
        u_grid=u_grid, grid_outputs=grid_outputs, grid_bounds=grid_bounds, interp_output_list=np.array(interp_output_list, dtype=object),
        interp_variable_list=np.array(interp_variable_list, dtype=object), interp_variable_list_regular=np.array(interp_variable_list_regular, dtype=object),
        other_interp_variable_list=np.array(other_interp_variable_list, dtype=object), grid_type=np.array(grid_type), label_list=np.array(label_list),
        save_float32=np.array(save_float32), read_all_at_once=np.array(read_all_at_once), parallel_file_read=np.array(parallel_file_read),
        parallel_backend=np.array(parallel_backend), parallel_interp=np.array(parallel_interp), interp_backend=np.array(interp_backend),
        grid_dtype=np.array(str(grid_outputs.dtype)), filled_group_count=np.array(filled_group_count), skipped_group_count=np.array(skipped_group_count),
        u_save_range=np.array([u_save_min, u_save_max]), u_grid_was_cropped=np.array(u_save_range is not None), n_u_original=np.array(n_u),
        n_u_saved=np.array(len(u_grid)), merge_age_groups=np.array(merge_age_groups), age_break=np.array(age_break), age_step_low=np.array(age_step_low),
        age_step_high=np.array(age_step_high), age_jitter_fraction=np.array(age_jitter_fraction), age_jitter_abs=np.array(age_jitter_abs))
    for c in other_interp_variable_list:
        save_dict[f'{c}_grid'] = axis_grids[c]
    np.savez_compressed(save_file, **save_dict)
    total_bytes = grid_outputs.nbytes + grid_bounds.nbytes + u_grid.nbytes
    for c in other_interp_variable_list:
        total_bytes += axis_grids[c].nbytes
    print()
    print('Finished u-grid regularization.')
    print(f'grid_type: {grid_type}')
    print(f'merge_age_groups: {merge_age_groups}')
    print(f'Age_grid shape: {Age_grid.shape}')
    print(f'Age_grid range: [{Age_grid.min()}, {Age_grid.max()}]')
    print(f'read_all_at_once: {read_all_at_once}')
    print(f'parallel_file_read: {parallel_file_read}')
    print(f'parallel_backend: {parallel_backend}')
    print(f'parallel_interp: {parallel_interp}')
    print(f'interp_backend: {interp_backend}')
    print(f'max_workers: {max_workers}')
    print(f'interp_max_workers: {interp_max_workers}')
    print(f'save_float32: {save_float32}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'u_grid shape: {u_grid.shape}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_bounds shape: {grid_bounds.shape}')
    print(f'filled_group_count: {filled_group_count}')
    print(f'skipped_group_count: {skipped_group_count}')
    print(f'bad_group_count: {len(bad_group_df)}')
    print(f'duplicated_u_count: {len(duplicated_u_df)}')
    print(f'bad group records saved at:\n{os.path.abspath(bad_group_path)}')
    print(f'duplicated u records saved at:\n{os.path.abspath(duplicated_u_path)}')
    if len(age_group_summary) > 0:
        print(f'Age group summary saved at:\n{os.path.abspath(age_summary_path)}')
    print(f'u_save_range: [{u_save_min}, {u_save_max}]')
    print(f'n_u_original: {n_u}')
    print(f'n_u_saved: {len(u_grid)}')
    print(f'Grids are saved at:\n{os.path.abspath(save_file)}')
    print(f'total grid memory: {total_bytes / 1024 ** 2:.2f} MB ({total_bytes / 1024 ** 3:.2f} GB)')
    return (os.path.abspath(save_file), bad_group_df, duplicated_u_df)

def load_regularized_u_interpolator(npz_path, force_float64=False):
    """
    Load a u-regularized PARSEC grid and construct interpolation functions.

    The function loads the regularized grid and constructs interpolators for
    the parameter-dependent Mini bounds and the interpolated outputs.

    Args:
        npz_path (str or os.PathLike): Path to the saved NPZ file.
        force_float64 (bool): Whether to convert the output grid and bounds
            to float64 after loading.

    Returns:
        tuple: A tuple containing:
            - interp_func: Interpolator from physical parameters to outputs.
            - interp_u: Interpolator from physical parameters to normalized u.
            - interp_u_to_outputs: Interpolator from u and other parameters
              to outputs.
            - interp_u_bounds: Interpolator for the valid Mini bounds.
            - interp_variable_list (list): Names of the input parameters.
            - interp_output_list (list): Names of the interpolated outputs.
            - data (NpzFile): The loaded NPZ data.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_type = str(data['grid_type'])
    u_grid = data['u_grid']
    grid_outputs = data['grid_outputs']
    grid_bounds = data['grid_bounds']
    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)
        grid_bounds = grid_bounds.astype(np.float64)
    interp_output_list = data['interp_output_list'].astype(str).tolist()
    interp_variable_list = data['interp_variable_list'].astype(str).tolist()
    interp_variable_list_regular = data['interp_variable_list_regular'].astype(str).tolist()
    other_interp_variable_list = data['other_interp_variable_list'].astype(str).tolist()
    param_grids = []
    for c in other_interp_variable_list:
        param_grids.append(data[f'{c}_grid'])
    interp_u_bounds = RegularGridInterpolator(tuple(param_grids), grid_bounds, bounds_error=False, fill_value=np.nan)
    interp_u_to_outputs = RegularGridInterpolator(tuple([u_grid] + param_grids), grid_outputs, bounds_error=False, fill_value=np.nan)
    n_param = len(interp_variable_list)

    def interp_u(x, eps=1e-12, clip_u=False, return_bounds=False):
        """
        Convert physical parameters to normalized u coordinates.

        Args:
            x (array-like): Input parameters with shape (n_param,) or
                (N, n_param). Mini must be the first column, followed by
                the remaining interpolation parameters.
            eps (float): Minimum valid width of the Mini interval.
            clip_u (bool): Whether to clip normalized u values to [0, 1].
            return_bounds (bool): Whether to also return the interpolated
                Mini bounds.

        Returns:
            ndarray or tuple: Normalized u values, optionally accompanied
                by the corresponding Mini bounds. A single input returns
                scalar-shaped results.
        """
        x = np.asarray(x, dtype=np.float64)
        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}).')
            x = x[None, :]
            single_input = True
        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}).')
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
            return (u, bounds)
        return u

    def interp_func(x, eps=1e-12, clip_u=False, return_u=False, return_bounds=False):
        """
        Interpolate grid outputs from physical parameters.

        Args:
            x (array-like): Input parameters with shape (n_param,) or
                (N, n_param). Mini must be the first column.
            eps (float): Minimum valid width of the Mini interval.
            clip_u (bool): Whether to clip normalized u values to [0, 1].
            return_u (bool): Whether to return the normalized u values.
            return_bounds (bool): Whether to return the interpolated
                Mini bounds.

        Returns:
            ndarray or tuple: Interpolated outputs, optionally accompanied
                by normalized u values and/or Mini bounds. A single input
                returns a one-dimensional output array.
        """
        x = np.asarray(x, dtype=np.float64)
        single_input = False
        if x.ndim == 1:
            if x.shape[0] != n_param:
                raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}), while x.shape[0]={x.shape[0]}')
            x = x[None, :]
            single_input = True
        if x.ndim != 2 or x.shape[1] != n_param:
            raise ValueError(f'Input must have shape ({n_param},) or (N, {n_param}).')
        u, bounds = interp_u(x, eps=eps, clip_u=clip_u, return_bounds=True)
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
            return (y, u, bounds)
        elif return_u:
            return (y, u)
        elif return_bounds:
            return (y, bounds)
        else:
            return y
    total_bytes = grid_outputs.nbytes + grid_bounds.nbytes + u_grid.nbytes
    for g in param_grids:
        total_bytes += g.nbytes
    print('Loaded u-regularized grid.')
    print(f'grid_type: {grid_type}')
    if 'label_list' in data.files:
        print(f"label_list: {data['label_list'].tolist()}")
    print(f'interp_variable_list: {interp_variable_list}')
    print(f'interp_variable_list_regular: {interp_variable_list_regular}')
    print(f'interp_output_list: {interp_output_list}')
    print(f'u_grid shape: {u_grid.shape}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_bounds shape: {grid_bounds.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_bounds dtype: {grid_bounds.dtype}')
    if 'read_all_at_once' in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")
    if 'save_float32' in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")
    if 'u_save_range' in data.files:
        print(f"u_save_range: {data['u_save_range'].tolist()}")
    if 'u_grid_was_cropped' in data.files:
        print(f"u_grid_was_cropped: {bool(data['u_grid_was_cropped'])}")
    if 'n_u_original' in data.files and 'n_u_saved' in data.files:
        print(f"n_u_original: {int(data['n_u_original'])}")
        print(f"n_u_saved: {int(data['n_u_saved'])}")
    print(f'total grid memory: {total_bytes / 1024 ** 2:.2f} MB ({total_bytes / 1024 ** 3:.2f} GB)')
    return (interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variable_list, interp_output_list, data)

def regularize_PARSECv2_u_grid(
        path=None, save_path=None, save_name=None, age_initial='1e6', age_end='2001e6', age_interval='10e6', age_type='linear',
        photometry_suffix='2mass_tycho2_gaia_dr3', grid_type='with_inc', label_list=None, min_width_threshold=0.01,
        save_float32=True, read_all_at_once=False, **kwargs):
    """
    Generate and save a u-regularized PARSEC v2.0 isochrone grid.

    This is a wrapper for ``regularization_on_u_from_files()`` that provides
    default input paths, output directories, and filenames.

    Default input:
        DATA_DIR/isochrones/PARSEC_v2/
        1e6_2001e6yr_linear_isochrones_rearranged (2mass & tycho2 & gaia dr3)/*

    Default output:
        DATA_DIR/isochrones/regular_PARSECv2_u_grids/

    Args:
        path (str, os.PathLike, list, or tuple, optional): Input file path(s)
            or glob pattern(s). If None, paths are automatically located
            using ``photometry_suffix``.
        save_path (str or os.PathLike, optional): Output directory. Defaults
            to the standard PARSEC v2.0 u-grid directory.
        save_name (str, optional): Output filename. If None, it is generated
            from the age range, sampling interval, grid type, and labels.
        age_initial (str): Initial age used in the default filename.
        age_end (str): Final age used in the default filename.
        age_interval (str): Age sampling interval used in the default
            filename.
        age_type (str): Age sampling type used in the default filename.
        photometry_suffix (str): Photometry suffix used to locate default
            input files.
        grid_type (str): Grid type, either ``"with_inc"`` or
            ``"without_inc"``.
        label_list (list, optional): Evolutionary labels to retain.
            Defaults to ``[1]``.
        min_width_threshold (float): Minimum valid Mini range.
        save_float32 (bool): Whether to store grid arrays as float32.
        read_all_at_once (bool): Whether to load all input data at once.
        **kwargs: Additional keyword arguments passed to
            ``regularization_on_u_from_files()``.

    Returns:
        tuple: Output file path, a dataframe of problematic parameter groups,
            and a dataframe of duplicated u records.
    """
    if label_list is None:
        label_list = [1]
    if path is None:
        path = _find_PARSECv2_rearranged_paths(
            photometry_suffix=photometry_suffix,
        )
    elif isinstance(path, (str, os.PathLike)):
        path = [_as_glob_pattern_string(path)]
    elif isinstance(path, (list, tuple)):
        path = [_as_glob_pattern_string(p) for p in path]
    else:
        raise ValueError('path must be None, str, Path, list, or tuple.')
    if save_path is None:
        save_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_u_grids'
    save_path = _as_dir_string(save_path, create=True)
    if save_name is None:
        label_suffix = _label_list_to_suffix(label_list)
        save_name = f'regularized_u_grid_{age_initial}_{age_end}yr_step{age_interval}_{age_type}_isochrones_{label_suffix}_{grid_type}.npz'
    save_file, bad_group_df, duplicated_u_df = regularization_on_u_from_files(
        path=path, save_path=save_path, save_name=save_name, grid_type=grid_type, label_list=label_list, min_width_threshold=min_width_threshold,
        save_float32=save_float32, read_all_at_once=read_all_at_once, **kwargs)
    return (save_file, bad_group_df, duplicated_u_df)

def load_default_regularized_PARSECv2_u_interpolator(
        npz_path=None, age_initial='1e6', age_end='2001e6', age_interval='10e6', age_type='linear',
        label_list=None, grid_type=None, force_float64=False):
    """
    Load a regularized PARSEC v2.0 u-grid interpolator.

    If ``npz_path`` is None, the default file path is constructed from
    the specified age range, sampling interval, evolutionary labels, and
    grid type.

    If ``npz_path`` is provided, ``label_list`` and ``grid_type`` are ignored
    for path selection. The actual grid type is read from the NPZ metadata.

    Args:
        npz_path (str or os.PathLike, optional): Path to the NPZ file.
            If None, the default path is constructed automatically.
        age_initial (str): Initial age used in the default filename.
        age_end (str): Final age used in the default filename.
        age_interval (str): Age sampling interval used in the default
            filename.
        age_type (str): Age sampling type used in the default filename.
        label_list (list, optional): Evolutionary labels used to construct
            the default filename. Defaults to ``[1]``.
        grid_type (str, optional): Grid type used to construct the default
            filename. Defaults to ``"with_inc"``.
        force_float64 (bool): Whether to convert the output grid and bounds
            to float64 when loading the interpolator.

    Returns:
        tuple: The return value of ``load_regularized_u_interpolator()``,
            containing the interpolation functions, parameter and output
            variable lists, and loaded NPZ data.

    Raises:
        FileNotFoundError: If the specified or constructed NPZ file does
            not exist.
    """
    if npz_path is None:
        if label_list is None:
            label_list = [1]
        if grid_type is None:
            grid_type = 'with_inc'
        save_dir = DATA_DIR / 'isochrones' / 'regular_PARSECv2_u_grids'
        label_suffix = _label_list_to_suffix(label_list)
        save_name = f'regularized_u_grid_{age_initial}_{age_end}yr_step{age_interval}_{age_type}_isochrones_{label_suffix}_{grid_type}.npz'
        npz_path = save_dir / save_name
    npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f'Regularized PARSEC v2 u grid not found: {npz_path}')
    return load_regularized_u_interpolator(npz_path=npz_path, force_float64=force_float64)


def read_MIST(
    file_path, add_iso_index=True, add_initial_params=False,
    omega_col="omega_ini", feh_col="[Fe/H]_init",
    strict_omega=False, strict_feh=False, return_file_type=False
):
    """Read a MIST isochrone or photometry file.

    The file type is identified from its column names rather than its filename.

    For photometry files, ``omega_ini`` is always added from the MIST header.
    For isochrone files, ``omega_ini`` and ``[Fe/H]_init`` are added only
    when ``add_initial_params=True``.

    Args:
        file_path (str or pathlib.Path): Path to a MIST ``.iso`` or
            ``.iso.cmd`` file.
        add_iso_index (bool): Whether to add an ``iso_index`` column for
            each isochrone block.
        add_initial_params (bool): Whether to add ``omega_ini`` and
            ``[Fe/H]_init`` to isochrone files. Photometry files always
            include ``omega_ini``.
        omega_col (str): Column name for the initial rotation rate
            (v/vcrit).
        feh_col (str): Column name for the initial metallicity.
        strict_omega (bool): Whether to raise a ValueError if ``v/vcrit``
            cannot be parsed from the header.
        strict_feh (bool): Whether to raise a ValueError if ``[Fe/H]``
            cannot be parsed from the header.
        return_file_type (bool): Whether to also return the detected file
            type (``"iso"`` or ``"phot"``).

    Returns:
        pandas.DataFrame or tuple: Combined DataFrame. If
        ``return_file_type=True``, returns ``(df_all, file_type)``.
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # Parse initial parameters from the MIST header.
    omega_ini = np.nan
    feh_ini = np.nan

    for i, line in enumerate(lines):
        s = line.strip()
        if "[Fe/H]" not in s and "v/vcrit" not in s:
            continue

        header = s.lstrip("#").split()
        for j in range(i + 1, len(lines)):
            value_line = lines[j].strip()
            if not value_line:
                continue
            if not value_line.startswith("#"):
                break

            values = value_line.lstrip("#").split()
            if len(values) != len(header):
                continue

            if "[Fe/H]" in header:
                try:
                    feh_ini = float(values[header.index("[Fe/H]")])
                except (ValueError, IndexError):
                    pass

            if "v/vcrit" in header:
                try:
                    omega_ini = float(values[header.index("v/vcrit")])
                except (ValueError, IndexError):
                    pass
            break

        if np.isfinite(omega_ini) and np.isfinite(feh_ini):
            break

    if strict_omega and not np.isfinite(omega_ini):
        raise ValueError(f"Cannot parse v/vcrit from MIST header: {file_path}")
    if strict_feh and not np.isfinite(feh_ini):
        raise ValueError(f"Cannot parse [Fe/H] from MIST header: {file_path}")

    # Split the file into isochrone blocks.
    blocks = []
    current_block = None

    for line in lines:
        s = line.strip()
        if s.startswith("# number of EEPs, cols"):
            if current_block:
                blocks.append(current_block)
            current_block = [line]
        elif current_block is not None:
            current_block.append(line)

    if current_block:
        blocks.append(current_block)

    # Read individual blocks and identify the file type.
    dfs = []
    file_type = None

    phot_markers = {
        "Bessell_U", "Bessell_B", "Bessell_V",
        "2MASS_J", "2MASS_H", "2MASS_Ks",
        "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3",
    }
    iso_markers = {
        "log_R", "surf_avg_omega", "surf_avg_omega_div_omega_crit",
        "center_h1", "he_core_mass",
    }

    for i, block in enumerate(blocks):
        columns = None
        data = []
        data_started = False

        for line in block:
            s = line.strip()
            if not s:
                continue
            if s.startswith("# EEP"):
                columns = s[1:].strip().split()
                data_started = True
                continue
            if not data_started or s.startswith("#"):
                continue

            parts = s.split()
            if columns is not None and len(parts) == len(columns):
                data.append(parts)

        if columns is None or not data:
            continue

        if file_type is None:
            col_set = set(columns)
            if col_set & phot_markers:
                file_type = "phot"
            elif col_set & iso_markers:
                file_type = "iso"
            else:
                raise ValueError(
                    "Cannot determine whether this is a MIST iso or phot "
                    f"file from columns in: {file_path}"
                )

        df = pd.DataFrame(np.asarray(data, dtype=np.float64), columns=columns)
        if add_iso_index:
            df["iso_index"] = i
        dfs.append(df)

    if not dfs:
        df_all = pd.DataFrame()
        return (df_all, file_type) if return_file_type else df_all

    # Combine all isochrone blocks.
    df_all = pd.concat(dfs, ignore_index=True, copy=False)

    # Add initial parameters when required.
    extra_cols = {}

    if file_type == "phot":
        if omega_col not in df_all.columns:
            extra_cols[omega_col] = np.full(len(df_all), omega_ini, dtype=float)

    elif file_type == "iso" and add_initial_params:
        if omega_col not in df_all.columns:
            extra_cols[omega_col] = np.full(len(df_all), omega_ini, dtype=float)
        if feh_col not in df_all.columns:
            extra_cols[feh_col] = np.full(len(df_all), feh_ini, dtype=float)

    if extra_cols:
        df_all = pd.concat(
            [df_all.reset_index(drop=True), pd.DataFrame(extra_cols)],
            axis=1, copy=False
        )

    return (df_all, file_type) if return_file_type else df_all


def MIST_merge_isos_phot(df_isos, df_phot, check_order=True):
    """
    Merge MIST isochrone and photometry DataFrames.

    Uses ``df_isos`` as the base and appends columns that are present only
    in ``df_phot``. Common columns are used to match the rows.

    Args:
        df_isos (pd.DataFrame): Isochrone DataFrame.
        df_phot (pd.DataFrame): Photometry DataFrame.
        check_order (bool): If True, require common-column values to match
            row by row. If False, merge by common columns regardless of
            row order.

    Returns:
        pd.DataFrame: Merged DataFrame containing all columns from
            ``df_isos`` and the additional columns from ``df_phot``.

    Raises:
        ValueError: If there are no common columns, row counts differ,
            common-column values do not match when ``check_order=True``,
            or duplicate matching keys prevent a one-to-one merge.
    """
    common_cols = [c for c in df_isos.columns if c in df_phot.columns]
    phot_only_cols = [c for c in df_phot.columns if c not in common_cols]
    if len(common_cols) == 0:
        raise ValueError('df_isos and df_phot have no common columns.')
    if check_order:
        if len(df_isos) != len(df_phot):
            raise ValueError(f'Row count mismatch: len(df_isos)={len(df_isos)}, len(df_phot)={len(df_phot)}')
        is_same = df_isos[common_cols].equals(df_phot[common_cols])
        if not is_same:
            diff_mask = df_isos[common_cols] != df_phot[common_cols]
            diff_mask = diff_mask.fillna(False)
            bad_rows = diff_mask.any(axis=1)
            first_bad_idx = bad_rows.idxmax() if bad_rows.any() else None
            raise ValueError(f'Common columns are not identical row by row. First mismatch at row index: {first_bad_idx}')
        df_merged = pd.concat([df_isos.reset_index(drop=True), df_phot[phot_only_cols].reset_index(drop=True)], axis=1)
    else:
        if df_phot.duplicated(subset=common_cols).any():
            raise ValueError('df_phot has duplicated rows in common columns; cannot safely merge.')
        if df_isos.duplicated(subset=common_cols).any():
            raise ValueError('df_isos has duplicated rows in common columns; cannot safely merge.')
        df_merged = df_isos.merge(df_phot[common_cols + phot_only_cols], on=common_cols, how='left', validate='one_to_one')
    return df_merged

def parse_mist_filename(file_path):
    """
    Parse metallicity, alpha enhancement, and rotation from a MIST filename.

    Supported filename patterns include:
        MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd
        MIST_v1.2_feh_m0.25_afe_p0.0_vvcrit0.4_full.iso

    Args:
        file_path (str or os.PathLike): Path to the MIST file.

    Returns:
        dict: Parsed filename information containing:
            - path (Path): Resolved filename path.
            - feh_tag (str): Initial metallicity tag.
            - afe_tag (str): Alpha-enhancement tag.
            - vvcrit (float): Initial rotation rate (v/vcrit).
            - kind (str): File type, either ``"phot"``, ``"full"``,
              or ``"unknown"``.
            - key (tuple): Parameter key ``(feh_tag, afe_tag, vvcrit)``.

    Raises:
        ValueError: If the metallicity, alpha-enhancement, or rotation
            parameter cannot be parsed from the filename.
    """
    file_path = Path(file_path)
    name = file_path.name
    feh_match = re.search('feh_([mp][0-9]+(?:\\.[0-9]+)?)', name)
    afe_match = re.search('afe_([mp][0-9]+(?:\\.[0-9]+)?)', name)
    vvcrit_match = re.search('vvcrit([0-9]+(?:\\.[0-9]+)?)', name)
    if feh_match is None:
        raise ValueError(f'Cannot parse feh tag from filename: {name}')
    if afe_match is None:
        raise ValueError(f'Cannot parse afe tag from filename: {name}')
    if vvcrit_match is None:
        raise ValueError(f'Cannot parse vvcrit from filename: {name}')
    feh_tag = feh_match.group(1)
    afe_tag = afe_match.group(1)
    vvcrit = float(vvcrit_match.group(1))
    if name.endswith('_UBVRIplus.iso.cmd'):
        kind = 'phot'
    elif name.endswith('_full.iso'):
        kind = 'full'
    else:
        kind = 'unknown'
    key = (feh_tag, afe_tag, vvcrit)
    return {'path': file_path, 'feh_tag': feh_tag, 'afe_tag': afe_tag, 'vvcrit': vvcrit, 'kind': kind, 'key': key}

def find_mist_file_pairs(mist_root_or_folders, verbose=True):
    """
    Find and pair MIST full isochrone and photometry files.

    Args:
        mist_root_or_folders (str, os.PathLike, or list): A root directory
            containing subdirectories, a single directory, or a list of
            directories to search recursively.
        verbose (bool): Whether to print the number of discovered files,
            matched pairs, and unmatched files.

    Returns:
        list of dict: Matched file pairs. Each dictionary contains:
            - key (tuple): Parameter key ``(feh_tag, afe_tag, vvcrit)``.
            - full_file (Path): Path to the full isochrone file.
            - phot_file (Path): Path to the photometry file.

    Raises:
        FileNotFoundError: If any specified directory does not exist.
        ValueError: If no full or photometry files are found, duplicate
            parameter keys are encountered, or no matching pairs exist.
    """
    if isinstance(mist_root_or_folders, (str, os.PathLike)):
        mist_root_or_folders = [mist_root_or_folders]
    all_full_files = []
    all_phot_files = []
    for folder in mist_root_or_folders:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f'Folder does not exist: {folder}')
        all_full_files += list(folder.rglob('*_full.iso'))
        all_phot_files += list(folder.rglob('*_UBVRIplus.iso.cmd'))
    if len(all_full_files) == 0:
        raise ValueError('No *_full.iso files found.')
    if len(all_phot_files) == 0:
        raise ValueError('No *_UBVRIplus.iso.cmd files found.')
    full_dict = {}
    phot_dict = {}
    for file_path in all_full_files:
        info = parse_mist_filename(file_path)
        if info['kind'] != 'full':
            continue
        key = info['key']
        if key in full_dict:
            raise ValueError(f'Duplicated full file key: {key}')
        full_dict[key] = info['path']
    for file_path in all_phot_files:
        info = parse_mist_filename(file_path)
        if info['kind'] != 'phot':
            continue
        key = info['key']
        if key in phot_dict:
            raise ValueError(f'Duplicated phot file key: {key}')
        phot_dict[key] = info['path']
    common_keys = sorted(set(full_dict.keys()) & set(phot_dict.keys()))
    missing_phot = sorted(set(full_dict.keys()) - set(phot_dict.keys()))
    missing_full = sorted(set(phot_dict.keys()) - set(full_dict.keys()))
    if verbose:
        print(f'Found full files: {len(full_dict)}')
        print(f'Found phot files: {len(phot_dict)}')
        print(f'Matched pairs: {len(common_keys)}')
        print(f'Missing phot files: {len(missing_phot)}')
        print(f'Missing full files: {len(missing_full)}')
    if len(common_keys) == 0:
        raise ValueError('No matched MIST full/phot file pairs found.')
    pairs = []
    for key in common_keys:
        pairs.append({'key': key, 'full_file': full_dict[key], 'phot_file': phot_dict[key]})
    return pairs

def standardize_mist_dataframe(df_mist):
    """
    Convert a merged MIST DataFrame to the standard grid format.

    The output contains the standardized stellar parameters:
    ``Mini``, ``logAge``, ``Age``, ``MH``, ``omega_ini``, ``Mass``, ``omega``,
    ``logL``, ``logTe``, and ``Req``.

    The age is converted from years to Myr, and the stellar radius is
    converted from log10(R/R_sun) to R/R_sun. Non-numeric and non-finite
    values are removed.

    Args:
        df_mist (pd.DataFrame): Merged MIST DataFrame containing the original
            stellar parameters.

    Returns:
        pd.DataFrame: Standardized DataFrame containing the selected stellar
            parameters, with invalid rows removed.

    Raises:
        ValueError: If any required columns are missing from ``df_mist``.
    """
    col_map = {'initial_mass': 'Mini', 'log10_isochrone_age_yr': 'logAge', 'star_mass': 'Mass',
               'surf_avg_omega_div_omega_crit': 'omega', '[Fe/H]_init': 'MH', '[Fe/H]': 'MH_current',
               'log_L': 'logL', 'log_Teff': 'logTe'}
    required_cols = list(col_map.keys()) + ['log_R', 'omega_ini']
    missing_cols = [c for c in required_cols if c not in df_mist.columns]
    if len(missing_cols) > 0:
        raise ValueError(f'Missing required columns in merged MIST DataFrame: {missing_cols}')
    df_new = df_mist[required_cols].copy()
    df_new = df_new.rename(columns=col_map)
    idx_logAge = df_new.columns.get_loc('logAge')
    age = np.round(10.0 ** (df_new['logAge'].to_numpy(dtype=np.float64) - 6.0), 8)
    df_new.insert(idx_logAge + 1, 'Age', age)
    df_new['Req'] = 10.0 ** df_new['log_R']
    df_new = df_new.drop(columns=['log_R'])
    keep_cols = ['Mini', 'logAge', 'Age', 'MH', 'omega_ini', 'Mass', 'omega', 'logL', 'logTe', 'Req']
    df_new = df_new[keep_cols].copy()
    for col in keep_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
    df_new = df_new.replace([np.inf, -np.inf], np.nan).dropna(subset=keep_cols)
    df_new = df_new.reset_index(drop=True)
    return df_new

def _read_one_mist_pair_for_regularization(args):
    """
    Worker function for reading one MIST full/phot pair.
    """
    pair_i, full_file, phot_file, check_order = args
    df_isos = read_MIST(full_file, add_initial_params=True)
    df_phot = read_MIST(phot_file)
    df_mist = MIST_merge_isos_phot(df_isos, df_phot, check_order=check_order)
    df_new = standardize_mist_dataframe(df_mist)
    df_new['pair_index'] = pair_i
    df_new['full_file'] = str(full_file)
    df_new['phot_file'] = str(phot_file)
    return df_new

def _process_one_MIST_Mini_group(args):
    """
    Regularize one group along Mini.

    One group is fixed:
        Age, MH, omega_ini

    The function interpolates:
        Mass, omega, logL, Req

    onto the global Mini_grid.
    """
    loop_i, group_key, df_i, group_cols, compare_cols, interp_output_list, Mini_grid, dtype, Age_to_idx, MH_to_idx, omega_ini_to_idx, pair_i, source_info = args
    group_info = {col: group_key[k] for k, col in enumerate(group_cols)}
    result = {'status': 'skipped', 'problematic_records': [], 'safe_duplicate_count': 0, 'filled_group_count': 0, 'skipped_group_count': 0, 'grid_index': None, 'y_grid': None}
    df_i = df_i.sort_values('Mini').reset_index(drop=True)
    if len(df_i) < 2:
        result['skipped_group_count'] += 1
        return result
    interp_x = df_i['Mini'].values
    interp_y = df_i[interp_output_list].values
    dup_mask_all = pd.Series(interp_x).duplicated(keep=False).to_numpy()
    if dup_mask_all.any():
        local_Mini_min = np.nanmin(interp_x)
        local_Mini_max = np.nanmax(interp_x)
        local_range = local_Mini_max - local_Mini_min
        dup_groups = df_i.loc[dup_mask_all].groupby('Mini', sort=False)
        for mini_value, g in dup_groups:
            n_unique_rows = g[compare_cols].drop_duplicates().shape[0]
            if n_unique_rows == 1:
                result['safe_duplicate_count'] += len(g) - 1
            else:
                bad_pct = 100.0 * (mini_value - local_Mini_min) / local_range if local_range > 0 else np.nan
                record = {'pair_index': pair_i, 'source_info': source_info, 'loop_index_in_this_pair': loop_i, 'Mini': mini_value, 'position_pct': bad_pct,
                          'n_duplicates': len(g), 'n_unique_rows_in_compare_cols': n_unique_rows, 'local_Mini_min': local_Mini_min, 'local_Mini_max': local_Mini_max}
                for col in group_cols:
                    record[col] = group_info[col]
                result['problematic_records'].append(record)
        keep_mask = ~pd.Series(interp_x).duplicated(keep='first').to_numpy()
        df_i = df_i.loc[keep_mask].reset_index(drop=True)
        interp_x = df_i['Mini'].values
        interp_y = df_i[interp_output_list].values
    if len(interp_x) < 2:
        result['skipped_group_count'] += 1
        return result
    if not np.all(np.diff(interp_x) > 0):
        record = {'pair_index': pair_i, 'source_info': source_info, 'loop_index_in_this_pair': loop_i, 'reason': 'Mini is not strictly increasing after duplicate handling'}
        for col in group_cols:
            record[col] = group_info[col]
        result['problematic_records'].append(record)
        result['skipped_group_count'] += 1
        return result
    local_Mini_min = np.nanmin(interp_x)
    local_Mini_max = np.nanmax(interp_x)
    y_grid = np.full((len(Mini_grid), len(interp_output_list)), np.nan, dtype=dtype)
    valid_mask = (Mini_grid >= local_Mini_min) & (Mini_grid <= local_Mini_max)
    if np.sum(valid_mask) == 0:
        result['skipped_group_count'] += 1
        return result
    for j, col in enumerate(interp_output_list):
        f = interp1d(interp_x, interp_y[:, j], kind='linear', bounds_error=False, fill_value=np.nan, assume_sorted=True)
        y_grid[valid_mask, j] = f(Mini_grid[valid_mask]).astype(dtype)
    idx_Age = Age_to_idx[group_info['Age']]
    idx_MH = MH_to_idx[group_info['MH']]
    idx_omega_ini = omega_ini_to_idx[group_info['omega_ini']]
    grid_index = (idx_Age, idx_MH, idx_omega_ini)
    result['status'] = 'filled'
    result['filled_group_count'] = 1
    result['grid_index'] = grid_index
    result['y_grid'] = y_grid
    return result

def regularization_on_Mini_from_MIST_folders(
        mist_root_or_folders, save_path='.', save_name='regularized_MIST_Mini_grid.npz',
        Mini_setting=(1.0, 5.0, 0.01), save_float32=True, read_all_at_once=False, parallel_file_read=True,
        parallel_backend='process', max_workers=None, check_order=True, local_max_workers=8, print_pairs=False):
    """
    Regularize MIST isochrones onto a uniform initial-mass grid.

    The function reads matched MIST isochrone and photometry files, collects
    the available age, metallicity, and rotation grids, and interpolates the
    stellar parameters onto a regular ``Mini`` grid. The resulting arrays are
    saved in a compressed NPZ file. Records with problematic duplicated
    ``Mini`` values are also saved to a CSV file.

    Args:
        mist_root_or_folders (str, os.PathLike, or list): Root directory,
            individual directory, or list of directories containing MIST files.
        save_path (str or os.PathLike): Output directory. Created if it does
            not exist.
        save_name (str): Name of the output NPZ file.
        Mini_setting (tuple or None): Initial-mass grid specification
            ``(Mini_min, Mini_max, Mini_step)``. Defaults to ``(1.0, 5.0, 0.01)``.
        save_float32 (bool): Whether to save the interpolated grid as float32.
        read_all_at_once (bool): If True, load all matched file pairs into one
            DataFrame before processing; otherwise, process one pair at a time.
        parallel_file_read (bool): Whether to read file pairs in parallel when
            ``read_all_at_once=True``.
        parallel_backend (str): Parallel backend for file reading: ``"process"``
            or ``"thread"``.
        max_workers (int or None): Number of workers for parallel file reading.
            Defaults to the available CPU count or the number of file pairs,
            whichever is smaller.
        check_order (bool): Whether to check row ordering when merging MIST
            isochrone and photometry files.
        local_max_workers (int): Number of threads used to process groups
            within each file or combined DataFrame.
        print_pairs (bool): Whether to print the matched file pairs.

    Returns:
        tuple: A tuple containing:
            - str: Absolute path to the saved NPZ file.
            - pd.DataFrame: Records of problematic duplicated ``Mini`` values.

    Raises:
        ValueError: If ``Mini_setting`` is invalid, the parallel backend is
            unsupported, or required columns are missing.
        FileNotFoundError: If the specified input directories do not exist.
    """
    if not read_all_at_once and parallel_file_read:
        print('[info] parallel_read is ignored because read_all_at_once=False.')
    if isinstance(save_path, (str, os.PathLike)):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    dtype = np.float32 if save_float32 else np.float64
    interp_variable_list = ['Mini', 'Age', 'MH', 'omega_ini']
    group_cols = ['Age', 'MH', 'omega_ini']
    interp_output_list = ['Mass', 'omega', 'logL', 'logTe', 'Req']
    in_cols = interp_variable_list
    compare_cols = in_cols + interp_output_list
    pairs = find_mist_file_pairs(mist_root_or_folders, verbose=True)
    if print_pairs:
        for i, pair in enumerate(pairs):
            print(f"[{i}] key={pair['key']}")
            print(f"    full: {pair['full_file']}")
            print(f"    phot: {pair['phot_file']}")
    if Mini_setting is None:
        Mini_setting = (1.0, 5.0, 0.01)
    if not isinstance(Mini_setting, (list, tuple, np.ndarray)) or len(Mini_setting) != 3:
        raise ValueError('Mini_setting must be (Mini_min, Mini_max, Mini_step).')
    Mini_min, Mini_max, Mini_step = Mini_setting
    if Mini_step <= 0:
        raise ValueError('Mini_step must be positive.')
    n_Mini = int(np.floor((Mini_max - Mini_min) / Mini_step + 0.5)) + 1
    Mini_grid = Mini_min + np.arange(n_Mini) * Mini_step
    Mini_grid = np.round(Mini_grid, 10)
    Mini_grid = Mini_grid[Mini_grid <= Mini_max + 1e-10]
    df_all = None
    if read_all_at_once:
        print('Reading all MIST pairs at once...')
        read_args = [(pair_i, pairs[pair_i]['full_file'], pairs[pair_i]['phot_file'], check_order) for pair_i in range(len(pairs))]
        if parallel_file_read:
            if max_workers is None:
                max_workers = min(os.cpu_count() or 1, len(read_args))
            if parallel_backend == 'process':
                Executor = ProcessPoolExecutor
            elif parallel_backend == 'thread':
                Executor = ThreadPoolExecutor
            else:
                raise ValueError("parallel_backend must be 'process' or 'thread'.")
            with Executor(max_workers=max_workers) as executor:
                df_list = list(tqdm(executor.map(_read_one_mist_pair_for_regularization, read_args), total=len(read_args)))
        else:
            df_list = []
            for args in tqdm(read_args):
                df_list.append(_read_one_mist_pair_for_regularization(args))
        df_all = pd.concat(df_list, ignore_index=True)
        del df_list
        print(f'df_all shape: {df_all.shape}')
        print(f'df_all memory: {df_all.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB ({df_all.memory_usage(deep=True).sum() / 1024 ** 3:.2f} GB)')
    print('Collecting axis values...')
    Age_list = []
    MH_list = []
    omega_ini_list = []
    if read_all_at_once:
        df = df_all
        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f'Missing columns in df_all: {missing_cols}')
        Age_list.append(df['Age'].dropna().unique())
        MH_list.append(df['MH'].dropna().unique())
        omega_ini_list.append(df['omega_ini'].dropna().unique())
    else:
        for pair_i, pair in enumerate(tqdm(pairs)):
            df = _read_one_mist_pair_for_regularization((pair_i, pair['full_file'], pair['phot_file'], check_order))
            missing_cols = [c for c in compare_cols if c not in df.columns]
            if len(missing_cols) > 0:
                raise ValueError(f'Missing columns in pair {pair_i}: {missing_cols}')
            Age_list.append(df['Age'].dropna().unique())
            MH_list.append(df['MH'].dropna().unique())
            omega_ini_list.append(df['omega_ini'].dropna().unique())
            del df
    Age_grid = np.sort(np.unique(np.concatenate(Age_list)))
    MH_grid = np.sort(np.unique(np.concatenate(MH_list)))
    omega_ini_grid = np.sort(np.unique(np.concatenate(omega_ini_list)))
    Age_to_idx = {v: i for i, v in enumerate(Age_grid)}
    MH_to_idx = {v: i for i, v in enumerate(MH_grid)}
    omega_ini_to_idx = {v: i for i, v in enumerate(omega_ini_grid)}
    grid_outputs = np.full((len(Mini_grid), len(Age_grid), len(MH_grid), len(omega_ini_grid), len(interp_output_list)), np.nan, dtype=dtype)
    print(f'Mini_grid: [min: {np.min(Mini_grid)}, max: {np.max(Mini_grid)}]')
    print(f'Mini_grid shape: {Mini_grid.shape}')
    print(f'Age_grid: [min: {np.min(Age_grid)}, max: {np.max(Age_grid)}] Myr')
    print(f'Age_grid shape: {Age_grid.shape}')
    print(f'MH_grid shape: {MH_grid.shape}')
    print(f'omega_ini_grid shape: {omega_ini_grid.shape}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'grid_outputs memory: {grid_outputs.nbytes / 1024 ** 2:.2f} MB ({grid_outputs.nbytes / 1024 ** 3:.2f} GB)')
    print('Filling regular grids...')
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
            source_info = 'df_all'
        else:
            df = _read_one_mist_pair_for_regularization((pair_i, pair['full_file'], pair['phot_file'], check_order))
            source_info = f"full={pair['full_file']}; phot={pair['phot_file']}"
        missing_cols = [c for c in compare_cols if c not in df.columns]
        if len(missing_cols) > 0:
            raise ValueError(f'Missing columns during filling: {missing_cols}')
        grouped = list(df.groupby(group_cols, sort=False))
        task_args = []
        for loop_i, (group_key, df_i) in enumerate(grouped):
            if len(group_cols) == 1:
                group_key = (group_key,)
            task_args.append((loop_i, group_key, df_i, group_cols, compare_cols, interp_output_list, Mini_grid, dtype, Age_to_idx, MH_to_idx, omega_ini_to_idx, pair_i, source_info))
        with ThreadPoolExecutor(max_workers=local_max_workers) as executor:
            iterator = executor.map(_process_one_MIST_Mini_group, task_args)
            if read_all_at_once:
                iterator = tqdm(iterator, total=len(task_args))
            results = list(iterator)
        for res in results:
            problematic_records.extend(res['problematic_records'])
            safe_duplicate_count += res['safe_duplicate_count']
            skipped_group_count += res['skipped_group_count']
            if res['status'] != 'filled':
                continue
            y_grid = res['y_grid']
            idx_Age, idx_MH, idx_omega_ini = res['grid_index']
            grid_outputs[:, idx_Age, idx_MH, idx_omega_ini, :] = y_grid
            filled_group_count += res['filled_group_count']
        if not read_all_at_once:
            del df
    if read_all_at_once:
        del df_all
    problematic_df = pd.DataFrame(problematic_records)
    save_file = save_path / save_name
    problematic_path = save_path / 'problematic_duplicated_Mini_records_MIST.csv'
    save_dict = dict(
        Mini_grid=Mini_grid, Age_grid=Age_grid, MH_grid=MH_grid, omega_ini_grid=omega_ini_grid, grid_outputs=grid_outputs,
        in_cols=np.array(in_cols, dtype=object), interp_variable_list=np.array(interp_variable_list, dtype=object),
        interp_output_list=np.array(interp_output_list, dtype=object), group_cols=np.array(group_cols, dtype=object),
        grid_type=np.array('MIST'), age_unit=np.array('Myr'), save_float32=np.array(save_float32), read_all_at_once=np.array(read_all_at_once),
        grid_dtype=np.array(str(grid_outputs.dtype)), safe_duplicate_count=np.array(safe_duplicate_count), filled_group_count=np.array(filled_group_count),
        skipped_group_count=np.array(skipped_group_count))
    np.savez_compressed(save_file, **save_dict)
    problematic_df.to_csv(problematic_path, index=False)
    total_bytes = grid_outputs.nbytes + Mini_grid.nbytes + Age_grid.nbytes + MH_grid.nbytes + omega_ini_grid.nbytes
    print('Finished MIST regularization.')
    print(f'save_float32: {save_float32}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    print(f'filled_group_count: {filled_group_count}')
    print(f'skipped_group_count: {skipped_group_count}')
    print(f'safe_duplicate_count: {safe_duplicate_count}')
    print(f'problematic_duplicate_count: {len(problematic_df)}')
    print(f'problematic duplicated Mini records are saved at: {os.path.abspath(problematic_path)}')
    print(f'Grids are saved at: {os.path.abspath(save_file)}')
    print(f'total grid memory: {total_bytes / 1024 ** 2:.2f} MB ({total_bytes / 1024 ** 3:.2f} GB)')
    return (os.path.abspath(save_file), problematic_df)

def load_regularized_MIST_interpolator(npz_path, force_float64=False):
    """
    Load a regularized MIST Mini grid and create an interpolator.

    The interpolator uses ``Mini``, ``Age``, ``MH``, and ``omega_ini`` as input
    coordinates, with age in Myr. The output variables are defined by
    ``interp_output_list`` stored in the NPZ file.

    Args:
        npz_path (str or os.PathLike): Path to the saved regularized MIST grid.
        force_float64 (bool): If True, convert ``grid_outputs`` to float64
            before constructing the interpolator.

    Returns:
        tuple: A tuple containing:
            - RegularGridInterpolator: Interpolator over the regularized MIST grid.
            - list of str: Names of the interpolation input variables.
            - list of str: Names of the interpolation output variables.
            - NpzFile: Loaded NPZ data object.

    Raises:
        ValueError: If the file does not contain a MIST grid or its interpolation
            input variables do not match the expected order.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_type = str(data['grid_type'])
    if grid_type != 'MIST':
        raise ValueError(f'This does not look like a MIST grid. grid_type={grid_type}')
    Mini_grid = data['Mini_grid']
    Age_grid = data['Age_grid']
    MH_grid = data['MH_grid']
    omega_ini_grid = data['omega_ini_grid']
    grid_outputs = data['grid_outputs']
    if force_float64:
        grid_outputs = grid_outputs.astype(np.float64)
    interp_variable_list = data['interp_variable_list'].astype(str).tolist()
    if interp_variable_list != ['Mini', 'Age', 'MH', 'omega_ini']:
        raise ValueError(f'Unexpected MIST interpolation variables: {interp_variable_list}')
    interp_output_list = data['interp_output_list'].astype(str).tolist()
    points = (Mini_grid, Age_grid, MH_grid, omega_ini_grid)
    interp_func = RegularGridInterpolator(points, grid_outputs, bounds_error=False, fill_value=np.nan)
    total_grid_bytes = grid_outputs.nbytes + Mini_grid.nbytes + Age_grid.nbytes + MH_grid.nbytes + omega_ini_grid.nbytes
    print(f"Age_grid: [{np.min(Age_grid)}, {np.max(Age_grid)}] {str(data['age_unit'])}")
    print(f'Age_grid shape: {Age_grid.shape}')
    print('Loaded regularized MIST Mini grid.')
    print(f'grid_type: {grid_type}')
    print(f'interp_variable_list: {interp_variable_list}')
    print(f'interp_output_list: {interp_output_list}')
    print(f'grid_outputs shape: {grid_outputs.shape}')
    print(f'grid_outputs dtype: {grid_outputs.dtype}')
    if 'save_float32' in data.files:
        print(f"save_float32: {bool(data['save_float32'])}")
    if 'read_all_at_once' in data.files:
        print(f"read_all_at_once: {bool(data['read_all_at_once'])}")
    print(f'total grid memory: {total_grid_bytes / 1024 ** 2:.2f} MB ({total_grid_bytes / 1024 ** 3:.2f} GB)')
    return (interp_func, interp_variable_list, interp_output_list, data)

def regularize_MIST_Mini_grid(
        mist_root_or_folders=None, save_path=None, save_name=None, Mini_setting=(1.0, 5.0, 0.001),
        save_float32=True, read_all_at_once=False, parallel_file_read=True, parallel_backend='process',
        max_workers=None, check_order=True, local_max_workers=8, print_pairs=False):
    """
    Regularize MIST isochrones onto a uniform initial-mass grid.

    This is a wrapper for ``regularization_on_Mini_from_MIST_folders()``.
    MIST does not provide inclination grids; therefore, only the PARS method
    is available.

    Default input directory:
        ``DATA_DIR / 'isochrones' / 'MIST'``

    Default output file:
        ``DATA_DIR / 'isochrones' / 'regular_MIST_Mini_grids' /
        'regularized_MIST_Mini_grid.npz'``

    Args:
        mist_root_or_folders (str, os.PathLike, or None): Input directory or
            directories containing MIST files. Defaults to the MIST data directory.
        save_path (str, os.PathLike, or None): Output directory. Defaults to
            the regularized MIST grid directory.
        save_name (str or None): Output NPZ filename.
        Mini_setting (tuple): Initial-mass grid specification
            ``(Mini_min, Mini_max, Mini_step)``. Defaults to ``(1.0, 5.0, 0.001)``.
        save_float32 (bool): Whether to save the interpolated grid as float32.
        read_all_at_once (bool): If True, load all matched file pairs into one
            DataFrame before processing; otherwise, process one pair at a time.
        parallel_file_read (bool): Whether to read file pairs in parallel when
            ``read_all_at_once=True``.
        parallel_backend (str): Parallel backend for file reading: ``"process"``
            or ``"thread"``.
        max_workers (int or None): Number of workers for parallel file reading.
        check_order (bool): Whether to check row ordering when merging MIST
            isochrone and photometry files.
        local_max_workers (int): Number of threads used to process groups
            within each file or combined DataFrame.
        print_pairs (bool): Whether to print the matched file pairs.

    Returns:
        tuple: A tuple containing:
            - str: Absolute path to the saved NPZ file.
            - pd.DataFrame: Records of problematic duplicated ``Mini`` values.
    """
    if mist_root_or_folders is None:
        mist_root_or_folders = DATA_DIR / 'isochrones' / 'MIST'
    if save_path is None:
        save_path = DATA_DIR / 'isochrones' / 'regular_MIST_Mini_grids'
    if save_name is None:
        save_name = 'regularized_MIST_Mini_grid.npz'
    mist_root_or_folders = Path(mist_root_or_folders).expanduser().resolve()
    save_path = Path(save_path).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    save_file, problematic_df = regularization_on_Mini_from_MIST_folders(
        mist_root_or_folders=mist_root_or_folders, save_path=save_path, save_name=save_name, Mini_setting=Mini_setting,
        save_float32=save_float32, read_all_at_once=read_all_at_once, parallel_file_read=parallel_file_read, parallel_backend=parallel_backend,
        max_workers=max_workers, check_order=check_order, local_max_workers=local_max_workers, print_pairs=print_pairs)
    return (save_file, problematic_df)

def load_default_regularized_MIST_Mini_interpolator(npz_path=None, save_path=None, save_name=None, force_float64=False):
    """
    Load the default regularized MIST Mini-grid interpolator.

    If ``npz_path`` is not provided, the default grid file is used:
    ``DATA_DIR / 'isochrones' / 'regular_MIST_Mini_grids' /
    'regularized_MIST_Mini_grid.npz'``.

    Args:
        npz_path (str, os.PathLike, or None): Path to an existing regularized
            MIST grid file. If None, construct the path from ``save_path``
            and ``save_name``.
        save_path (str, os.PathLike, or None): Directory containing the grid
            file. Defaults to the standard regularized MIST grid directory.
            Used only when ``npz_path`` is None.
        save_name (str or None): Grid filename. Defaults to
            ``'regularized_MIST_Mini_grid.npz'``. Used only when
            ``npz_path`` is None.
        force_float64 (bool): If True, convert ``grid_outputs`` to float64
            before constructing the interpolator.

    Returns:
        tuple: The return values of ``load_regularized_MIST_interpolator()``:
            - RegularGridInterpolator: Interpolator over the MIST grid.
            - list of str: Names of the interpolation input variables.
            - list of str: Names of the interpolation output variables.
            - NpzFile: Loaded NPZ data object.

    Raises:
        FileNotFoundError: If the specified regularized MIST grid file does not exist.
    """
    if npz_path is None:
        if save_path is None:
            save_path = DATA_DIR / 'isochrones' / 'regular_MIST_Mini_grids'
        if save_name is None:
            save_name = 'regularized_MIST_Mini_grid.npz'
        npz_path = Path(save_path).expanduser().resolve() / save_name
    else:
        npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f'Regularized MIST Mini grid not found: {npz_path}')
    return load_regularized_MIST_interpolator(npz_path=npz_path, force_float64=force_float64)





def read_parsec_track(file_path):
    """
    Read a PARSEC v2.0 evolutionary track file.

    The function locates the header containing the ``MASS`` and ``AGE``
    column names, then reads the subsequent non-comment data lines into
    a DataFrame.

    Args:
        file_path (str or os.PathLike): Path to the evolutionary track file.

    Returns:
        pd.DataFrame: Evolutionary track data with column names taken from
            the file header. Returns an empty DataFrame with the same columns
            if no data rows are present.

    Raises:
        ValueError: If no header containing both ``MASS`` and ``AGE`` is found.
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the header line containing MASS and AGE
    header_idx = None
    columns = None

    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        names = stripped.split()

        if "MASS" in names and "AGE" in names:
            header_idx = i
            columns = names
            break

    if header_idx is None:
        raise ValueError(f"Cannot find MASS and AGE header in {file_path}")

    # Remove comments and blank lines from the data
    data_lines = [
        line for line in lines[header_idx + 1:]
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not data_lines:
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(
        StringIO("".join(data_lines)),
        sep=r"\s+",
        names=columns,
        header=None,
    )

    return df

