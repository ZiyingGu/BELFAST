import re
import os
import time
import warnings
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from math import erf, sqrt
from urllib.parse import quote
import matplotlib.pyplot as plt
from urllib.request import urlopen
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from astroquery.vizier import Vizier
from scipy.stats import gaussian_kde
from scipy.stats import norm as stats_norm
from concurrent.futures import ThreadPoolExecutor, as_completed

from belfast.config import id_columns
from belfast.target_utils import query_data, cal_abs_mag, remove_extinction




# Following codes:
# preparing data of cluster members
def safe_catalog_name(catalog_id):
    """
    Convert a catalog identifier into a filesystem-friendly name.

    Args:
        catalog_id: Catalog identifier.

    Returns:
        A sanitized catalog name.
    """
    return str(catalog_id).strip().replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def value_to_clean_str(x):
    """
    Convert an Astropy or NumPy scalar into a clean string.

    Args:
        x: Value to convert, typically a source identifier.

    Returns:
        A cleaned string, or an empty string for missing values.
    """
    if x is None or np.ma.is_masked(x):
        return ""

    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore").strip()

    try:
        if hasattr(x, "item"):
            x = x.item()
    except (ValueError, TypeError):
        pass

    if isinstance(x, str):
        value = x.strip()
        return "" if value.lower() in {"", "nan", "none", "--"} else value

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        return f"{x:.0f}" if float(x).is_integer() else str(x)

    value = str(x).strip()
    return "" if value.lower() in {"", "nan", "none", "--"} else value


def find_source_column(tab, preferred_source_cols=None):
    """
    Find the source identifier column in a VizieR table.

    Args:
        tab: Astropy table returned by VizieR.
        preferred_source_cols: Optional ordered list of preferred column names.

    Returns:
        The matching column name, or None if no suitable column is found.
    """
    if preferred_source_cols is None:
        preferred_source_cols = [
            "SourceCP", "Source", "source", "source_id", "SourceID",
            "GaiaDR2", "GaiaDR3", "GaiaDR2Source", "GaiaDR3Source",
            "Gaia", "ID", "id",
        ]

    colnames = list(tab.colnames)
    lower_to_real = {name.lower(): name for name in colnames}

    for name in preferred_source_cols:
        if name in colnames:
            return name
        if name.lower() in lower_to_real:
            return lower_to_real[name.lower()]

    normalize = lambda name: name.lower().replace("-", "").replace("_", "").replace(" ", "")
    normalized_cols = {normalize(name): name for name in colnames}
    keywords = ["sourcecp", "source", "sourceid", "gaiadr2", "gaiadr3", "gaia", "id"]

    for keyword in keywords:
        for normalized_name, original_name in normalized_cols.items():
            if keyword in normalized_name:
                return original_name

    return None


def download_vizier_catalog_with_target_name(
    catalog_id,
    output_dir=None,
    row_limit=-1,
    target_prefix="Gaia DR2 ",
    target_col_name="target_name",
    preferred_source_cols=None,
    save_readme=True,
    verbose=True,
):
    """
    Download a VizieR catalog and save its tables as CSV files.

    Each table receives a target-name column derived from its source identifiers.

    Args:
        catalog_id: VizieR catalog identifier.
        output_dir: Output directory. Defaults to a catalog-specific directory.
        row_limit: Maximum number of rows per table; -1 retrieves all available rows.
        target_prefix: Prefix added to each non-empty source identifier.
        target_col_name: Name of the generated target-name column.
        preferred_source_cols: Ordered list of preferred source identifier columns.
        save_readme: Whether to download the catalog ReadMe file.
        verbose: Whether to print progress and warning messages.

    Returns:
        A dictionary containing the catalog metadata, downloaded tables, and output paths.
    """
    catalog_id = str(catalog_id).strip()
    safe_name = safe_catalog_name(catalog_id)
    output_dir = Path(output_dir) if output_dir is not None else Path(f"./vizier_{safe_name}")
    output_dir = Path(str(output_dir).replace(" ", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    readme_path = output_dir / f"{safe_name}_readme.txt"

    if save_readme:
        readme_url = f"https://cdsarc.cds.unistra.fr/ftp/{quote(catalog_id, safe='/+')}/ReadMe"
        try:
            with urlopen(readme_url, timeout=60) as response:
                readme_text = response.read().decode("utf-8", errors="replace")
            readme_path.write_text(readme_text, encoding="utf-8")
            if verbose:
                print(f"[saved ReadMe] {readme_path}")
        except Exception as exc:
            if verbose:
                print(f"[warning] Failed to download ReadMe: {readme_url}")
                print(f"          {type(exc).__name__}: {exc}")

    viz = Vizier(columns=["**"], row_limit=row_limit)
    tables = viz.get_catalogs(catalog_id)
    table_summary_rows = []
    saved_table_paths = []

    if verbose:
        print(f"[catalog] {catalog_id}")
        print(f"[tables] {list(tables.keys())}")

    for table_index, tab in enumerate(tables, start=1):
        table_name = tab.meta.get("name", f"table{table_index}")
        source_col = find_source_column(tab, preferred_source_cols)
        df = tab.to_pandas()

        if target_col_name in df.columns:
            df = df.drop(columns=[target_col_name])

        if source_col is None:
            target_names = [""] * len(df)
            if verbose:
                print(f"[warning] Table {table_index}: no source identifier column found.")
        else:
            target_names = [
                f"{target_prefix}{value}" if (value := value_to_clean_str(v)) else ""
                for v in tab[source_col]
            ]

        df.insert(0, target_col_name, target_names)
        table_csv_path = output_dir / f"{safe_name}_table{table_index}.csv"
        df.to_csv(table_csv_path, index=False)
        saved_table_paths.append(table_csv_path)

        table_summary_rows.append({
            "table_index": table_index,
            "vizier_table_name": table_name,
            "saved_csv": str(table_csv_path),
            "source_column_used": source_col or "",
            "target_prefix": target_prefix,
            "n_rows_downloaded": len(df),
            "n_columns_saved": len(df.columns),
            "columns": ", ".join(df.columns),
            "description": tab.meta.get("description", ""),
        })

        if verbose:
            print(f"[saved table{table_index}] {table_name}")
            print(f"              source column: {source_col}")
            print(f"              csv: {table_csv_path}")
            print(f"              rows={len(df)}, cols={len(df.columns)}")

    table_summary_path = output_dir / f"{safe_name}_table_summary.csv"
    df_table_summary = pd.DataFrame(table_summary_rows)
    df_table_summary.to_csv(table_summary_path, index=False)

    if verbose:
        print(f"[saved summary] {table_summary_path}")

    return {
        "catalog_id": catalog_id,
        "safe_name": safe_name,
        "output_dir": output_dir,
        "readme_path": readme_path if save_readme else None,
        "table_paths": saved_table_paths,
        "table_summary_path": table_summary_path,
        "table_summary": df_table_summary,
        "tables": tables,
    }


def _clean_target_name(x):
    """
    Normalize a target name and convert missing values to an empty string.

    Args:
        x: Input target name.

    Returns:
        A cleaned target name.
    """
    if x is None or pd.isna(x):
        return ""

    value = str(x).strip()
    return "" if value.lower() in {"", "nan", "none", "--", "<na>"} else value


def _query_one_target_safe(target_name, index=None, max_retries=3, query_verbose=False):
    """
    Query one target and convert failures into structured results.

    Args:
        target_name: Target identifier passed to query_data().
        index: Original position of the target in the input sequence.
        max_retries: Maximum number of retries for each query operation.
        query_verbose: Whether to enable verbose output in query_data().

    Returns:
        A dictionary containing the query result, input metadata, and failure status.
    """
    target_name = _clean_target_name(target_name)
    out_base = {
        "_input_index": index,
        "query_input_name": target_name,
        "query_failed": True,
        "query_error": "",
    }

    if not target_name:
        out_base["query_error"] = "Empty target name"
        return out_base

    try:
        df_query = query_data(
            target_name,
            max_retries=max_retries,
            return_df=True,
            verbose=query_verbose,
        )

        if df_query is None or df_query.empty:
            raise RuntimeError("query_data returned an empty DataFrame.")

        result = df_query.iloc[0].to_dict()
        result.update({
            "_input_index": index,
            "query_input_name": target_name,
            "query_failed": False,
            "query_error": "",
        })
        return result

    except Exception as exc:
        out_base["query_error"] = f"{type(exc).__name__}: {exc}"
        return out_base


def query_data_many(
    target_names,
    max_retries=3,
    n_workers=4,
    query_verbose=False,
    verbose=True,
):
    """
    Query multiple targets concurrently while preserving input order.

    Args:
        target_names: Sequence of target identifiers.
        max_retries: Maximum number of retries for each query operation.
        n_workers: Number of worker threads.
        query_verbose: Whether to enable verbose output in query_data().
        verbose: Whether to print progress and summary information.

    Returns:
        A DataFrame containing one row per input target.
    """
    target_names = list(target_names)
    n_targets = len(target_names)

    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    if n_targets == 0:
        return pd.DataFrame(columns=["query_input_name", "query_failed", "query_error"])

    if verbose:
        print(f"[n targets] {n_targets}")
        print(f"[n workers] {n_workers}")

    results = [None] * n_targets
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(_query_one_target_safe, name, i, max_retries, query_verbose): i
            for i, name in enumerate(target_names)
        }

        for n_done, future in enumerate(as_completed(future_to_index), start=1):
            index = future_to_index[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "_input_index": index,
                    "query_input_name": _clean_target_name(target_names[index]),
                    "query_failed": True,
                    "query_error": f"{type(exc).__name__}: {exc}",
                }

            results[index] = result

            if verbose:
                status = "FAILED" if result.get("query_failed", True) else "OK"
                print(f"[{n_done}/{n_targets}] {status}: {result['query_input_name']}")

    df_query = pd.DataFrame(results).sort_values("_input_index").reset_index(drop=True)
    df_query = df_query.drop(columns="_input_index")

    if verbose:
        elapsed = time.time() - start_time
        n_failed = int(df_query["query_failed"].sum())
        print("\nDone.")
        print(f"[failed] {n_failed}/{n_targets}")
        print(f"[elapsed] {elapsed:.1f} sec")

    return df_query


def append_query_data_to_table(
    input_table,
    output_table,
    target_col=None,
    max_retries=3,
    n_workers=4,
    query_verbose=False,
    verbose=True,
):
    """
    Append query_data results to an existing CSV table.

    Original columns are preserved. Query columns already present in the input
    table are not appended a second time.

    Args:
        input_table: Path to the input CSV file.
        output_table: Path to the output CSV file.
        target_col: Target-name column. Defaults to the first input column.
        max_retries: Maximum number of retries for each query operation.
        n_workers: Number of worker threads.
        query_verbose: Whether to enable verbose output in query_data().
        verbose: Whether to print progress and summary information.

    Returns:
        The merged DataFrame.
    """
    input_table = Path(input_table)
    output_table = Path(output_table)

    df_input = pd.read_csv(input_table, dtype={col: "string" for col in id_columns})

    if df_input.empty and len(df_input.columns) == 0:
        raise ValueError("Input table has no columns.")

    if len(df_input.columns) == 0:
        raise ValueError("Input table has no columns.")

    target_col = df_input.columns[0] if target_col is None else target_col
    if target_col not in df_input.columns:
        raise ValueError(f"Target column '{target_col}' not found in input table.")

    if verbose:
        print(f"[input] {input_table}")
        print(f"[output] {output_table}")
        print(f"[target column] {target_col}")
        print(f"[n rows] {len(df_input)}")

    df_query = query_data_many(
        df_input[target_col].tolist(),
        max_retries=max_retries,
        n_workers=n_workers,
        query_verbose=query_verbose,
        verbose=verbose,
    )

    if "RUWE" in df_query.columns:
        df_query = df_query.rename(columns={"RUWE": "RUWE_DR3"})

    if "P_binary" in df_query.columns and "RUWE_DR3" in df_query.columns:
        cols = list(df_query.columns)
        cols.remove("RUWE_DR3")
        cols.insert(cols.index("P_binary") + 1, "RUWE_DR3")
        df_query = df_query[cols]

    duplicated_cols = [col for col in df_query.columns if col in df_input.columns]
    if duplicated_cols and verbose:
        print("[drop duplicated query columns]")
        print(duplicated_cols)

    df_query = df_query.drop(columns=duplicated_cols)
    df_output = pd.concat(
        [df_input.reset_index(drop=True), df_query.reset_index(drop=True)],
        axis=1,
    )

    output_table.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_table, index=False)

    if verbose:
        n_failed = int(df_output["query_failed"].sum()) if "query_failed" in df_output else 0
        print(f"\n[saved] {output_table}")
        print(f"[failed] {n_failed}/{len(df_output)}")

    return df_output


def parallax_to_distance(parallax_mas, parallax_err_mas):
    """
    Convert a positive parallax into a distance and its propagated uncertainty.

    Args:
        parallax_mas: Parallax in milliarcseconds.
        parallax_err_mas: Parallax uncertainty in milliarcseconds.

    Returns:
        A tuple containing distance and uncertainty in parsecs.
    """
    if not np.isfinite(parallax_mas) or parallax_mas <= 0:
        return np.nan, np.nan

    distance = 1000.0 / parallax_mas
    distance_err = (
        1000.0 * parallax_err_mas / parallax_mas**2
        if np.isfinite(parallax_err_mas) and parallax_err_mas >= 0
        else np.nan
    )
    return distance, distance_err


def interp_nan_safe(x, y, x_new):
    """
    Interpolate finite data points without extrapolating beyond the input range.

    Duplicate x coordinates are combined by averaging their corresponding y values.

    Args:
        x: One-dimensional array of input coordinates.
        y: One-dimensional array of input values.
        x_new: Coordinate at which to evaluate the interpolation.

    Returns:
        The interpolated value, or NaN if interpolation is not possible.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or not np.isfinite(x_new):
        return np.nan

    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return np.nan

    x, y = x[good], y[good]
    order = np.argsort(x)
    x, y = x[order], y[order]

    unique_x, inverse = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        y = np.bincount(inverse, weights=y) / np.bincount(inverse)
        x = unique_x

    if x.size < 2 or x_new < x[0] or x_new > x[-1]:
        return np.nan

    return np.interp(x_new, x, y)


def split_gtomo_output_by_input_targets(input_csv, output_csv):
    """
    Split a G-Tomo output table into individual target profiles.

    Args:
        input_csv: CSV file containing a column named 'targets'.
        output_csv: G-Tomo output CSV containing indexed profile columns.

    Returns:
        A dictionary mapping target names to DataFrames containing their profiles.
    """
    df_input = pd.read_csv(input_csv)
    df_output = pd.read_csv(output_csv)

    if "targets" not in df_input.columns:
        raise ValueError("Input CSV must contain a column named 'targets'.")

    target_names = df_input["targets"].map(_clean_target_name).tolist()
    profile_dict = {}

    for i, target_name in enumerate(target_names):
        columns = {
            "distance": f"{i}_x",
            "d_extin": f"{i}_y",
            "d_extin_err": f"{i}_erry",
            "c_extin": f"{i}_yc",
            "c_extin_err": f"{i}_erryc",
        }

        missing = [col for col in columns.values() if col not in df_output.columns]
        if missing:
            raise ValueError(f"Missing G-Tomo columns for target '{target_name}': {missing}")

        profile_dict[target_name] = (
            df_output[list(columns.values())]
            .rename(columns={value: key for key, value in columns.items()})
            .copy()
        )

    return profile_dict


def add_gtomo_from_input_output(
    df_targets_path,
    input_csv,
    output_csv,
    target_col="target_name",
    parallax_col="parallax",
    parallax_err_col="parallax_err",
    A_col="A550",
    A_err_col="A550_err",
    A_floor=None,
    insert_after="Ks_err",
    g_mag_col="G_mag",
    g_err_col="G_err",
    bp_mag_col="G_BP_mag",
    bp_err_col="G_BP_err",
    rp_mag_col="G_RP_mag",
    rp_err_col="G_RP_err",
    mag_col="MG",
    color_col="MBP_MRP",
    extinction_type="A_V",
):
    """
    Add G-Tomo extinction estimates and extinction-corrected Gaia quantities.

    The function calculates distances from parallaxes, interpolates cumulative
    extinction profiles, propagates extinction uncertainties, and derives
    extinction-corrected absolute magnitudes and colors.

    Args:
        df_targets_path: Path to the input target CSV file.
        input_csv: G-Tomo input CSV containing target names.
        output_csv: G-Tomo output CSV containing extinction profiles.
        target_col: Column containing target names.
        parallax_col: Parallax column in milliarcseconds.
        parallax_err_col: Parallax uncertainty column in milliarcseconds.
        A_col: Output column for cumulative extinction.
        A_err_col: Output column for the adopted extinction uncertainty.
        A_floor: Optional minimum adopted extinction uncertainty.
        insert_after: Existing column after which to insert distance and extinction columns.
        g_mag_col: Gaia G-band magnitude column.
        g_err_col: Gaia G-band magnitude uncertainty column.
        bp_mag_col: Gaia BP-band magnitude column.
        bp_err_col: Gaia BP-band magnitude uncertainty column.
        rp_mag_col: Gaia RP-band magnitude column.
        rp_err_col: Gaia RP-band magnitude uncertainty column.
        mag_col: Output column for extinction-corrected absolute G magnitude.
        color_col: Output column for extinction-corrected BP-RP color.
        extinction_type: Extinction convention passed to remove_extinction().

    Returns:
        A DataFrame containing the original target data and the derived quantities.
    """
    if A_floor is not None and (not np.isfinite(A_floor) or A_floor < 0):
        raise ValueError("A_floor must be a finite non-negative value or None.")

    profile_dict = split_gtomo_output_by_input_targets(input_csv, output_csv)
    df_targets = pd.read_csv(df_targets_path, dtype={col: "string" for col in id_columns})

    needed_cols = [
        target_col, parallax_col, parallax_err_col, g_mag_col, g_err_col,
        bp_mag_col, bp_err_col, rp_mag_col, rp_err_col,
    ]
    missing_cols = [col for col in needed_cols if col not in df_targets.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    gtomo_cols = [
        "distance_pc_from_parallax",
        "distance_pc_err_from_parallax",
        A_col,
        f"{A_col}_map_err",
        f"{A_col}_dist_err",
        f"{A_col}_formal_err",
        A_err_col,
    ]
    derived_cols = list(dict.fromkeys(gtomo_cols + [mag_col, color_col]))
    df_targets = df_targets.drop(columns=[col for col in derived_cols if col in df_targets.columns])

    def _safe_float(value):
        """Convert a scalar to a finite float, returning NaN when conversion fails."""
        try:
            result = float(value)
            return result if np.isfinite(result) else np.nan
        except (TypeError, ValueError):
            return np.nan

    rows = []

    for _, row in df_targets.iterrows():
        target_name = _clean_target_name(row[target_col])
        if target_name not in profile_dict:
            raise KeyError(f"Target '{target_name}' not found in the G-Tomo input target list.")

        profile = profile_dict[target_name]
        parallax = _safe_float(row[parallax_col])
        parallax_err = _safe_float(row[parallax_err_col])
        distance, distance_err = parallax_to_distance(parallax, parallax_err)

        x = profile["distance"].to_numpy(dtype=float)
        cumulative_extinction = profile["c_extin"].to_numpy(dtype=float)
        cumulative_extinction_err = profile["c_extin_err"].to_numpy(dtype=float)

        A_center = interp_nan_safe(x, cumulative_extinction, distance)
        A_map_err = interp_nan_safe(x, cumulative_extinction_err, distance)

        if np.isfinite(distance) and np.isfinite(distance_err):
            A_minus = interp_nan_safe(x, cumulative_extinction, max(distance - distance_err, 0.0))
            A_plus = interp_nan_safe(x, cumulative_extinction, distance + distance_err)
            delta_list = [
                abs(value - A_center)
                for value in (A_minus, A_plus)
                if np.isfinite(value) and np.isfinite(A_center)
            ]
            A_dist_err = max(delta_list) if delta_list else np.nan
        else:
            A_dist_err = np.nan

        err_terms = [value for value in (A_map_err, A_dist_err) if np.isfinite(value)]
        A_formal_err = np.hypot(*err_terms) if err_terms else np.nan
        A_used_err = max(A_formal_err, A_floor) if A_floor is not None and np.isfinite(A_formal_err) else (
            A_floor if A_floor is not None else A_formal_err
        )

        obs_mags = np.array([
            _safe_float(row[g_mag_col]),
            _safe_float(row[bp_mag_col]),
            _safe_float(row[rp_mag_col]),
        ])
        obs_err_mags = np.array([
            _safe_float(row[g_err_col]),
            _safe_float(row[bp_err_col]),
            _safe_float(row[rp_err_col]),
        ])

        MG = np.nan
        MBP_MRP = np.nan

        common_ok = (
            np.isfinite(parallax)
            and parallax > 0
            and np.isfinite(parallax_err)
            and parallax_err >= 0
            and np.isfinite(A_center)
            and np.isfinite(A_used_err)
        )

        if common_ok and np.isfinite(obs_mags).all() and np.isfinite(obs_err_mags).all():
            mags_dered, errs_dered = remove_extinction(
                obs_mags,
                obs_err_mags,
                ["G", "G_BP", "G_RP"],
                extinction=A_center,
                extinction_err=A_used_err,
                extinction_type=extinction_type,
                verbose=False,
            )
            abs_mags, _ = cal_abs_mag(mags_dered, errs_dered, parallax, parallax_err)
            MG = abs_mags[0]
            MBP_MRP = abs_mags[1] - abs_mags[2]

        rows.append({
            "distance_pc_from_parallax": distance,
            "distance_pc_err_from_parallax": distance_err,
            A_col: A_center,
            f"{A_col}_map_err": A_map_err,
            f"{A_col}_dist_err": A_dist_err,
            f"{A_col}_formal_err": A_formal_err,
            A_err_col: A_used_err,
            mag_col: MG,
            color_col: MBP_MRP,
        })

    df_add = pd.DataFrame(rows, index=df_targets.index)
    df_out = pd.concat([df_targets, df_add], axis=1)

    move_cols = list(dict.fromkeys(gtomo_cols))
    cols = [col for col in df_out.columns if col not in move_cols + [mag_col, color_col]]
    insert_pos = cols.index(insert_after) + 1 if insert_after in cols else len(cols)

    cols[insert_pos:insert_pos] = move_cols
    cols.extend([mag_col, color_col])

    return df_out[cols]



# Following codes:
# summarizing sampling results
def posterior_summary(samples_2d, sigma=1):
    """Calculate the median and asymmetric uncertainties of posterior samples.

    Args:
        samples_2d: Posterior samples in any array shape.
        sigma: Gaussian-equivalent central interval in units of standard deviations.

    Returns:
        Tuple containing the median, upper uncertainty, and lower uncertainty.
    """
    samples_flat = samples_2d.reshape(-1)
    median = np.percentile(samples_flat, 50)

    cdf_upper = 0.5 * (1 + erf(sigma / sqrt(2)))
    cdf_lower = 1 - cdf_upper
    p_upper = np.percentile(samples_flat, 100 * cdf_upper)
    p_lower = np.percentile(samples_flat, 100 * cdf_lower)

    return median, p_upper - median, median - p_lower


def summarize_cluster_age(
    samples_glob: str,
    df_targets_path: str | os.PathLike,
    save_name: str | os.PathLike = None,
    verbose: bool = True,
):
    """Summarize posterior estimates and merge them with target metadata.

    Args:
        samples_glob: Glob pattern matching posterior `.npz` files.
        df_targets_path: Path to the target metadata CSV file.
        save_name: Optional path for saving the merged DataFrame as a CSV file.
        verbose: Whether to print the first five rows of the merged DataFrame.

    Returns:
        DataFrame containing target metadata, posterior summaries, and P_risk.
    """
    df_targets = pd.read_csv(df_targets_path, dtype={col: "string" for col in id_columns})
    records = []

    for path in glob(str(samples_glob)):
        data = np.load(path, allow_pickle=True)
        run_name = str(data["run_name"])
        model_idx = min(i for i in (run_name.find("_PARSEC"), run_name.find("_MIST")) if i != -1)
        name_cluster = run_name[:model_idx]
        posterior_name = name_cluster.rsplit("_", 1)[0].replace("_", " ").strip()

        P_risk = np.nan
        summary_path = os.path.join(os.path.dirname(path), "summary.txt")

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_text = f.read()

            match = re.search(
                r"Probability of contamination by SGB stars:\s*"
                r"P_risk\s*=\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
                summary_text,
            )
            if match:
                P_risk = float(match.group(1))
        except (OSError, UnicodeError, ValueError):
            pass

        samples = data["samples"]
        blobs = data["blobs"]
        output_list = data["output_list"].tolist()
        param_names = data["param_names"].tolist()

        mass_chain = blobs[:, :, output_list.index("Mass")].T
        age_chain = samples[:, :, param_names.index("Age")].T
        u_chain = blobs[:, :, output_list.index("u")].T
        sigma_chain = samples[:, :, param_names.index("sigma")].T

        burnin = int(data["n_discard"])
        mass_med, mass_up, mass_low = posterior_summary(mass_chain[:, burnin:])
        age_med, age_up, age_low = posterior_summary(age_chain[:, burnin:])
        u_med, u_up, u_low = posterior_summary(u_chain[:, burnin:], sigma=2)
        sig_med, sig_up, sig_low = posterior_summary(sigma_chain[:, burnin:])

        records.append({
            "posterior_name": posterior_name,
            "posterior_path": os.path.abspath(path),
            "mass": mass_med,
            "mass_upper_uncertainty": mass_up,
            "mass_lower_uncertainty": mass_low,
            "age": age_med,
            "age_upper_uncertainty": age_up,
            "age_lower_uncertainty": age_low,
            "u": u_med,
            "u_upper_uncertainty": u_up,
            "u_lower_uncertainty": u_low,
            "sigma": sig_med,
            "sigma_upper_uncertainty": sig_up,
            "sigma_lower_uncertainty": sig_low,
            "P_risk": P_risk,
        })

    df_post = pd.DataFrame(records)

    def get_col_value(row, target_name):
        """Return a row value while tolerating differences in column naming."""
        target = re.sub(r"[\s_]+", " ", str(target_name)).strip().lower()
        for col in row.index:
            if re.sub(r"[\s_]+", " ", str(col)).strip().lower() == target:
                return row[col]
        return np.nan

    def clean_catalog_id(x):
        """Normalize catalog IDs and remove unnecessary decimal suffixes."""
        if pd.isna(x):
            return ""

        if isinstance(x, (int, np.integer)):
            return str(x)

        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return ""
            return str(int(x)) if x.is_integer() else str(x)

        s = str(x).strip()
        if s.lower() in {"", "nan", "none", "--", "<na>"}:
            return ""

        if re.fullmatch(r"[+-]?\d+\.0+", s):
            s = s.split(".")[0]

        return s

    def build_posterior_name(row):
        """Build a posterior name using the available catalog identifiers."""
        hip_id = clean_catalog_id(get_col_value(row, "HIP_ID"))
        hd_id = clean_catalog_id(get_col_value(row, "HD_ID"))

        if hip_id:
            return f"HIP {hip_id}"
        if hd_id:
            return f"HD {hd_id}"

        other_name = get_col_value(row, "Other Name")
        if pd.isna(other_name):
            return ""

        other_name = str(other_name).strip()
        if other_name.lower() in {"", "nan", "none", "--", "<na>"}:
            return ""

        return re.sub(r"\s+", " ", other_name)

    df_targets = df_targets.copy()
    df_targets["posterior_name"] = df_targets.apply(build_posterior_name, axis=1)

    df_all = pd.merge(df_targets, df_post, on="posterior_name", how="left")
    df_all = df_all.sort_values("mass", ascending=False).reset_index(drop=True)

    if "P_risk" in df_all.columns:
        df_all = df_all[[col for col in df_all.columns if col != "P_risk"] + ["P_risk"]]

    if save_name is not None:
        df_all.to_csv(save_name, index=False)

    if verbose:
        print(df_all.head())

    return df_all



# Following codes:
# plotting CMDs and clsuter ages
def _get_quantiles(grid, pdf):
    """Calculate the 15.85th, 50th, and 84.15th percentiles of a gridded PDF.

    Args:
        grid (array-like): Strictly increasing grid points.
        pdf (array-like): Probability density evaluated on the grid.

    Returns:
        tuple: Lower, median, and upper quantiles.
    """
    grid = np.asarray(grid, dtype=float)
    pdf = np.asarray(pdf, dtype=float)

    if grid.ndim != 1 or pdf.ndim != 1 or len(grid) != len(pdf) or len(grid) < 2:
        raise ValueError("Grid and PDF must be 1D arrays of equal length, with at least two points.")
    if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        raise ValueError("Grid must contain finite, strictly increasing values.")
    if not np.all(np.isfinite(pdf)):
        raise ValueError("PDF contains non-finite values.")

    pdf = np.maximum(pdf, 0.0)
    normalization = np.trapezoid(pdf, grid)
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Cannot normalize the PDF.")

    pdf /= normalization
    cdf = np.zeros_like(grid)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * np.diff(grid))
    cdf /= cdf[-1]

    return tuple(np.interp([0.1585, 0.5, 0.8415], cdf, grid))


def _prepare_cluster_data(
    df_post_path,
    kept_list=None,
    dropped_list=None,
    weight=False,
    age_name="Age",
    MH_name="MH",
    max_samples_per_star=100_000,
    random_seed=42,
):
    """Load, filter, and validate stellar posterior samples.

    Args:
        df_post_path (str or Path): CSV containing stellar posterior metadata.
        kept_list (list of str or None): Optional list of stars to retain.
        dropped_list (list of str or None): Optional list of stars to exclude.
        weight (bool or str): Equal weights if False or None; otherwise, a weight column name.
        age_name (str): Age parameter name in the MCMC files.
        MH_name (str): Metallicity parameter name in the MCMC files.
        max_samples_per_star (int or None): Maximum samples retained per star.
        random_seed (int): Random seed for reproducible subsampling.

    Returns:
        tuple: Filtered DataFrame, weights, and a list of two-column sample arrays.
    """
    required_columns = {
        "posterior_name",
        "posterior_path",
        "mass",
        "mass_lower_uncertainty",
        "mass_upper_uncertainty",
        "age",
        "age_lower_uncertainty",
        "age_upper_uncertainty",
    }

    df = pd.read_csv(df_post_path)
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required CSV columns: {sorted(missing_columns)}")

    df = df.copy()
    df["posterior_name"] = df["posterior_name"].astype("string").str.strip()
    df["posterior_path"] = df["posterior_path"].astype("string").str.strip()

    if kept_list is not None:
        keep_set = {str(name).strip() for name in kept_list}
        df = df[df["posterior_name"].isin(keep_set)]

    if dropped_list is not None:
        drop_set = {str(name).strip() for name in dropped_list}
        df = df[~df["posterior_name"].isin(drop_set)]

    # Remove missing paths and warn about files that cannot be found.
    valid_paths = df["posterior_path"].notna() & df["posterior_path"].ne("")
    df = df[valid_paths].copy()

    existing = df["posterior_path"].map(lambda path: Path(path).is_file())
    for name, path in zip(df.loc[~existing, "posterior_name"], df.loc[~existing, "posterior_path"]):
        warnings.warn(f"Skipping {name}: posterior file does not exist: {path}", stacklevel=2)
    df = df[existing].reset_index(drop=True)

    if df.empty:
        raise ValueError("No stars remain after filtering and checking posterior files.")

    # Convert columns used by the stellar mass-age plot to numeric values.
    plot_columns = [
        "mass",
        "mass_lower_uncertainty",
        "mass_upper_uncertainty",
        "age",
        "age_lower_uncertainty",
        "age_upper_uncertainty",
    ]
    for column in plot_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_plot_data = ~np.isfinite(df[plot_columns].to_numpy(dtype=float)).all(axis=1)
    if invalid_plot_data.any():
        names = df.loc[invalid_plot_data, "posterior_name"].tolist()
        raise ValueError(f"Non-finite mass, age, or uncertainty values for: {names}")

    if (df[["mass_lower_uncertainty", "mass_upper_uncertainty",
            "age_lower_uncertainty", "age_upper_uncertainty"]] < 0).any().any():
        raise ValueError("Mass and age uncertainties must be non-negative.")

    if weight is False or weight is None:
        alphas = np.ones(len(df), dtype=float)
    elif isinstance(weight, str):
        if weight not in df.columns:
            raise KeyError(f"Weight column '{weight}' is not found.")
        alphas = pd.to_numeric(df[weight], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(alphas)) or np.any(alphas < 0):
            raise ValueError(f"Weight column '{weight}' must contain finite, non-negative values.")
    else:
        raise TypeError("weight must be False, None, or a column name.")

    if not np.any(alphas > 0):
        raise ValueError("At least one stellar weight must be greater than zero.")

    if max_samples_per_star is not None:
        if (isinstance(max_samples_per_star, (bool, np.bool_))
                or not isinstance(max_samples_per_star, (int, np.integer))
                or max_samples_per_star < 2):
            raise ValueError("max_samples_per_star must be an integer >= 2 or None.")

    rng = np.random.default_rng(random_seed)
    samples_list = []

    for name, path in zip(df["posterior_name"], df["posterior_path"]):
        try:
            with np.load(path, allow_pickle=True) as data:
                required_keys = {"samples", "param_names", "n_discard"}
                missing_keys = required_keys - set(data.files)
                if missing_keys:
                    raise KeyError(f"Missing posterior keys: {sorted(missing_keys)}")

                samples = np.asarray(data["samples"], dtype=float)
                param_names = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in data["param_names"]
                ]
                n_discard = np.asarray(data["n_discard"])

        except Exception as exc:
            raise ValueError(f"Failed to read posterior for {name} from {path}: {exc}") from exc

        if samples.ndim != 3 or samples.shape[2] != len(param_names):
            raise ValueError(f"Invalid sample shape or parameter names for {name}: {samples.shape}")
        if samples.shape[0] < 2 or samples.shape[1] < 1:
            raise ValueError(f"Insufficient MCMC steps or walkers for {name}.")

        missing_params = {age_name, MH_name} - set(param_names)
        if missing_params:
            raise ValueError(f"Missing parameters for {name}: {sorted(missing_params)}")

        n_walkers = samples.shape[1]

        # NaN burn-in means that burn-in has not been determined.
        if n_discard.ndim == 0:
            if not np.isfinite(n_discard):
                warnings.warn(f"{name}: n_discard is NaN; using the full chain.", stacklevel=2)
                burns = np.zeros(n_walkers, dtype=int)
            else:
                if n_discard != int(n_discard):
                    raise ValueError(f"{name}: n_discard must be an integer.")
                burns = np.full(n_walkers, int(n_discard), dtype=int)
        elif n_discard.ndim == 1 and len(n_discard) == n_walkers:
            burns = np.zeros(n_walkers, dtype=int)
            for walker, value in enumerate(n_discard):
                if not np.isfinite(value):
                    warnings.warn(
                        f"{name}: n_discard for walker {walker} is NaN; using its full chain.",
                        stacklevel=2,
                    )
                elif value != int(value):
                    raise ValueError(f"{name}: n_discard for walker {walker} must be an integer.")
                else:
                    burns[walker] = int(value)
        else:
            raise ValueError(f"{name}: n_discard must be a scalar or an array of length {n_walkers}.")

        if np.any(burns < 0) or np.any(burns >= samples.shape[0]):
            raise ValueError(f"{name}: n_discard is outside the valid chain range.")

        age_idx = param_names.index(age_name)
        MH_idx = param_names.index(MH_name)
        age = np.concatenate([samples[burns[w]:, w, age_idx] for w in range(n_walkers)])
        MH = np.concatenate([samples[burns[w]:, w, MH_idx] for w in range(n_walkers)])

        valid = np.isfinite(age) & np.isfinite(MH)
        age, MH = age[valid], MH[valid]

        if len(age) < 2:
            warnings.warn(f"Skipping {name}: fewer than two valid Age-[M/H] samples.", stacklevel=2)
            samples_list.append(None)
            continue

        if max_samples_per_star is not None and len(age) > max_samples_per_star:
            idx = rng.choice(len(age), size=max_samples_per_star, replace=False)
            age, MH = age[idx], MH[idx]

        sample_array = np.column_stack([age, MH])

        if np.linalg.matrix_rank(np.cov(sample_array.T)) < 2:
            warnings.warn(f"Skipping {name}: Age-[M/H] samples have singular covariance.", stacklevel=2)
            samples_list.append(None)
            continue

        samples_list.append(sample_array)

    # Keep weights and sample arrays aligned after removing invalid stars.
    valid_stars = np.array([samples is not None for samples in samples_list])
    if not valid_stars.all():
        df = df.loc[valid_stars].reset_index(drop=True)
        alphas = alphas[valid_stars]
        samples_list = [samples for samples in samples_list if samples is not None]

    if df.empty or not np.any(alphas > 0):
        raise ValueError("No valid stars with positive weights remain.")

    return df, alphas, samples_list


def _build_age_mh_grids(samples_list, n_age_grid=500, n_MH_grid=300, age_range=None,
                        MH_range=(-0.4, 0.4)):
    """Build common Age and metallicity grids for posterior combination.

    Args:
        samples_list (list of ndarray): Per-star arrays with Age and [M/H] columns.
        n_age_grid (int): Number of Age grid points.
        n_MH_grid (int): Number of metallicity grid points.
        age_range (tuple or None): Age grid limits; inferred from samples if None.
        MH_range (tuple or None): Metallicity grid limits; inferred from samples if None.

    Returns:
        tuple: Age and metallicity grids.
    """
    for name, value in (("n_age_grid", n_age_grid), ("n_MH_grid", n_MH_grid)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 2:
            raise ValueError(f"{name} must be an integer >= 2.")

    all_age = np.concatenate([samples[:, 0] for samples in samples_list])
    all_MH = np.concatenate([samples[:, 1] for samples in samples_list])

    if age_range is None:
        age_lo, age_hi = np.percentile(all_age, [0.1, 99.9])
        age_pad = 0.05 * (age_hi - age_lo)
        age_range = (age_lo - age_pad, age_hi + age_pad)

    if MH_range is None:
        MH_lo, MH_hi = np.percentile(all_MH, [0.1, 99.9])
        MH_pad = 0.05 * (MH_hi - MH_lo)
        MH_range = (MH_lo - MH_pad, MH_hi + MH_pad)

    for name, limits in (("age_range", age_range), ("MH_range", MH_range)):
        if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[0] >= limits[1]:
            raise ValueError(f"{name} must contain two finite, increasing limits.")

    age_grid = np.linspace(*age_range, n_age_grid)
    MH_grid = np.linspace(*MH_range, n_MH_grid)

    # Warn when observed samples fall outside the requested grid.
    for label, values, limits in (
        ("Age", all_age, age_range),
        ("[M/H]", all_MH, MH_range),
    ):
        if np.any((values < limits[0]) | (values > limits[1])):
            warnings.warn(f"Some {label} samples fall outside the grid and may be truncated.", stacklevel=2)

    return age_grid, MH_grid


def _combine_cluster_posteriors(
    samples_list,
    alphas,
    age_grid,
    MH_grid,
    MH_prior_median=0.09,
    MH_prior_std=0.079,
    bw_method=None,
):
    """Combine stellar Age-[M/H] posteriors and calculate a 1D Age reference.

    The 2D calculation divides each stellar posterior by the specified [M/H] prior.
    This recovers a likelihood only if that prior matches the prior used in the original
    MCMC and other relevant prior effects are appropriately accounted for.

    Args:
        samples_list (list of ndarray): Per-star Age-[M/H] samples.
        alphas (array-like): Non-negative stellar weights.
        age_grid (ndarray): Common Age grid.
        MH_grid (ndarray): Common metallicity grid.
        MH_prior_median (float): Mean of the Gaussian metallicity prior.
        MH_prior_std (float): Standard deviation of the Gaussian metallicity prior.
        bw_method: Bandwidth method passed to scipy.stats.gaussian_kde.

    Returns:
        tuple: Dictionaries containing the 2D and 1D posterior results.
    """
    if not np.isfinite(MH_prior_median) or not np.isfinite(MH_prior_std) or MH_prior_std <= 0:
        raise ValueError("MH_prior_median must be finite and MH_prior_std must be finite and positive.")

    alphas = np.asarray(alphas, dtype=float)
    if len(samples_list) != len(alphas) or not np.all(np.isfinite(alphas)) or np.any(alphas < 0):
        raise ValueError("Samples and weights must have matching lengths and finite non-negative weights.")
    if not np.any(alphas > 0):
        raise ValueError("At least one stellar weight must be greater than zero.")

    AGE, MH_mesh = np.meshgrid(age_grid, MH_grid, indexing="ij")
    positions = np.vstack([AGE.ravel(), MH_mesh.ravel()])
    eps = 1e-300

    prior_1d = stats_norm.pdf(MH_grid, loc=MH_prior_median, scale=MH_prior_std)
    prior_2d = np.broadcast_to(prior_1d, (len(age_grid), len(MH_grid)))
    log_cluster_2d = np.log(np.maximum(prior_2d, eps)).copy()

    log_cluster_1d = np.zeros_like(age_grid, dtype=float)

    for index, (samples, alpha) in enumerate(zip(samples_list, alphas)):
        if alpha == 0:
            continue

        try:
            kde_2d = gaussian_kde(samples.T, bw_method=bw_method)
            pdf_2d = kde_2d(positions).reshape(len(age_grid), len(MH_grid))
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"2D KDE failed for star index {index}: {exc}") from exc

        norm_2d = np.trapezoid(np.trapezoid(pdf_2d, MH_grid, axis=1), age_grid)
        if not np.isfinite(norm_2d) or norm_2d <= 0:
            raise ValueError(f"Invalid 2D KDE normalization for star index {index}.")

        pdf_2d /= norm_2d
        likelihood_2d = pdf_2d / np.maximum(prior_2d, eps)
        log_cluster_2d += alpha * np.log(np.maximum(likelihood_2d, eps))

        try:
            kde_1d = gaussian_kde(samples[:, 0], bw_method=bw_method)
            pdf_1d = np.maximum(kde_1d(age_grid), eps)
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"1D KDE failed for star index {index}: {exc}") from exc

        norm_1d = np.trapezoid(pdf_1d, age_grid)
        if not np.isfinite(norm_1d) or norm_1d <= 0:
            raise ValueError(f"Invalid 1D KDE normalization for star index {index}.")

        log_cluster_1d += alpha * np.log(pdf_1d / norm_1d)

    log_cluster_2d -= np.max(log_cluster_2d)
    joint_pdf = np.exp(log_cluster_2d)
    joint_norm = np.trapezoid(np.trapezoid(joint_pdf, MH_grid, axis=1), age_grid)

    if not np.isfinite(joint_norm) or joint_norm <= 0:
        raise ValueError("Invalid joint posterior normalization.")

    joint_pdf /= joint_norm
    age_pdf_2d = np.trapezoid(joint_pdf, MH_grid, axis=1)
    MH_pdf_2d = np.trapezoid(joint_pdf, age_grid, axis=0)

    for label, grid, pdf in (("Age", age_grid, age_pdf_2d), ("[M/H]", MH_grid, MH_pdf_2d)):
        norm_value = np.trapezoid(pdf, grid)
        if not np.isfinite(norm_value) or norm_value <= 0:
            raise ValueError(f"Invalid marginalized {label} posterior normalization.")
        pdf /= norm_value

    log_cluster_1d -= np.max(log_cluster_1d)
    age_pdf_1d = np.exp(log_cluster_1d)
    norm_1d = np.trapezoid(age_pdf_1d, age_grid)
    if not np.isfinite(norm_1d) or norm_1d <= 0:
        raise ValueError("Invalid combined 1D Age posterior normalization.")
    age_pdf_1d /= norm_1d

    # Warn if the marginalized posterior remains appreciable at either grid boundary.
    for label, grid, pdf in (("Age", age_grid, age_pdf_2d), ("[M/H]", MH_grid, MH_pdf_2d)):
        if np.max(pdf[[0, -1]]) > 0.01 * np.max(pdf):
            warnings.warn(f"The marginalized {label} posterior remains appreciable at a grid boundary.",
                          stacklevel=2)

    result_2d = {
        "age_grid": age_grid,
        "MH_grid": MH_grid,
        "joint_pdf": joint_pdf,
        "age_pdf": age_pdf_2d,
        "MH_pdf": MH_pdf_2d,
        "age_quantiles": _get_quantiles(age_grid, age_pdf_2d),
        "MH_quantiles": _get_quantiles(MH_grid, MH_pdf_2d),
    }
    result_1d = {
        "age_grid": age_grid,
        "age_pdf": age_pdf_1d,
        "age_quantiles": _get_quantiles(age_grid, age_pdf_1d),
    }

    return result_2d, result_1d


def _plot_cluster_age_results(
    cluster,
    df,
    result_2d,
    result_1d,
    axes=None,
    title=None,
    age_xlim=None,
    legend_loc="upper right",
    save_name=None,
    show=True,
):
    """Plot stellar mass-age estimates and combined Age posteriors in a vertical layout.

    Args:
        cluster (str): Cluster name.
        df (DataFrame): Filtered stellar metadata.
        result_2d (dict): Combined 2D Age-[M/H] posterior.
        result_1d (dict): Combined 1D Age posterior.
        axes (tuple or None): Optional pair of existing Matplotlib axes.
        title (str, bool or None): Plot title; False suppresses the title.
        age_xlim (tuple or None): Explicit Age-axis limits.
        legend_loc (str): Legend location.
        save_name (str, Path or None): Optional output path.
        show (bool): Whether to show a newly created figure.

    Returns:
        tuple: Figure and dictionary containing the stellar and Age axes.
    """
    label_fs, tick_fs, legend_fs = 14, 12, 11
    sigma_color, sigma_alpha = "lightskyblue", 0.25
    color_2d, color_1d = "tab:blue", "tab:orange"
    age_grid = result_2d["age_grid"]
    age_pdf_2d = result_2d["age_pdf"]
    age_pdf_1d = result_1d["age_pdf"]
    age16, age50, age84 = result_2d["age_quantiles"]
    age16_1d, age50_1d, age84_1d = result_1d["age_quantiles"]

    own_figure = axes is None
    if own_figure:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(6, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
        )
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain exactly two Matplotlib axes: (ax_top, ax_bottom).")
        ax_top, ax_bot = axes
        if ax_top.figure is not ax_bot.figure:
            raise ValueError("Both axes must belong to the same Figure.")
        fig = ax_top.figure

    age_min = np.min(df["age"] - df["age_lower_uncertainty"])
    age_max = np.max(df["age"] + df["age_upper_uncertainty"])
    age_pad = 0.05 * (age_max - age_min)
    plot_min, plot_max = age_min - age_pad, age_max + age_pad

    ax_top.axvspan(age16, age84, facecolor=sigma_color, alpha=sigma_alpha, edgecolor="none", zorder=0)
    ax_top.errorbar(
        x=df["age"], y=df["mass"],
        xerr=df[["age_lower_uncertainty", "age_upper_uncertainty"]].to_numpy().T,
        yerr=df[["mass_lower_uncertainty", "mass_upper_uncertainty"]].to_numpy().T,
        fmt="o", color="black", zorder=2,
    )
    ax_top.set_ylabel(r"Mass ($M_\odot$)", fontsize=label_fs)
    ax_top.tick_params(axis="both", labelsize=tick_fs)
    ax_top.tick_params(labelbottom=False)

    if title is False:
        ax_top.set_title("")
    elif title is None:
        ax_top.set_title(f"Mass-Age relation of the {cluster}")
    else:
        ax_top.set_title(str(title))

    ax_bot.plot(age_grid, age_pdf_2d, color=color_2d, lw=2, zorder=3)
    ax_bot.fill_between(age_grid, 0, age_pdf_2d, color=color_2d, alpha=0.15, zorder=1)
    ax_bot.axvspan(age16, age84, facecolor=sigma_color, alpha=sigma_alpha, edgecolor="none", zorder=0)
    ax_bot.plot(age_grid, age_pdf_1d, color=color_1d, linestyle="--", lw=2, zorder=4)

    age_2d_label = (
        r"common-$\tau$--$\mathrm{[M/H]}$" + "\n"
        + rf"${age50:.0f}^{{+{age84 - age50:.0f}}}_{{-{age50 - age16:.0f}}}\,\mathrm{{Myr}}$"
    )
    age_1d_label = (
        r"common-$\tau$" + "\n"
        + rf"${age50_1d:.0f}^{{+{age84_1d - age50_1d:.0f}}}_{{-{age50_1d - age16_1d:.0f}}}"
        rf"\,\mathrm{{Myr}}$"
    )
    ax_bot.legend(
        handles=[
            Patch(facecolor=sigma_color, alpha=sigma_alpha, edgecolor=color_2d, label=age_2d_label),
            Line2D([0], [0], color=color_1d, linestyle="--", lw=2, label=age_1d_label),
        ],
        frameon=False, fontsize=legend_fs, loc=legend_loc,
    )
    ax_bot.set_xlabel("Age (Myr)", fontsize=label_fs)
    ax_bot.set_ylabel("Probability density", fontsize=label_fs)
    ax_bot.tick_params(axis="both", labelsize=tick_fs)

    if age_xlim is not None:
        ax_bot.set_xlim(age_xlim)
    else:
        ax_bot.set_xlim(min(plot_min, age_grid[0]), max(plot_max, age_grid[-1]))

    if own_figure:
        fig.subplots_adjust(hspace=0)

    if save_name is not None:
        save_path = Path(save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.05)

    if show and own_figure:
        plt.show()

    return fig, {"stellar": ax_top, "age": ax_bot}


def plot_cluster_age_2d(
    cluster,
    df_post_path,
    kept_list=None,
    dropped_list=None,
    weight=False,
    MH_prior_median=0.09,
    MH_prior_std=0.079,
    age_name="Age",
    MH_name="MH",
    n_age_grid=500,
    n_MH_grid=300,
    age_range=None,
    MH_range=(-0.4, 0.4),
    bw_method=None,
    max_samples_per_star=100_000,
    random_seed=42,
    title=None,
    save_name=None,
    age_xlim=None,
    legend_loc="upper right",
    axes=None,
    show=True,
):
    """Combine stellar Age-[M/H] posteriors and plot the cluster age estimate.

    Args:
        cluster (str): Cluster name.
        df_post_path (str or Path): CSV containing stellar posterior metadata.
        kept_list (list of str or None): Optional list of stars to retain.
        dropped_list (list of str or None): Optional list of stars to exclude.
        weight (bool or str): Equal weights if False or None; otherwise, a weight column name.
        MH_prior_median (float): Mean of the Gaussian [M/H] prior.
        MH_prior_std (float): Standard deviation of the Gaussian [M/H] prior.
        age_name (str): Age parameter name in the MCMC files.
        MH_name (str): Metallicity parameter name in the MCMC files.
        n_age_grid (int): Number of Age grid points.
        n_MH_grid (int): Number of metallicity grid points.
        age_range (tuple or None): Age grid limits; inferred from samples if None.
        MH_range (tuple or None): Metallicity grid limits; inferred from samples if None.
        bw_method: Bandwidth method passed to scipy.stats.gaussian_kde.
        max_samples_per_star (int or None): Maximum samples retained per star.
        random_seed (int): Random seed for reproducible subsampling.
        title (str, bool or None): Plot title; False suppresses the title.
        save_name (str, Path or None): Optional output path.
        age_xlim (tuple or None): Explicit Age-axis limits.
        legend_loc (str): Legend location.
        axes (tuple or None): Optional pair of existing Matplotlib axes.
        show (bool): Whether to show a newly created figure.

    Returns:
        dict: Filtered data, weights, posterior results, Figure, and axes.
    """
    df, alphas, samples_list = _prepare_cluster_data(
        df_post_path=df_post_path,
        kept_list=kept_list,
        dropped_list=dropped_list,
        weight=weight,
        age_name=age_name,
        MH_name=MH_name,
        max_samples_per_star=max_samples_per_star,
        random_seed=random_seed,
    )

    print(f"\n{cluster}: {len(df)} stars finally used")
    for name, alpha in zip(df["posterior_name"], alphas):
        print(f"  {name:<15} weight = {alpha:.3f}")

    age_grid, MH_grid = _build_age_mh_grids(
        samples_list=samples_list,
        n_age_grid=n_age_grid,
        n_MH_grid=n_MH_grid,
        age_range=age_range,
        MH_range=MH_range,
    )

    result_2d, result_1d = _combine_cluster_posteriors(
        samples_list=samples_list,
        alphas=alphas,
        age_grid=age_grid,
        MH_grid=MH_grid,
        MH_prior_median=MH_prior_median,
        MH_prior_std=MH_prior_std,
        bw_method=bw_method,
    )

    age16, age50, age84 = result_2d["age_quantiles"]
    MH16, MH50, MH84 = result_2d["MH_quantiles"]
    age16_1d, age50_1d, age84_1d = result_1d["age_quantiles"]

    print("\n2D Age-[M/H] combination:")
    print(f"  Age = {age50:.1f} -{age50 - age16:.1f} +{age84 - age50:.1f} Myr")
    print(f"  [M/H] = {MH50:.3f} -{MH50 - MH16:.3f} +{MH84 - MH50:.3f}")
    print("\n1D Age combination:")
    print(f"  Age = {age50_1d:.1f} -{age50_1d - age16_1d:.1f} +{age84_1d - age50_1d:.1f} Myr")

    fig, plot_axes = _plot_cluster_age_results(
        cluster=cluster,
        df=df,
        result_2d=result_2d,
        result_1d=result_1d,
        axes=axes,
        title=title,
        age_xlim=age_xlim,
        legend_loc=legend_loc,
        save_name=save_name,
        show=show,
    )

    return {
        "df_used": df,
        "weights": alphas,
        "result_2d": result_2d,
        "result_1d": result_1d,
        "fig": fig,
        "axes": plot_axes,
    }


