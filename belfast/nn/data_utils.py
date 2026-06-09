import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_data(
    df,
    abs_mags,
    abs_err_mags,
    split_ratio=(0.8, 0.1, 0.1),
    device=None,
    extra_features=None
):
    """
    Prepare normalized input/output tensors and compute deviation for each sample.
    Optionally include extra features like 'logAge', 'logTe', 'logg', 'Rpol'.

    This function normalizes input features and output magnitudes using MinMaxScaler,
    computes the deviation per sample as the average normalized difference between
    observed and absolute magnitudes, optionally appends extra features, and splits
    the dataset into training, validation, and test sets based on the provided split_ratio.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and magnitude columns.
        abs_mags (list or array): Absolute magnitudes for normalization.
        abs_err_mags (list or array): Errors in absolute magnitudes for deviation calculation.
        split_ratio (tuple): Ratios for train, validation, and test splits (default (0.8, 0.1, 0.1)).
        device (torch.device, optional): Device to place tensors on (CPU or GPU). If None, auto-detect.
        extra_features (list, optional): List of additional feature names to include.

    Returns:
        tuple: Tensors for training, validation, and test sets (inputs, outputs, deviations),
               and a dictionary of fitted scalers for each feature and output.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    abs_mags = np.array(abs_mags)
    abs_err_mags = np.array(abs_err_mags)
    bands_list = df.columns[-len(abs_mags):]
    Y_all = df[bands_list].values

    # ✅ Compute deviation per sample as the average normalized difference
    deviation = np.abs((Y_all - abs_mags) / abs_err_mags).mean(axis=1)
    deviation_tensor = torch.tensor(deviation, dtype=torch.float32).to(device)

    # ✅ Extract and normalize input features
    feature_list = []

    def fit_and_scale(key, data, scaler=None):
        s = scaler if scaler else MinMaxScaler()
        return s.fit(data), s.fit_transform(data)

    mass = df[['Mass']].values
    mh = df[['MH']].values
    log1p_req = np.log1p(df[['Req']].values)
    logl = df[['logL']].values
    omega_k = df[['omega_k']].values
    incl = df[['Inclination']].values
    Y_raw = Y_all

    scalers = {}

    scalers['mass'], mass_scaled = fit_and_scale('mass', mass)
    scalers['mh'], mh_scaled = fit_and_scale('mh', mh, MinMaxScaler(feature_range=(-1, 1)))
    scalers['log1p_req'], log1p_req_scaled = fit_and_scale('log1p_req', log1p_req)
    scalers['logl'], logl_scaled = fit_and_scale('logl', logl)
    scalers['incl'], incl_scaled = fit_and_scale('incl', incl)

    feature_list += [mass_scaled, mh_scaled, log1p_req_scaled, logl_scaled, omega_k, incl_scaled]

    if extra_features:
        if 'logAge' in extra_features:
            logage = df[['logAge']].values
            scalers['logAge'], logage_scaled = fit_and_scale('logAge', logage)
            feature_list.append(logage_scaled)
        if 'logTe' in extra_features:
            logteff = df[['logTe']].values
            scalers['logTe'], logteff_scaled = fit_and_scale('logTe', logteff)
            feature_list.append(logteff_scaled)
        if 'logg' in extra_features:
            logteff = df[['logg']].values
            scalers['logg'], logteff_scaled = fit_and_scale('logg', logteff)
            feature_list.append(logteff_scaled)
        if 'Rpol' in extra_features:
            log1p_rpol = np.log1p(df[['Rpol']].values)
            scalers['log1p_rpol'], log1p_rpol_scaled = fit_and_scale('log1p_rpol', log1p_rpol)
            feature_list.append(log1p_rpol_scaled)

    X_scaled = np.concatenate(feature_list, axis=1)
    scalers['y'], Y_scaled = fit_and_scale('y', Y_raw)

    # ✅ Split into train, val, test sets
    X_temp, X_test, Y_temp, Y_test, dev_temp, dev_test = train_test_split(
        X_scaled, Y_scaled, deviation, test_size=split_ratio[2], random_state=42
    )
    val_ratio_adj = split_ratio[1] / (split_ratio[0] + split_ratio[1])
    X_train, X_val, Y_train, Y_val, dev_train, dev_val = train_test_split(
        X_temp, Y_temp, dev_temp, test_size=val_ratio_adj, random_state=42
    )

    return (
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(Y_train, dtype=torch.float32).to(device),
        torch.tensor(dev_train, dtype=torch.float32).to(device),
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(Y_val, dtype=torch.float32).to(device),
        torch.tensor(dev_val, dtype=torch.float32).to(device),
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(Y_test, dtype=torch.float32).to(device),
        torch.tensor(dev_test, dtype=torch.float32).to(device),
        scalers
    )

def transform_data_with_scalers(
    df,
    scalers,
    abs_mags,
    abs_err_mags,
    split_ratio=(0.8, 0.1, 0.1),
    device=None,
    deviation=-1,
    extra_features=None
):
    """
    Apply pre-fitted scalers to new data, optionally filter samples by deviation threshold,
    recompute normalized input/output tensors, split into train/validation/test sets, and return tensors.

    This function uses the provided scalers to transform features and outputs in the new dataset,
    optionally filters samples based on maximum allowed deviation from absolute magnitudes,
    and returns PyTorch tensors for model training or evaluation.

    Args:
        df (pd.DataFrame): New input DataFrame with features and magnitude columns.
        scalers (dict): Dictionary of pre-fitted scalers for features and outputs.
        abs_mags (list or array): Absolute magnitudes for deviation calculation.
        abs_err_mags (list or array): Errors in absolute magnitudes for deviation calculation.
        split_ratio (tuple): Ratios for train, validation, and test splits (default (0.8, 0.1, 0.1)).
        device (torch.device, optional): Device to place tensors on (CPU or GPU). If None, auto-detect.
        deviation (float, optional): Maximum allowed deviation threshold for filtering samples. If <=0, no filtering.
        extra_features (list, optional): List of additional feature names to include.

    Returns:
        tuple: Tensors for training, validation, and test sets (inputs, outputs, deviations).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if deviation > 0 and abs_mags is not None and abs_err_mags is not None:
        abs_mags = np.array(abs_mags)
        abs_err_mags = np.array(abs_err_mags)
        bands_list = df.columns[-len(abs_mags):]
        Y_all = df[bands_list].values
        Y_dev = np.abs((Y_all - abs_mags) / abs_err_mags)
        mask = (Y_dev <= deviation).all(axis=1)
        df = df[mask].copy()
        print(f"🎯 Filtered training data within {deviation}σ: {len(df)} samples")
    else:
        print(f"Using all data for training: {len(df)} samples")

    abs_mags = np.array(abs_mags)
    abs_err_mags = np.array(abs_err_mags)
    bands_list = df.columns[-len(abs_mags):]
    Y_all = df[bands_list].values
    deviation = np.abs((Y_all - abs_mags) / abs_err_mags).mean(axis=1)

    # Use pre-fitted scalers
    feature_list = []
    mass_scaled = scalers['mass'].transform(df[['Mass']].values)
    mh_scaled = scalers['mh'].transform(df[['MH']].values)
    log1p_req_scaled = scalers['log1p_req'].transform(np.log1p(df[['Req']].values))
    logl_scaled = scalers['logl'].transform(df[['logL']].values)
    omega_k = df[['omega_k']].values
    incl_scaled = scalers['incl'].transform(df[['Inclination']].values)

    feature_list += [mass_scaled, mh_scaled, log1p_req_scaled, logl_scaled, omega_k, incl_scaled]

    if extra_features:
        if 'logAge' in extra_features:
            logage_scaled = scalers['logAge'].transform(df[['logAge']].values)
            feature_list.append(logage_scaled)
        if 'logTe' in extra_features:
            logteff_scaled = scalers['logTe'].transform(df[['logTe']].values)
            feature_list.append(logteff_scaled)
        if 'logg' in extra_features:
            logteff_scaled = scalers['logg'].transform(df[['logg']].values)
            feature_list.append(logteff_scaled)
        if 'Rpol' in extra_features:
            log1p_rpol_scaled = scalers['log1p_rpol'].transform(np.log1p(df[['Rpol']].values))
            feature_list.append(log1p_rpol_scaled)

    X_scaled = np.concatenate(feature_list, axis=1)
    Y_scaled = scalers['y'].transform(Y_all)

    # Split into train / val / test sets
    if split_ratio[2] > 0:
        # Reserve a portion for the test set
        X_temp, X_test, Y_temp, Y_test, dev_temp, dev_test = train_test_split(
            X_scaled, Y_scaled, deviation, test_size=split_ratio[2], random_state=42
        )
    else:
        # Do not split test set; use all data in temp
        X_temp, X_test = X_scaled, 0
        Y_temp, Y_test = Y_scaled, 0
        dev_temp, dev_test = deviation, 0

    if split_ratio[1] > 0:
        # Calculate adjusted val ratio within the remaining data (train + val)
        val_ratio_adj = split_ratio[1] / (split_ratio[0] + split_ratio[1])
        X_train, X_val, Y_train, Y_val, dev_train, dev_val = train_test_split(
            X_temp, Y_temp, dev_temp, test_size=val_ratio_adj, random_state=42
        )
    else:
        # Do not split val set; use all data for training
        X_train, X_val = X_temp, 0
        Y_train, Y_val = Y_temp, 0
        dev_train, dev_val = dev_temp, 0


    return (
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(Y_train, dtype=torch.float32).to(device),
        torch.tensor(dev_train, dtype=torch.float32).to(device),
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(Y_val, dtype=torch.float32).to(device),
        torch.tensor(dev_val, dtype=torch.float32).to(device),
        torch.tensor(X_test, dtype=torch.float32).to(device),
        torch.tensor(Y_test, dtype=torch.float32).to(device),
        torch.tensor(dev_test, dtype=torch.float32).to(device)
    )



def balance_joint_bins(df, columns_to_balance, bins=5, min_count=200, max_count=800, random_state=42):
    """
    Balance a DataFrame by binning multiple columns into joint bins and oversampling/undersampling
    each joint bin group to ensure a minimum and maximum count of samples per group.

    This function creates quantile-based bins for each specified column, combines these bins into
    joint bin labels, and then resamples each joint bin group to have at least min_count samples
    (with replacement) and at most max_count samples (without replacement). This helps in creating
    a more balanced dataset across multiple feature dimensions.

    Args:
        df (pd.DataFrame): Input DataFrame to balance.
        columns_to_balance (list): List of column names to bin and balance jointly.
        bins (int): Number of quantile bins to create for each column (default 5).
        min_count (int): Minimum number of samples per joint bin group after balancing (default 200).
        max_count (int): Maximum number of samples per joint bin group after balancing (default 800).
        random_state (int): Random seed for reproducibility (default 42).

    Returns:
        pd.DataFrame: Balanced DataFrame with resampled joint bin groups.
    """
    df = df.copy()
    for col in columns_to_balance:
        df[col + '_bin'] = pd.qcut(df[col], q=bins, labels=False, duplicates='drop')

    joint_bin_cols = [col + '_bin' for col in columns_to_balance]
    df['joint_bin'] = df[joint_bin_cols].astype(str).agg('_'.join, axis=1)

    def sample_group(g):
        n = len(g)
        if n < min_count:
            return g.sample(n=min_count, replace=True, random_state=random_state)
        elif n > max_count:
            return g.sample(n=max_count, random_state=random_state)
        else:
            return g

    df_balanced = df.groupby('joint_bin').apply(sample_group, include_groups=False).reset_index(drop=True)
    return df_balanced
