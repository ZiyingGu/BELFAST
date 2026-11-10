import io
import os
import re
import contextlib
from tqdm import tqdm
from glob import glob
from pa.lib import util as ut
from pa.lib import limbdark as ld


def extract_MH(filename):
    """
    Extract metallicity [M/H] from a filename.

    Args:
        filename (str): File path or name containing a pattern such as
            'imXXk2.pck' or 'ipXXk2.pck'.

    Returns:
        float or None: Metallicity value, negative for 'im' and positive
            for 'ip'. Returns None if no matching pattern is found.
        """
    pattern = re.compile(r'(i[mp])(\d+)(k2\.pck)')
    file_basename = os.path.basename(filename)
    match = pattern.search(file_basename)
    if match:
        prefix = match.group(1)
        value = int(match.group(2)) / 10

        if prefix == 'im':
            sign = '-'
        elif prefix == 'ip':
            sign = '+'

        # return f"MH={sign}{value:.1f}"
        return float(f'{sign}{value:.1f}')


def make_limb_darkening_files(
    limb_darkening_MH_dic, filters, MH_min=None, MH_max=None, a=None, bounds=None,
    save=None, print_flag=False, show_progress=True
):
    """Generate limb-darkening models for a metallicity range and photometric filters.

    Args:
        limb_darkening_MH_dic (dict): Mapping of limb-darkening file paths to metallicities.
        filters (str): Glob pattern for photometric filter files.
        MH_min (float, optional): Minimum metallicity. Defaults to None.
        MH_max (float, optional): Maximum metallicity. Defaults to None.
        a (list, optional): Extinction values passed to the filter function. Defaults to [0].
        bounds (list, optional): Fit bounds. Defaults to [0.1, 0.4].
        save (str, optional): Path for saving fitted results.
        print_flag (bool): Whether to display intermediate output.
        show_progress (bool): Whether to display a progress bar.

    Returns:
        dict: Mapping of metallicities to fitted LimbDark objects.
    """
    if a is None:
        a = [0]
    if bounds is None:
        bounds = [0.1, 0.4]

    if MH_min is None and MH_max is None:
        required_limb_darkening_dict = {value: key for key, value in limb_darkening_MH_dic.items()}
    else:
        MH_lower_bound = max(
            (value for value in limb_darkening_MH_dic.values() if MH_min is not None and value < MH_min),
            default=None
        ) if MH_min is not None else None
        MH_upper_bound = min(
            (value for value in limb_darkening_MH_dic.values() if MH_max is not None and value > MH_max),
            default=None
        ) if MH_max is not None else None

        required_limb_darkening_dict = {
            value: key for key, value in limb_darkening_MH_dic.items()
            if (MH_lower_bound is None or value >= MH_lower_bound)
            and (MH_upper_bound is None or value <= MH_upper_bound)
        }

    made_limb_darkening_dict = {}
    filtfiles = sorted(glob(filters))
    iterable = tqdm(required_limb_darkening_dict) if show_progress else required_limb_darkening_dict

    for MH in iterable:
        ldfile = required_limb_darkening_dict[MH]

        if print_flag:
            lam, T, g = ld.getlamTg(ldfile)
            I = ld.getI(ldfile, lam, T, g)
            l = ld.LimbDark(lam, T, g, MH)
            ut.setlam(lam)
            I = l.filter(I, filtfiles, a_v=a)
            l.fit(I, bounds, save)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                lam, T, g = ld.getlamTg(ldfile)
                I = ld.getI(ldfile, lam, T, g)
                l = ld.LimbDark(lam, T, g, MH)
                ut.setlam(lam)
                I = l.filter(I, filtfiles, a_v=a)
                l.fit(I, bounds, save)

        made_limb_darkening_dict[MH] = l

    return made_limb_darkening_dict