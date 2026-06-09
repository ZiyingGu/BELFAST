# import necessary packages
import io
import os
import re
import contextlib
from tqdm import tqdm
from pa.lib import limbdark as ld
from pa.lib import util as ut
from glob import glob


def extract_MH(filename):
    """
    Extract metallicity [MH] from a filename based on a specific naming pattern.

    Args:
        filename (str): Name of the file containing metallicity info in the format 'imXXk2.pck' or 'ipXXk2.pck'.

    Returns:
        float: Metallicity value (negative for 'im', positive for 'ip').
    """
    pattern = re.compile(r'(i[m|p])(\d+)(k2\.pck)')
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



def make_limb_darkening_files(MH_min, MH_max, limb_darkening_MH_dic, filters, a = None, bounds = None, save = None, print_flag=False, use_tqdm=True):
    """
    Generate limb darkening files for a given metallicity range and photometric filters.

    Args:
        MH_min (float): Minimum metallicity of limb darkening files.
        MH_max (float): Maximum metallicity of limb darkening files.
        limb_darkening_MH_dic (dict): Dictionary mapping limb darkening filenames to their metallicity.
        filters (str): Glob pattern for filter files.
        a (list, optional): Extinction values for filtering (default: [0]).
        bounds (list, optional): Fit bounds for limb darkening (default: [0.1, 0.4]).
        save (str, optional): Path to save fitted limb darkening results.
        print_flag (bool, optional): Whether to print intermediate outputs (default: False).

    Returns:
        dict: Dictionary of metallicity to LimbDark objects.
    """
    if a is None:
        a = [0]
    if bounds is None:
        bounds = [0.1, 0.4]

    MH_upper_bound = min((value for value in limb_darkening_MH_dic.values() if value > MH_max), default=None)
    MH_lower_bound = max((value for value in limb_darkening_MH_dic.values() if value < MH_min), default=None)

    try:
        required_limb_darkening_dict = {value : key for key, value in limb_darkening_MH_dic.items() if MH_lower_bound <= value <= MH_upper_bound}
    except:
        required_limb_darkening_dict = {value : key for key, value in limb_darkening_MH_dic.items()}

    # print(required_limb_darkening_dict.keys())
    made_limb_darkening_dict = {}

    filtfiles = glob(filters)
    filtfiles.sort()

    if use_tqdm == True:
        iterable = tqdm(required_limb_darkening_dict.keys())
    else:
        iterable = required_limb_darkening_dict.keys()

    for MH in iterable:
        if print_flag == False:
            with contextlib.redirect_stdout(io.StringIO()):
                ldfile = required_limb_darkening_dict[MH]
                print(ldfile)
                lam, T, g = ld.getlamTg(ldfile)
                I = ld.getI(ldfile, lam, T, g)
                l = ld.LimbDark(lam, T, g, MH)

                ut.setlam( lam ) # set the wavelength array for filtering
                I = l.filter(I, filtfiles, a_v=a) # filter and redden the object, return band intensities

                l.fit(I, bounds, save)

                made_limb_darkening_dict[MH] = l
        else:
            ldfile = required_limb_darkening_dict[MH]
            print(ldfile)
            lam, T, g = ld.getlamTg(ldfile)
            I = ld.getI(ldfile, lam, T, g)
            l = ld.LimbDark(lam, T, g, MH)

            ut.setlam(lam)  # set the wavelength array for filtering
            I = l.filter(I, filtfiles, a_v=a)  # filter and redden the object, return band intensities

            l.fit(I, bounds, save)

            made_limb_darkening_dict[MH] = l

    return made_limb_darkening_dict


