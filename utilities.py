# utilities 
import os 
import re 
import calendar
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def get_mean_std_dict(mean_std_csv):
    from config import L30_bands_dict, S30_bands_dict
    mean_std_dict = {}
    mean_std_df = pd.read_csv(mean_std_csv)
    mean_dict = dict(zip(mean_std_df['band_name'], mean_std_df['mean']))
    std_dict = dict(zip(mean_std_df['band_name'], mean_std_df['std']))
    l30_band_used_names = list(L30_bands_dict.values())[:-2]  # not include sin, cos aspect
    s30_band_used_names = list(S30_bands_dict.values())[:-2]  # not include sin, cos aspect
    mean_std_dict['l30_mean'] = [mean_dict[k] for k in l30_band_used_names]  # doy + bands + dem + slope
    mean_std_dict['l30_std'] = [std_dict[k] for k in l30_band_used_names]
    mean_std_dict['s30_mean'] = [mean_dict[k] for k in s30_band_used_names]  # doy + bands + dem + slope
    mean_std_dict['s30_std'] = [std_dict[k] for k in s30_band_used_names]
    # not norm sin cos aspect, make mean std orders same as input bands
    mean_std_dict['l30_mean'] += [0, 0]
    mean_std_dict['l30_std'] += [1, 1]
    mean_std_dict['s30_mean'] += [0, 0]
    mean_std_dict['s30_std'] += [1, 1]
    return mean_std_dict

def find_file(date_str, tile_id, HLS_1year_files):
    l8_path, s2_path = None, None
    for f in HLS_1year_files:
        if f.endswith('.Fmask.tif') and f'HLS.L30.T{tile_id}.{date_str}' in f:
            l8_path = f
        if f.endswith('.Fmask.tif') and f'HLS.S30.T{tile_id}.{date_str}' in f:
            s2_path = f
    return l8_path, s2_path


## get file names one year After the date time_str
def get_files_1year(tile, time_str, file_list):
    # Convert the provided day_of_year (format: yyyyddd) to a datetime object

    defined_date = datetime.strptime(time_str, "%Y%j")
    end_date = defined_date + timedelta(days=365)  # One year After

    # Build a regex to extract the date from filenames matching the given tile.
    # e.g., for tile T15TUE, a file name may contain "HLS.S30.T15TUE.2023244T170859..."
    pattern = re.compile(r"HLS\.[SL]30\." + re.escape(tile) + r"\.(\d{7})T\d{6}")

    selected_files = []
    seen_basenames = set()

    for fname in file_list:
        if tile in fname:
            m = pattern.search(fname)
            if m:
                # print
                file_date_str = m.group(1)  # e.g., "2023244"
                file_date = datetime.strptime(file_date_str, "%Y%j")
                if defined_date <= file_date < end_date:
                    base_name = os.path.basename(fname)
                    if base_name not in seen_basenames:
                        seen_basenames.add(base_name)
                        selected_files.append(fname)

    selected_files.sort()
    return selected_files, len(selected_files)


## get file names Between start date and end date
def get_files_start_end(tile, start_date, end_date, file_list):
    # Convert the provided day_of_year (format: yyyyddd) to a datetime object

    defined_date = datetime.strptime(start_date, "%Y%j")
    end_date = datetime.strptime(end_date, "%Y%j")

    # Build a regex to extract the date from filenames matching the given tile.
    # e.g., for tile T15TUE, a file name may contain "HLS.S30.T15TUE.2023244T170859..."
    pattern = re.compile(r"HLS\.[SL]30\." + re.escape(tile) + r"\.(\d{7})T\d{6}")

    selected_files = []
    seen_basenames = set()

    for fname in file_list:
        if tile in fname:
            m = pattern.search(fname)
            if m:
                # print
                file_date_str = m.group(1)  # e.g., "2023244"
                file_date = datetime.strptime(file_date_str, "%Y%j")
                if defined_date <= file_date < end_date:
                    base_name = os.path.basename(fname)
                    if base_name not in seen_basenames:
                        seen_basenames.add(base_name)
                        selected_files.append(fname)

    selected_files.sort()
    return selected_files, len(selected_files)



def days_in_year(year):
    return 366 if calendar.isleap(year) else 365


def get_all_dates(HLS_1year_files):
    pattern = re.compile(r"HLS\.[SL]30\.[^\.]+\.(\d{7})T\d{6}")
    date_list = []
    for fname in HLS_1year_files:
        m = pattern.search(fname)
        if m:
            date_str = m.group(1)  # e.g., "2023244"
            date_list.append(date_str)
    return date_list


import numpy as np
from scipy.ndimage import distance_transform_edt

def fill_nodata_nearest_per_band(arr, nodata=-9999.0):
    """
    arr: np.ndarray, shape (C, H, W)
    nodata: -9999.
    return: np.ndarray, shape (C, H, W)
    """
    if arr.ndim != 3: # dem/slope/aspect
        raise ValueError("arr must have shape (C, H, W).")

    out = arr.copy()
    C, H, W = out.shape

    for c in range(C):
        band = out[c]
        mask = (band == nodata)

        if not np.any(mask):
            continue

        # if all are filled values, error
        if np.all(mask):
            raise ValueError(f"Band {c} is all nodata; cannot fill by nearest.")

        # distance_transform_edt: 对“mask为True的位置”返回到最近False的位置的索引
        # 这里我们需要的是：mask True 为需要填充；False 为有效值
        _, (iy, ix) = distance_transform_edt(mask, return_indices=True)
        # 把每个无效像素替换成它最近的有效像素值
        band[mask] = band[iy[mask], ix[mask]]
        out[c] = band

    return out


def mask_obs_after_crop_damage(testxi, firstDate, periods, N_after=9999):
    '''Find the N_after HLS good observation after the crop damage event date, and then mask all observations after this time.'''
    from config import FILL
    event_doy = (pd.to_datetime(firstDate).dayofyear - 1) / 366.0
    test_output = testxi.copy()
    if N_after is None:
        N_after = 9999
    for i in range(testxi.shape[0]):
        # Landsat
        l_doys = testxi[i, :periods, 0]
        l_valid = testxi[i, :periods, 1] != FILL
        l_mask = (l_doys >= event_doy) & l_valid

        # Sentinel-2
        s_doys = testxi[i, periods:, 0]
        s_valid = testxi[i, periods:, 1] != FILL
        s_mask = (s_doys >= event_doy) & s_valid

        # combine by actual date index
        x_mask = l_mask | s_mask
        one_index = np.where(x_mask)[0]

        if one_index.size == 0:
            continue

        current_index = one_index[min(N_after, one_index.size - 1)]

        # mask Landsat from current_index onward
        test_output[i, current_index:periods, 1:] = FILL

        # mask Sentinel-2 from current_index onward
        test_output[i, periods + current_index:, 1:] = FILL

    return test_output


from datetime import datetime
def parse_year_doy(date_str, name="date"):
    """
    Parse date string in YEAR+DOY format, e.g., 2023041.
    """
    if not isinstance(date_str, str):
        raise ValueError(f"{name} must be a string, got {type(date_str).__name__}.")

    if len(date_str) != 7 or not date_str.isdigit():
        raise ValueError(
            f"{name} must be in YEAR+DOY format with 7 digits, e.g., 2023041."
        )

    year = int(date_str[:4])
    doy = int(date_str[4:])

    if doy < 1 or doy > 366:
        raise ValueError(
            f"{name} has invalid DOY: {doy}. DOY must be between 001 and 366."
        )

    try:
        date = datetime.strptime(date_str, "%Y%j").date()
    except ValueError:
        raise ValueError(
            f"{name} is not a valid YEAR+DOY date: {date_str}. "
            f"Please check whether the year is a leap year."
        )

    if date.year != year:
        raise ValueError(
            f"{name} is not a valid YEAR+DOY date: {date_str}."
        )

    return date


def validate_date_range(start_date, end_date, max_days=366):
    """
    Validate date range in [start_date, end_date) format.

    Parameters
    ----------
    start_date : str
        Start date in YEAR+DOY format, e.g., 2023041. Included.
    end_date : str
        End date in YEAR+DOY format, e.g., 2024041. Excluded.
    max_days : int
        Maximum allowed number of days in the interval [start_date, end_date).

    Returns
    -------
    start_dt, end_dt, n_days
    """
    start_dt = parse_year_doy(start_date, name="start_date")
    end_dt = parse_year_doy(end_date, name="end_date")

    n_days = (end_dt - start_dt).days

    if n_days <= 0:
        raise ValueError(
            f"end_date must be later than start_date for a [start_date, end_date) interval. "
            f"Got start_date={start_date}, end_date={end_date}."
        )

    if n_days > max_days:
        raise ValueError(
            f"The date interval [start_date, end_date) must not exceed {max_days} days. "
            f"Got {n_days} days from {start_date} to {end_date}."
        )

    return start_dt, end_dt, n_days

