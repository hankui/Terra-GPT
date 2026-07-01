# HLS_process.py
import numpy as np
import tensorflow as tf
import gc 
import rasterio
from rasterio.windows import Window
import os
import config 
import utilities
import importlib
import HLS_io_chunks
import model_load
importlib.reload (utilities)
importlib.reload (config)
importlib.reload (model_load)
VALID_DATA_THRESHOLD_IN_YEAR = 4
from config import TASK, Y_SCALE, Y_OFFSET, hls_transformer_model_path
from config import BANDS_N, FILL
from config import L8_bands_n, S2_bands_n, INPUT_BANDS

if TASK=="GAP_FILL":
    n_out = BANDS_N-1
elif TASK=="SOIL_MOISTURE":
    n_out = 2
elif TASK=="FUEL_MOISTURE":
    n_out = 2
elif TASK=="CROP_MAPPING":
    n_out = 13
elif TASK=="CROP_DAMAGE":
    n_out = 1
    from config import EVENT_DATE, N_AFTER
else:
    raise ValueError(f"{TASK} not defined")

# process the entire image chunk by chunk 
# reconstructed_dates = '2023195,2023220'.split(',')
# is_evaluation=False; dem_dir=DEM_DIR
def process_by_chunks(HLS_LIST, tile_id, IMG_WIDTH, IMG_HEIGHT, CHUNK_SIZE, BATCH_SIZE, start_date, mean_std_dict, reconstructed_dates="", is_evaluation=False, dem_dir=None):
    
    HLS_1year_files, total_n = utilities.get_files_1year("T" + tile_id, str(start_date), file_list=HLS_LIST)
    # mean and std
    l30_mean = mean_std_dict['l30_mean']
    l30_std = mean_std_dict['l30_std']
    s30_mean = mean_std_dict['s30_mean']
    s30_std = mean_std_dict['s30_std']
    
    # get all dates (year+doy)
    files_dates = utilities.get_all_dates(HLS_1year_files)
    print (files_dates)
    unique_dates = list(set(files_dates))
    not_includes_dates = [x for x in reconstructed_dates if x not in unique_dates]
    all_dates = sorted(unique_dates + not_includes_dates)
    ## derive output holder 
    if reconstructed_dates=="" and TASK=="SOIL_MOISTURE": # derive for all the soil moisture dates 
        # reconstructed_dates = all_dates.copy() 
        output_landsat  = np.full((IMG_HEIGHT, IMG_WIDTH, len(all_dates)          , n_out+1), FILL, dtype=np.float32)
        output_sentinel = np.full((IMG_HEIGHT, IMG_WIDTH, len(all_dates)          , n_out+1), FILL, dtype=np.float32)
    elif TASK=="CROP_MAPPING":
        output_landsat  = np.full((IMG_HEIGHT, IMG_WIDTH, 1), FILL, dtype=np.float32)
        output_sentinel = np.full((IMG_HEIGHT, IMG_WIDTH, 1), FILL, dtype=np.float32)
    elif TASK=="CROP_DAMAGE":
        output_landsat = np.full((IMG_HEIGHT, IMG_WIDTH, 1), FILL, dtype=np.float32)
        output_sentinel = np.full((IMG_HEIGHT, IMG_WIDTH, 1), FILL, dtype=np.float32)
    else:
        output_landsat  = np.full((IMG_HEIGHT, IMG_WIDTH, len(reconstructed_dates), n_out+1), FILL, dtype=np.float32)
        output_sentinel = np.full((IMG_HEIGHT, IMG_WIDTH, len(reconstructed_dates), n_out+1), FILL, dtype=np.float32)
    
    min_year = int(all_dates[0][:4])
    row_start = 0; col_start = 0
    for row_start in range(0, IMG_HEIGHT, CHUNK_SIZE):
        for col_start in range(0, IMG_WIDTH, CHUNK_SIZE):
            print('start to process row {}, col {}'.format(row_start, col_start))
            row_end = min(row_start + CHUNK_SIZE, IMG_HEIGHT)
            col_end = min(col_start + CHUNK_SIZE, IMG_WIDTH)
            width  = col_end - col_start
            height = row_end - row_start
            patch_data, patch_qa, used_dates = process_patch_yearly(tile_id, HLS_1year_files, row_start, col_start, width, height, all_dates, reconstructed_dates, is_evaluation, dem_dir)    # shape: width, height, MAX_LANDSAT + MAX_SENTINEL2, BANDS (DOY + band) (512, 512, 352, 12)
            periods = int(patch_qa.shape[2] / 2)
            # only input valid pixels to model, num l8>N and s2>N
            # patch_qa_landsat = patch_qa[:, :, :periods]
            # patch_qa_sentinel = patch_qa[:, :, periods:]
            # valid_patch_qa = np.logical_and(patch_qa_landsat.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR, patch_qa_sentinel.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR)
            valid_patch_qa = patch_qa.sum(axis=2) > VALID_DATA_THRESHOLD_IN_YEAR # an 2D patch array 
            qa_flat = valid_patch_qa.reshape(-1) # height * width an 1D array on spaital 
            valid_indices = np.where(qa_flat)[0] # an 1D array on spaital 
            patch_data_reshaped = patch_data.reshape(-1, periods+periods, S2_bands_n)
            valid_patches = patch_data_reshaped[valid_indices]  # shape: (N_valid, 352, BANDS=12)
            # norm
            norm_data(valid_patches, l30_mean, l30_std, range(1, L8_bands_n), slice(0, periods), offset=0)
            norm_data(valid_patches, s30_mean, s30_std, range(1, S2_bands_n), slice(periods, periods + periods), offset=0)

            valid_patches = valid_patches.astype(np.float32, copy=False)
            print('finish preparing chunk data-------------------')
            
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                model_hls = model_load.load_model(hls_transformer_model_path, periods)
            
            N, T, B = valid_patches.shape # T = periods+periods
            STEP = int(1e5)
            if N < STEP:
                predictions = model_hls.predict(valid_patches, verbose=2, batch_size=BATCH_SIZE)  # N_valid * 352 * 11, crop mapping is N*13
            else:
                if TASK=="GAP_FILL":
                    predictions = np.full(shape=(N, T, n_out), fill_value=FILL, dtype=np.float32)
                elif TASK=="SOIL_MOISTURE":
                    predictions = np.full(shape=(N, T//2, n_out), fill_value=FILL, dtype=np.float32)
                elif TASK=="FUEL_MOISTURE":
                    predictions = np.full(shape=(N, T//2, n_out), fill_value=FILL, dtype=np.float32)
                elif TASK=="CROP_MAPPING":
                    predictions = np.full(shape=(N, n_out), fill_value=FILL, dtype=np.float32)
                elif TASK=="CROP_DAMAGE":
                    predictions = np.full(shape=(N, T//2, n_out), fill_value=FILL, dtype=np.float32)
                    if EVENT_DATE is not None:
                        valid_patches = utilities.mask_obs_after_crop_damage(valid_patches, EVENT_DATE, periods, N_AFTER)
                else:
                    raise ValueError(f"{TASK} not defined")
                
                for i in range(0, N, STEP):
                    start = i
                    end = min(i + STEP, N)
                    print('subsquence {} to {}'.format(start, end))
                    tempx = valid_patches[start:end]
                    tempy = model_hls.predict(tempx, batch_size=BATCH_SIZE, verbose=2)
                    predictions[start:end] = tempy
            
            ## *******************************************************************************
            ## wrap up the predicted values 
            landsat_obs  = patch_data_reshaped[valid_indices, :periods, :]  # shape: (N_valid, 176, 12)
            sentinel_obs = patch_data_reshaped[valid_indices, periods:, :]  # shape: (N_valid, 176, 12)
            ## missing indicator: 1 is missing (no observation) and 0 is not missing (with observation) and -9999 is no prediction
            landsat_missing  = landsat_obs [:, :, 1] == FILL   #  shape: (N_valid, 176)
            sentinel_missing = sentinel_obs[:, :, 1] == FILL
            predict_doys = [int(date[:4]) - min_year + (int(date[4:]) - 1) / 366.0 for date in reconstructed_dates]
            print('finish reconstruction-------------------')
            
            if TASK=="GAP_FILL":
                landsat_obs [landsat_missing , 1:BANDS_N] = predictions[:, :periods, :][landsat_missing]   # fixed bug on 2/24/2026, BANDS_N=12, avoid shape mismatch when add dem data
                sentinel_obs[sentinel_missing, 1:BANDS_N] = predictions[:, periods:, :][sentinel_missing]
                
                # Init outputs in patch size
                result_landsat  = np.full((height * width, len(predict_doys), n_out+1), FILL, dtype=np.float32)
                result_sentinel = np.full((height * width, len(predict_doys), n_out+1), FILL, dtype=np.float32)
                result_landsat [valid_indices,:,:n_out] = extract_by_doy_optimized(landsat_obs [:, :, 0], landsat_obs [:, :, 1:BANDS_N], predict_doys)   # modified on 2/24/2026
                result_sentinel[valid_indices,:,:n_out] = extract_by_doy_optimized(sentinel_obs[:, :, 0], sentinel_obs[:, :, 1:BANDS_N], predict_doys)
                result_landsat [valid_indices,:,-1    ] = extract_by_doy_optimized(landsat_obs [:, :, 0], landsat_missing [:,:,np.newaxis], predict_doys)[:,:,0]
                result_sentinel[valid_indices,:,-1    ] = extract_by_doy_optimized(sentinel_obs[:, :, 0], sentinel_missing[:,:,np.newaxis], predict_doys)[:,:,0]
                
                # Reshape to patch size and fit in the image
                output_landsat [row_start:row_end, col_start:col_end, :, :] = result_landsat .reshape(height, width, len(predict_doys), n_out+1)
                output_sentinel[row_start:row_end, col_start:col_end, :, :] = result_sentinel.reshape(height, width, len(predict_doys), n_out+1)
            elif TASK=="SOIL_MOISTURE": ## derive on the observed 
                both_missing = np.logical_and(landsat_missing,sentinel_missing) 
                result_landsat  = np.full((height * width, periods, n_out+1), FILL, dtype=np.float32)
                result_landsat [valid_indices,:, :1] =        predictions[:,:, :1]
                result_landsat [valid_indices,:,1:2] = np.exp(predictions[:,:,1:2])
                result_landsat [valid_indices,:,-1 ] = both_missing
                doy_index = np.isin(all_dates, used_dates)
                # Reshape to patch size and fit in the image
                output_landsat [row_start:row_end, col_start:col_end, doy_index, :] = result_landsat .reshape(height, width, periods, n_out+1)
                output_sentinel = output_landsat
                
            elif TASK=="FUEL_MOISTURE": ## derive for reconstruction dates only 
                both_missing = np.logical_and(landsat_missing,sentinel_missing) 
                # Init outputs in patch size
                result_landsat  = np.full((height * width, len(predict_doys), n_out+1), FILL, dtype=np.float32)
                result_landsat [valid_indices,:, :1] =        extract_by_doy_optimized(landsat_obs [:, :, 0], predictions[:,:, :1], predict_doys)*Y_SCALE+Y_OFFSET
                result_landsat [valid_indices,:,1:2] = np.exp(extract_by_doy_optimized(landsat_obs [:, :, 0], predictions[:,:,1:2], predict_doys))*Y_SCALE
                result_landsat [valid_indices,:,-1    ] = extract_by_doy_optimized(landsat_obs [:, :, 0], both_missing [:,:,np.newaxis], predict_doys)[:,:,0]
                
                # Reshape to patch size and fit in the image
                output_landsat [row_start:row_end, col_start:col_end, :, :] = result_landsat .reshape(height, width, len(predict_doys), n_out+1)
                output_sentinel = output_landsat

            elif TASK=="CROP_MAPPING": # output one map
                y_pred = np.argmax(predictions, axis=1) #
                result_landsat = np.full((height * width), FILL, dtype=np.float32)
                result_landsat[valid_indices] = y_pred*Y_SCALE+Y_OFFSET
                output_landsat[row_start:row_end, col_start:col_end, 0] = result_landsat.reshape(height, width)
                output_sentinel = output_landsat

            elif TASK=="CROP_DAMAGE":
                l_mask = valid_patches[:, :periods, 1] != FILL
                s_mask = valid_patches[:, periods:, 1] != FILL
                x_mask = np.logical_or(l_mask, s_mask)
                # index of last True in each row
                last_true_idx = x_mask.shape[1] - 1 - np.argmax(x_mask[:, ::-1], axis=1)
                # get prediction at the last True position
                pred_last = predictions[np.arange(predictions.shape[0]), last_true_idx, 0]
                y_pred = tf.nn.sigmoid(pred_last)
                y_pred = (y_pred > 0.5).numpy().astype(np.int64)
                result_landsat = np.full((height * width), FILL, dtype=np.float32)
                result_landsat[valid_indices] = y_pred
                output_landsat[row_start:row_end, col_start:col_end, 0] = result_landsat.reshape(height, width)
                output_sentinel = output_landsat

            else:
                raise ValueError(f"{TASK} not defined")

            tf.keras.backend.clear_session()
            gc.collect()
            # break 
        # break
    
    return output_landsat, output_sentinel, all_dates



def search_doy(target_doy, doy_list):
    find = None
    for i in range(len(doy_list)):
        if abs(target_doy - doy_list[i]) < 1e-4:
            find = i
            break
    return find


# def extract_by_doy_optimized(data, target_doys):
    # N, T, B = data.shape
    # M = len(target_doys)
    # doy_sequence = data[0, :, 0]
    # target_indices = []
    # for doy in target_doys:
        # idx = search_doy(doy, doy_sequence.flatten().tolist())
        # if idx is not None:
            # target_indices.append(idx)
        # else:
            # target_indices.append(-1)
    # result = np.full((N, M, B-1), -9999, dtype=np.float32)
    # for m, idx in enumerate(target_indices):
        # if idx >= 0:
            # result[:, m, :] = data[:, idx, 1:]
    # return result

# doy_sequence, data, target_doys = landsat_obs [0, :, 0], landsat_obs [0, :, 1:], predict_doys
# def extract_doy_index (doy_sequence, target_doys):

# doy_sequence, data, target_doys = landsat_obs [0, :, 0], landsat_obs [0, :, 1:], predict_doys
def extract_by_doy_optimized(doy_sequence, data, target_doys):
    N, T, B = data.shape
    M = len(target_doys)
    target_indices = []
    for doy in target_doys:
        idx = search_doy(doy, doy_sequence.flatten().tolist())
        if idx is not None:
            target_indices.append(idx)
        else:
            target_indices.append(-1)
    
    result = np.full((N, M, B), -9999, dtype=np.float32)
    for m, idx in enumerate(target_indices):
        if idx >= 0:
            result[:, m, :] = data[:, idx, :]
    return result


## read yearly data, note only those patches with valid data are kept
# tile_id, HLS_1year_files, row_start, col_start, width, height, all_dates, reconstructed_dates, is_evaluation, dem_dir)    # shape: width, height, MAX_LANDSAT + MAX_SENTINEL2, BANDS (DOY + band) (512, 512, 352, 12)
# tile_id, HLS_1year_files, row_top, col_left, width, height, all_dates, reconstructed_dates, is_evaluation, dem_dir):
# row_top, col_left = row_start, col_start
def process_patch_yearly(tile_id, HLS_1year_files, row_top, col_left, width, height, all_dates, reconstructed_dates, is_evaluation, dem_dir):
    skipped_days = reconstructed_dates if is_evaluation else []
    periods = len(all_dates)
    patch_data = np.full([height, width, (periods + periods), S2_bands_n], fill_value=FILL, dtype=np.float32)
    patch_qa = np.full([height, width, (periods + periods)], fill_value=False, dtype=bool)
    min_year = int(all_dates[0][:4])
    delete_time_index = []
    
    dem_array=None
    if INPUT_BANDS == 'THERMAL_DEM':
        dempath = os.path.join(dem_dir, f"projected_srtm_{tile_id}.v1_2.tif")
        window = Window(row_off=row_top, col_off=col_left, height=height, width=width)
        demdata = rasterio.open(dempath).read(window=window) # 3*1220*1220
        demdata = utilities.fill_nodata_nearest_per_band(demdata)   # slope and aspect may have -9999 on borders
        aspect = demdata[-1,:,:]
        rad = np.deg2rad(aspect)     # convert degrees to radian before calculate sin and cosine aspect, modified on 2/24/2026
        sin_aspect = np.sin(rad)
        cos_aspect = np.cos(rad)
        dem_array = np.concatenate([demdata[:-1], sin_aspect[np.newaxis, :, :], cos_aspect[np.newaxis, :, :]], axis=0)
    
    i = 0
    used_dates = []
    for i in range(periods):
        date_str = all_dates[i]
        l8_path, s2_path = utilities.find_file(date_str, tile_id, HLS_1year_files)
        fyear = int(date_str[:4])
        fdoy = int(date_str[4:])
        patch_data[:, :, i, 0] = int(fyear - min_year) + int(fdoy - 1) / 366.0  ## for DOY add to all the pixels
        patch_data[:, :, i + periods, 0] = int(fyear - min_year) + int(fdoy - 1) / 366.0  ## for DOY add to all the pixels
        if date_str in skipped_days:
            continue
        
        l8_valid_sum = 0
        if l8_path is not None:
            HLSi = HLS_io_chunks.HLS_tile(l8_path.rstrip("\n"), is_L8=True)
            HLSi.load_data(row_top, col_left, width=width, height=height)
            l8_valid_sum = HLSi.is_valid.sum()
            patch_qa[HLSi.is_valid, i] = True
            if INPUT_BANDS == 'THERMAL_DEM':
                HLSi.reflectance_30m = np.concatenate([HLSi.reflectance_30m, dem_array], axis=0)
            
            for bi in range(1, L8_bands_n):
                patch_data[HLSi.is_valid, i, bi] = HLSi.reflectance_30m[bi - 1, HLSi.is_valid]
            
        if INPUT_BANDS == 'THERMAL_DEM':
            for bi in range(L8_bands_n-4, L8_bands_n):
                patch_data[:, :, i, bi] = dem_array[bi - (L8_bands_n-4), :, :]
                
        s2_valid_sum = 0
        if s2_path is not None:
            HLSi = HLS_io_chunks.HLS_tile(s2_path.rstrip("\n"), is_L8=False)
            HLSi.load_data(row_top, col_left, width=width, height=height)
            s2_valid_sum = HLSi.is_valid.sum()
            patch_qa[HLSi.is_valid, i + periods] = True
            if INPUT_BANDS == 'THERMAL_DEM':
                HLSi.reflectance_30m = np.concatenate([HLSi.reflectance_30m, dem_array], axis=0)
            for bi in range(1, S2_bands_n):
                patch_data[HLSi.is_valid, i + periods, bi] = HLSi.reflectance_30m[bi - 1, HLSi.is_valid]
        
        if INPUT_BANDS == 'THERMAL_DEM':
            for bi in range(S2_bands_n-4, S2_bands_n):
                patch_data[:, :, i + periods, bi] = dem_array[bi - (S2_bands_n-4), :, :]
        
        
        total_valid_sum = l8_valid_sum + s2_valid_sum
        # if l8, s2 do not contain valid pixels and current doy is not in reconstructed_dates, this time should be removed from patch_data to reduce the periods !!
        if total_valid_sum == 0 and date_str not in reconstructed_dates:
            delete_time_index.extend([i, i + periods])
        else: 
            used_dates.append (date_str)
    
    if len(delete_time_index) > 0:
        del_idx = np.unique(np.asarray(delete_time_index, dtype=int))
        H, W, T, C = patch_data.shape
        keep_mask = np.ones(T, dtype=bool)
        keep_mask[del_idx] = False
        patch_data_new = patch_data[:, :, keep_mask, :]
        patch_qa_new = patch_qa[:, :, keep_mask]
        return patch_data_new, patch_qa_new, used_dates
    
    return patch_data, patch_qa, used_dates

## 
def norm_data(data, x_mean, x_std, indices, slice_range, offset):
    for index in indices:
        valid_indices = data[:, slice_range, index] != -9999.0
        data[:, slice_range, index][valid_indices] -= x_mean[index + offset]
        data[:, slice_range, index][valid_indices] /= x_std[index + offset]


def get_tile_example_img(HLS_LIST, tile):
    find_file = False
    for line in HLS_LIST:
        if tile in line:
            find_file = True
            return line.strip()
    if not find_file:
        raise FileNotFoundError('No tile example image in ' + HLS_LIST)



def save_result_as_geotiff(HLS_LIST, tile, data, output_path):
    tile_example = get_tile_example_img(HLS_LIST, tile)
    with rasterio.open(tile_example) as src:
        profile = src.profile.copy()
        profile.update({
            'count': data.shape[0],
            'dtype': 'float32',
            'nodata': -9999,
            'compress': 'deflate'
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)
    print('saved to', output_path)


