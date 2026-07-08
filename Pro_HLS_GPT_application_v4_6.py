# V4_6 add end date
# V4_5 add crop damage
# v4_4 add crop mapping
# v4_3 add soil moisture and fuel moisture
# v4_2 use new mean and model, the code can adapt to different inputs (with thermal, dem/slope/aspect)
# v4_1 Hank wants to change for multiple outputs including soil moisture, crop damage and fuel moisture

# v4, change end_date to start_date, split many subsquences when using model_hls.predict
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import HLS_process
import tensorflow as tf
import config
from config import TASK, INPUT_BANDS
from utilities import get_mean_std_dict, validate_date_range

np.set_printoptions(suppress=True)

mean_std_file = 'CONUS_scale60_all_tiles_v2_1.train.mean.02.02.2026.csv'  # use new mean, changed on Feb 20, 2026

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TF sees GPUs:", tf.config.list_physical_devices('GPU'))

if __name__ == "__main__":

    if len(sys.argv) > 1:
        #### input parameters#################################
        tile_id = sys.argv[1]  # '14TNP'
        start_date = sys.argv[2]  # year+doy, the start date of the annual input time series
        end_date = sys.argv[3]  # year+doy, the end date of the annual input time series
        hls_data_dir = sys.argv[4]  # The input HLS time series dir
        output_dir = sys.argv[5]  # output dir

    reconstructed_dates = ""
    if len(sys.argv) > 6:
        reconstructed_dates_str = sys.argv[6].strip()  # string, year+doy, e.g. '2023140', if there are multiple dates, separate them with commas.    '2023039,2023040,2023041'
        reconstructed_dates = [d.strip() for d in reconstructed_dates_str.split(",") if d.strip()]

    ######################################################
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    start_dt, end_dt, n_days = validate_date_range(start_date, end_date)
    print(f'input time series, start date: {start_dt}, end date: {end_dt}, n_days: {n_days}')
    print(f'task: {TASK}, input_bands: {INPUT_BANDS}')
    if reconstructed_dates != "" and TASK not in ["GAP_FILL", "FUEL_MOISTURE"]:
        raise ValueError(
            f"Possible error: reconstructed_dates is not empty, but TASK={TASK} is not GAP_FILL or FUEL_MOISTURE."
        )
    if reconstructed_dates == "" and TASK in ["GAP_FILL", "FUEL_MOISTURE"]:
        raise ValueError("Please provide reconstructed_dates for GAP_FILL or FUEL_MOISTURE.")

    #  Default parameters
    is_evaluation = False  # Default is False, if True, all pixels of the tile on the predicted date will be masked and not participate in model inference, is used to evaluate model reconstruction accuracy
    CHUNK_SIZE = 1220  # 1220, 915, 732
    IMG_WIDTH = IMG_HEIGHT = 3660
    BATCH_SIZE = 2048
    DEM_DIR = '/mmfs1/scratch/jacks.local/hankui.zhang/srtm/hls_tile_srtm'

    start = datetime.now()
    print_str = '\n\n\nstart time: ' + str(start)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    search_dir = Path(hls_data_dir)

    HLS_LIST = [str(p) for p in search_dir.rglob('*.Fmask.tif')]
    mean_std_dict = get_mean_std_dict(mean_std_file)

    # if reconstructed_dates=="":
    # print(f"Derive parameters for all the observed dates")

    output_landsat, output_sentinel, all_dates = HLS_process.process_by_chunks(HLS_LIST, tile_id, IMG_WIDTH, IMG_HEIGHT,
                                                                               CHUNK_SIZE, BATCH_SIZE, start_date, end_date,
                                                                               mean_std_dict, reconstructed_dates,
                                                                               is_evaluation, DEM_DIR)

    import true_color_noC
    import color_display

    if TASK == "GAP_FILL":
        for i in range(len(reconstructed_dates)):
            landsat_data = output_landsat[:, :, i, :7]  # shape: (H, W, B)
            landsat_data = landsat_data.transpose(2, 0, 1)  # (B, H, W)

            output_path = os.path.join(output_dir,
                                       f'tile_{tile_id}_date_{reconstructed_dates[i]}_{INPUT_BANDS}_landsat.tif')  # add INPUT_BANDS to file name
            if os.path.exists(output_path):
                os.remove(output_path)

            HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, landsat_data, output_path)

            # Sentinel
            sentinel_data = output_sentinel[:, :, i, :]  # shape: (H, W, B)
            sentinel_data = sentinel_data.transpose(2, 0, 1)  # (B, H, W)

            output_path = os.path.join(output_dir,
                                       f'tile_{tile_id}_date_{reconstructed_dates[i]}_{INPUT_BANDS}_sentinel.tif')
            if os.path.exists(output_path):
                os.remove(output_path)

            HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, sentinel_data, output_path)
            if i == 0:
                output_path = os.path.join(output_dir,
                                           f'tile_browse_{tile_id}_date_{reconstructed_dates[i]}_{INPUT_BANDS}_sentinel.jpg')
                true_color_noC.true_color_from_image_noc((sentinel_data[1:4, :, :] * 10000).astype(np.int16),
                                                         output_path)

    elif TASK == "SOIL_MOISTURE":
        for i in range(len(all_dates)):
            landsat_data = output_landsat[:, :, i, :]  # shape: (H, W, B)
            landsat_data = landsat_data.transpose(2, 0, 1)  # (B, H, W)

            output_path = os.path.join(output_dir, f'tile_{tile_id}_date_{all_dates[i]}_sm.tif')
            if os.path.exists(output_path):
                os.remove(output_path)

            HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, landsat_data, output_path)
            # if i<=len(reconstructed_dates)-1:
            # output_path = os.path.join(output_dir, f'tile_browse_{tile_id}_date_{reconstructed_dates[i]}_lfmc.jpg')
            # color_display.color_display_from_image(landsat_data[0,:,:], dsr_file="LFMC_legend_10interval.dsr", output_tif=output_path)
    elif TASK == "FUEL_MOISTURE":
        for i in range(len(reconstructed_dates)):
            landsat_data = output_landsat[:, :, i, :]  # shape: (H, W, B)
            landsat_data = landsat_data.transpose(2, 0, 1)  # (B, H, W)

            output_path = os.path.join(output_dir, f'tile_{tile_id}_date_{reconstructed_dates[i]}_lfmc.tif')
            if os.path.exists(output_path):
                os.remove(output_path)

            HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, landsat_data, output_path)
            if i <= len(reconstructed_dates) - 1:
                output_path = os.path.join(output_dir, f'tile_browse_{tile_id}_date_{reconstructed_dates[i]}_lfmc.jpg')
                color_display.color_display_from_image(landsat_data[0, :, :], dsr_file="LFMC_legend_10interval.dsr",
                                                       output_tif=output_path)

    elif TASK == "CROP_MAPPING":
        crop_mapping = output_landsat.transpose(2, 0, 1)  # (B, H, W)
        output_path = os.path.join(output_dir, f'tile_{tile_id}_{start_date}_{end_date}_crop_mapping.tif')
        if os.path.exists(output_path):
            os.remove(output_path)
        HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, crop_mapping, output_path)

    elif TASK == "CROP_DAMAGE":
        crop_damage = output_landsat.transpose(2, 0, 1)  # (B, H, W)
        output_path = os.path.join(output_dir, f'tile_{tile_id}_{start_date}_{end_date}_crop_damage.tif')
        if os.path.exists(output_path):
            os.remove(output_path)
        HLS_process.save_result_as_geotiff(HLS_LIST, tile_id, crop_damage, output_path)

    else:
        raise ValueError('Task not recognized.')

    end = datetime.now()
    elapsed = end - start
    print_str = '\nEnd time = ' + str(end) + 'Elapsed time = ' + str(
        elapsed) + '\n======================================'
    print(print_str)

