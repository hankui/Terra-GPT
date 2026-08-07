# v4_7 format input parameters
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
import argparse
import HLS_process
import tensorflow as tf
import config
from config import TASK, INPUT_BANDS
from utilities import get_mean_std_dict, validate_date_range

np.set_printoptions(suppress=True)

mean_std_file = 'CONUS_scale60_all_tiles_v2_1.train.mean.02.02.2026.csv'  # use new mean, changed on Feb 20, 2026

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TF sees GPUs:", tf.config.list_physical_devices('GPU'))


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.strip().lower()

    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Terra-GPT"
    )

    parser.add_argument("--tile_id", type=str, required=True, help="HLS tile ID, e.g., 14TNP")

    parser.add_argument("--start_date", type=str, required=True, help="Start date in YEAR+DOY format, e.g., 2023001")

    parser.add_argument("--end_date", type=str, required=True, help="End date in YEAR+DOY format, e.g., 2023365")

    parser.add_argument("--hls_data_dir", type=str, required=True, help="Directory containing the input HLS time series")

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    parser.add_argument("--dem_dir", type=str, required=True, help="input dem directory")


    parser.add_argument("--chunk_size", type=int, default=1220, help="Processing chunk size. Default: 1220")

    parser.add_argument("--img_width", type=int, default=3660, help="Image width in pixels. Default: 3660")

    parser.add_argument("--img_height", type=int, default=3660, help="Image height in pixels. Default: 3660")

    parser.add_argument("--batch_size", type=int, default=2048, help="Inference batch size. Default: 2048")

    parser.add_argument("--reconstructed_dates", type=str, default="",
                        help=(
                            "Dates to reconstruct in YEAR+DOY format. Use commas to separate multiple dates, e.g., '2023039,2023040,2023041'. Default: empty string."
                        ))
    parser.add_argument(
        "--is_evaluation", type=str_to_bool, default=False, help=(
            "Whether to run evaluation mode when reconstructing HLS time series. In evaluation mode, all observations on the target date are masked before inference. Default: False."
        ))

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    tile_id = args.tile_id
    start_date = args.start_date
    end_date = args.end_date
    hls_data_dir = args.hls_data_dir
    output_dir = args.output_dir
    if args.reconstructed_dates.strip():
        reconstructed_dates = [
            date.strip()
            for date in args.reconstructed_dates.split(",")
            if date.strip()
        ]
    else:
        reconstructed_dates = ""
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

    is_evaluation = args.is_evaluation
    CHUNK_SIZE = args.chunk_size
    IMG_WIDTH = args.img_width
    IMG_HEIGHT = args.img_height
    BATCH_SIZE = args.batch_size
    DEM_DIR = args.dem_dir

    print("tile_id:", tile_id)
    print("start_date:", start_date)
    print("end_date:", end_date)
    print("hls_data_dir:", hls_data_dir)
    print("output_dir:", output_dir)
    print("reconstructed_dates:", reconstructed_dates)
    print("is_evaluation:", is_evaluation)
    print("CHUNK_SIZE:", CHUNK_SIZE)
    print("IMG_WIDTH:", IMG_WIDTH)
    print("IMG_HEIGHT:", IMG_HEIGHT)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("DEM_DIR:", DEM_DIR)

    start = datetime.now()
    print_str = '\n\n\nstart time: ' + str(start)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    search_dir = Path(hls_data_dir)

    HLS_LIST = [str(p) for p in search_dir.rglob('*.Fmask.tif')]
    if len(HLS_LIST) == 0:
        raise ValueError('No HLS time series found in ' + hls_data_dir)
    mean_std_dict = get_mean_std_dict(mean_std_file)

    output_landsat, output_sentinel, all_dates = HLS_process.process_by_chunks(HLS_LIST, tile_id, IMG_WIDTH, IMG_HEIGHT,
                                                                               CHUNK_SIZE, BATCH_SIZE, start_date, end_date,
                                                                               mean_std_dict, reconstructed_dates,
                                                                               is_evaluation, DEM_DIR)

    import true_color_noC
    import color_display

    if TASK == "GAP_FILL":
        for i in range(len(reconstructed_dates)):
            observation_indicator = output_landsat[:, :, i, -1]
            landsat_data = np.concatenate((output_landsat[:, :, i, :7], observation_indicator[:,:,np.newaxis]), axis=-1) # shape: (H, W, B)
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

