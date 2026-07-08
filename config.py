# config

TASK = "CROP_MAPPING" # options: "GAP_FILL", "SOIL_MOISTURE", "FUEL_MOISTURE", "CROP_MAPPING", "CROP_DAMAGE"
INPUT_BANDS = "THERMAL_DEM" # options: "DEFAULT", "THERMAL", "THERMAL_DEM"

# ========================= TASK and Input bands config =================================

TASK_CONFIGS = {
    "GAP_FILL": {
        "Y_SCALE": 1,
        "Y_OFFSET": 0,
    },
    "SOIL_MOISTURE": {
        "Y_SCALE": 1,
        "Y_OFFSET": 0,
    },
    "FUEL_MOISTURE": {
        "Y_SCALE": 100.000,
        "Y_OFFSET": 106.102,
    },
    "CROP_MAPPING": {
        "Y_SCALE": 1,
        "Y_OFFSET": 1,  # change from 0-12 to 1-13, same as https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification
    },
    "CROP_DAMAGE": {
        "Y_SCALE": 1,
        "Y_OFFSET": 0,
    }
}

INPUT_BAND_CONFIGS = {
    "DEFAULT": {
        "L8_bands_n": 8,
        "S2_bands_n": 12, # ================ L8_bands_n and S2_bands_n are used for data input and data processing,
        "L8_model_bands_N": 8,
        "S2_model_bands_N": 12, # ================ L8_model_bands_N and S2_model_bands_N are used for model definition.
    },
    "THERMAL": {
        "L8_bands_n": 10,
        "S2_bands_n": 12,
        "L8_model_bands_N": 10,
        "S2_model_bands_N": 12,
    },
    "THERMAL_DEM": {
        "L8_bands_n": 14,
        "S2_bands_n": 16,
        "L8_model_bands_N": 10,
        "S2_model_bands_N": 12,
    },
}

task_cfg = TASK_CONFIGS[TASK]
input_cfg = INPUT_BAND_CONFIGS[INPUT_BANDS]
Y_SCALE = task_cfg["Y_SCALE"]
Y_OFFSET = task_cfg["Y_OFFSET"]


L8_bands_n = input_cfg["L8_bands_n"]
S2_bands_n = input_cfg["S2_bands_n"]
L8_model_bands_N = input_cfg["L8_model_bands_N"]
S2_model_bands_N = input_cfg["S2_model_bands_N"]

MODEL_PATHS = {
    #### model path dict: (TASK, INPUT_BANDS): Model path ####
    ("GAP_FILL", "DEFAULT"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_32.h5",
    ("GAP_FILL", "THERMAL"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_33.h5",
    ("GAP_FILL", "THERMAL_DEM"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/best_model_hls2026.layer4.METHOD2.BATCH256.LR0.0001.EPOCH40.L20.1.GAPS0.5use_bt_dem.v7_35.h5",
    ("FUEL_MOISTURE", "THERMAL_DEM"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/MC_v6_4.SMmodel.B0004.r0.00001.e10.L0.10000.i5.h5",
    ("SOIL_MOISTURE", "THERMAL_DEM"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/v4_5.SMmodel.B0256.r0.00001.e30.L0.10000.U064.i4.h5",
    ("CROP_MAPPING", "THERMAL_DEM"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/v3_9.layer4.METHOD2.BATCH128.LR0.0001.EPOCH20.L20.1.FT1.use_bt_dem.i0.model.h5",
    ("CROP_DAMAGE", "THERMAL"): "/mmfs1/scratch/jacks.local/junjie.li/Foundation model/Model weights/v5_0.layer4.METHOD2.BATCH2048.LR1e-06.EPOCH30.L20.1.FT1.i0.model 1.h5"
}

hls_transformer_model_path = MODEL_PATHS.get((TASK, INPUT_BANDS))

# ========================= other parameters =================================
# training
MAX_LANDSAT   = 176
MAX_SENTINEL2 = 176
if TASK == "CROP_DAMAGE":
    MAX_LANDSAT = 68
    MAX_SENTINEL2 = 68

FILL = -9999.
BANDS_N = 12      # default output bands for GAP_FILL

L30_bands_dict = {
    0: 'l30_doy', 1: "l30_coastal", 2: "l30_blue", 3: "l30_green", 4: "l30_red", 5: "l30_nir", 6: "l30_swir1", 7: "l30_swir2", 8: "l30_bt1", 9: "l30_bt2",
    10: 'l30_dem', 11: 'l30_slope', 12: 'l30_sin_aspect', 13: 'l30_cos_aspect'
}
S30_bands_dict = {
    0: 's30_doy', 1: 's30_coastal', 2: 's30_blue', 3: 's30_green', 4: 's30_red', 5: 's30_nirA', 6: 's30_swir1', 7: 's30_swir2', 8: 's30_edge1', 9: 's30_edge2', 10: 's30_edge3', 11: 's30_nir8',
   12: 's30_dem', 13: 's30_slope', 14: 's30_sin_aspect', 15: 's30_cos_aspect'
}
