# 1 Terra-GPT Overview

Terra-GPT is based on HLS-GPT and is a generative pretrained Transformer for:

- Harmonized Landsat and Sentinel-2 reflectance reconstruction
- Daily live fuel moisture content (LFMC) retrieval
- Near-real-time soil moisture retrieval
- Within-season crop mapping across the United States
- Near-real-time crop damage mapping

All tasks are designed for HLS tiles over CONUS.

## 1.1 Updates

✅ 2026-08-06: We prepared a small demo data for quick start. 

## 1.2 Notes Before Running

- Download the pretrained models from https://zenodo.org/records/21810783 and update the `MODEL_PATHS` variable in `config.py` to point to the local model files.
- Select the task by setting `TASK` in `config.py`.
- Select the input band configuration by setting `INPUT_BANDS` in `config.py`.
- If `INPUT_BANDS="THERMAL_DEM"`, Terra-GPT requires a 30 m DEM GeoTIFF containing DEM, slope, and aspect. HLS-tiled DEM files are available at https://zenodo.org/records/21117230. Pass the DEM directory to `Pro_HLS_GPT_application_v4_7.py` using `--dem_dir`.

| TASK | INPUT_BANDS | Model name |
| --- | --- | --- |
| GAP_FILL | DEFAULT | `best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_32.h5` |
| GAP_FILL | THERMAL | `best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_33.h5` |
| GAP_FILL | THERMAL_DEM | `best_model_hls2026.layer4.METHOD2.BATCH256.LR0.0001.EPOCH40.L20.1.GAPS0.5use_bt_dem.v7_35.h5` |
| FUEL_MOISTURE | THERMAL_DEM | `MC_v6_4.SMmodel.B0004.r0.00001.e10.L0.10000.i5.h5` |
| SOIL_MOISTURE | THERMAL_DEM | `v4_5.SMmodel.B0256.r0.00001.e30.L0.10000.U064.i4.h5` |
| CROP_MAPPING | THERMAL_DEM | `v3_9.layer4.METHOD2.BATCH128.LR0.0001.EPOCH20.L20.1.FT1.use_bt_dem.i0.model.h5` |
| CROP_DAMAGE | THERMAL | `v5_0.layer4.METHOD2.BATCH2048.LR1e-06.EPOCH30.L20.1.FT1.i0.model 1.h5` |

The input band configurations are:

- `DEFAULT`: HLS reflectance bands only; no Landsat thermal bands and no topographic variables.
- `THERMAL`: HLS reflectance bands plus Landsat thermal bands; no topographic variables.
- `THERMAL_DEM`: HLS reflectance bands, Landsat thermal bands, DEM, slope, and aspect.

For crop damage, the model was fine-tuned using growing-season time series from March 1 to dates within the growing season up to October 1.

# 2. System requirements

## 2.1 All software dependencies and operating systems (including version numbers)

- Python 3.9–3.11 is recommended
- Operating systems: Linux and Windows 10 or later.
- TensorFlow library: `tensorflow>=2.10,<2.16`.
- Required Python packages: `numpy`, `pandas`, `scipy`, and `rasterio`.

## 1.2 Versions the software has been tested on

| Test environments | OS | Python version | TensorFlow version | Others |
| --- | --- | --- | --- | --- |
| 1 | Linux (Red Hat Enterprise Linux 9.7) | 3.11.0 | 2.15.0 | numpy 1.26.3,<br>pandas 2.1.4,<br>scipy 1.15.2,<br>rasterio 1.4.1 |
| 2 | Linux (Rocky Linux 9.2) | 3.12 | 2.15.1 | numpy 1.26.4,<br>pandas 2.2.3,<br>scipy 1.15.3,<br>rasterio 1.4.3 |
| 3 | Windows 11 Pro | 3.10.20 | 2.10.1 (GPU version) | numpy 1.23.5,<br>pandas 1.5.3,<br>scipy 1.10.1,<br>rasterio 1.4.4 |
| 4 | Windows 11 Pro | 3.9.21 | 2.15.0 (CPU version) | numpy 1.26.4,<br>pandas 2.3.1,<br>scipy 1.13.1,<br>rasterio 1.4.3 |

## 1.3 Any required non-standard hardware

None

# 2. Installation guide

## 2.1 Instructions

1. Install a TensorFlow version (`tensorflow>=2.10,<2.16`) that is compatible with your operating system, CUDA version, cuDNN version, and NVIDIA driver. For detailed installation instructions, visit https://www.tensorflow.org/install.
2. Install the remaining required packages: `numpy`, `pandas`, `scipy`, and `rasterio`.
3. Download Terra-GPT codes form the github: https://github.com/hankui/Terra-GPT

## Included Files

1. `Pro_HLS_GPT_application_v4_7.py`: Main command-line script for applying pretrained Terra-GPT models.
2. `config.py`: Task selection, input-band metadata, scaling constants, and model paths.
3. `HLS_process.py`: Chunk-based inference workflow and task-specific post-processing.
4. `HLS_io_chunks.py`: HLS tile reader and quality-mask handling.
5. `model_load.py`: Terra-GPT model construction and pretrained-weight loading.
6. `transformer_encoder44.py`: Transformer model definition.
7. `multi_head_from_ChatGPT.py`: Custom multi-head attention components.
8. `CONUS_scale60_all_tiles_v2_1.train.mean.02.02.2026.csv`: Band-wise training means and standard deviations used for normalization.

## 2.2 Typical install time on a ‘normal’ desktop computer

Spend time: approximately 4 minutes 42 seconds.

Device: local desktop computer running Windows 11 Pro with an NVIDIA GPU.

# 3. Demo

The demo data HLS data have only 128×128 30 m pixels in each tif file. This is for fast testing only. The codes can be directly applied to the original HLS tif files each with 3660×3660 30 m pixels.

## 3.1 Instructions to run on data

### (1) Download the trained model weights

Download the models located in https://zenodo.org/records/21810783 to local computers and change the `MODEL_PATHS` variable in the `config.py` file to reflect the model path users put in their local computer.

### (2) Terra-GPT codes and small demo data are included in the zip file; they can also be downloaded from:

Codes: https://github.com/hankui/Terra-GPT

Demo data: https://zenodo.org/records/21810783

### (3) Configure and run the Demo

You can find a detailed description of the input arguments in the GitHub repository:

https://github.com/hankui/Terra-GPT

Alternatively, run the following command to display the descriptions of all input parameters:

```bash
python Pro_HLS_GPT_application_v4_7.py --help
```

In the relevant script under `/Terra-GPT/demo/*.sh`, update `--hls_data_dir`, `--output_dir`, and `--dem_dir`. The required HLS and DEM files are included in the downloaded demo dataset.

Five tasks are currently supported.

#### For GAP_FILL task

In `config.py` file, set

```python
TASK = "GAP_FILL"
INPUT_BANDS = "THERMAL_DEM"
```

More information about the `INPUT_BANDS`, can be found in (4) Input configurations.

Run

```bash
cd /path/to/Terra-GPT
bash demo/GAP_FILL.sh
```

#### For FUEL_MOISTURE task

In `config.py` file, set

```python
TASK = "FUEL_MOISTURE"
INPUT_BANDS = "THERMAL_DEM"
```

Run

```bash
cd /path/to/Terra-GPT
bash demo/FUEL_MOISTURE.sh
```

#### For SOIL_MOISTURE task

In `config.py` file, set

```python
TASK = "SOIL_MOISTURE"
INPUT_BANDS = "THERMAL_DEM"
```

Run

```bash
cd /path/to/Terra-GPT
bash demo/SOIL_MOISTURE.sh
```

#### For CROP_MAPPING task

In `config.py` file, set

```python
TASK = "CROP_MAPPING"
INPUT_BANDS = "THERMAL_DEM"
```

Run

```bash
cd /path/to/Terra-GPT
bash demo/CROP_MAPPING.sh
```

#### For CROP_DAMAGE task

In `config.py` file, set

```python
TASK = "CROP_DAMAGE"
INPUT_BANDS = "THERMAL"
```

Run

```bash
cd /path/to/Terra-GPT
bash demo/CROP_DAMAGE.sh
```

### (4) Input configurations

Three model input configurations are available:

- DEFAULT
- THERMAL
- THERMAL_DEM

The band order for Landsat 30 m (L30) and Sentinel-2 30 m (S30) inputs is defined as follows:

```python
L30_bands = {
    0: 'l30_doy', 1: "l30_coastal", 2: "l30_blue", 3: "l30_green", 4: "l30_red", 5: "l30_nir", 6: "l30_swir1", 7: "l30_swir2", 8: "l30_bt1", 9: "l30_bt2",
    10: 'l30_dem', 11: 'l30_slope', 12: 'l30_sin_aspect', 13: 'l30_cos_aspect'
}

S30_bands = {
    0: 's30_doy', 1: 's30_coastal', 2: 's30_blue', 3: 's30_green', 4: 's30_red', 5: 's30_nirA', 6: 's30_swir1', 7: 's30_swir2', 8: 's30_edge1', 9: 's30_edge2', 10: 's30_edge3', 11: 's30_nir8', 12: 's30_dem', 13: 's30_slope', 14: 's30_sin_aspect', 15: 's30_cos_aspect'
}
```

`THERMAL_DEM`: Uses all bands listed above, including the Landsat thermal bands and auxiliary topographic variables.

`THERMAL`: Uses L30 bands 0–9 and S30 bands 0–11. Auxiliary topographic variables, including DEM, slope, and aspect, are excluded.

`DEFAULT`: Uses L30 bands 0–7 and S30 bands 0–11. Landsat thermal bands and auxiliary topographic variables are excluded.

## 3.2 Expect output

| Task | Output files | Descriptions |
| --- | --- | --- |
| GAP_FILL | `tile_10TDM_date_2023201_THERMAL_DEM_landsat.tif`<br>`tile_10TDM_date_2023201_THERMAL_DEM_sentinel.tif`<br>`tile_10TDM_date_2023227_THERMAL_DEM_landsat.tif`<br>`tile_10TDM_date_2023227_THERMAL_DEM_sentinel.tif`<br>`tile_10TDM_date_2023235_THERMAL_DEM_landsat.tif`<br>`tile_10TDM_date_2023235_THERMAL_DEM_sentinel.tif`<br>`tile_browse_10TDM_date_2023201_THERMAL_DEM_sentinel.jpg` | Reconstructed Landsat (8 bands) and Sentinel-2 (12 bands) reflectance for date 2023201,2023227,and 2023235.<br><br>The last band indicating whether the pixel has a cloud-free observation. |
| FUEL_MOISTURE | `tile_10TDM_date_2023201_lfmc.tif`<br>`tile_10TDM_date_2023227_lfmc.tif`<br>`tile_browse_10TDM_date_2023201_lfmc.jpg`<br>`tile_browse_10TDM_date_2023227_lfmc.jpg` | LFMC for date 2023201 and 2023227, RGB browse for visualization also provided. Each lfmc tif file includes three bands: lfmc, retreival uncertainty and flags indicate whether the pixel has a cloud-free observation. |
| SOIL_MOISTURE | `tile_15TVH_date_2016005_sm.tif`<br>`tile_15TVH_date_2016012_sm.tif`<br>…<br>`tile_15TVH_date_2016366_sm.tif` | Soil moisture time series. Each tif file file includes three bands: soil moisture, retreival uncertainty and flags indicate whether the pixel has a cloud-free observation. Note SM is generated only for pixels with valid HLS observations on that date. |
| CROP_MAPPING | `tile_15TVH_2016001_2017001_crop_mapping.tif` | A single-band GeoTIFF image indicating the crop type. The legend follows the IBM–NASA multi-temporal crop classification dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification. |
| CROP_DAMAGE | `tile_14TMK_2018091_2018194_crop_damage.tif` | A single-band GeoTIFF image indicating whether crop damage occurred up to the end date. |

## 3.3 Expected run time for demo on a “normal” desktop computer

Test system: local desktop computer running Windows 11 Pro, with a 12th Gen Intel(R) Core(TM) i7-12700 processor (2.10 GHz), 64.0 GB RAM, and an NVIDIA RTX A2000 GPU with 6 GB of memory.

| Task | Run time (seconds) |
| --- | ---: |
| GAP_FILL | 32 |
| FUEL_MOISTURE | 37 |
| SOIL_MOISTURE | 29 |
| CROP_MAPPING | 17 |
| CROP_DAMAGE | 33 |

# 4. Instructions for use

## 4.1 How to run the software on your data

1. download the HLS time series data (including L30 and S30, in CONUS region)
2. download the CONUS DEM tiles in the HLS MGRS tiling system (https://zenodo.org/records/21117230)
3. download the Terra-GPT trained mode weights (https://zenodo.org/records/21084486), and change the `MODEL_PATHS` variable in the `config.py` file
4. In `/Terra-GPT/demo/YOUR_TASK.sh`, update the input arguments. Then set `TASK` and `INPUT_BANDS` in `config.py`.
5. Run `bash demo/YOUR_TASK.sh`

`Pro_HLS_GPT_application_v4_7.py` uses named command-line arguments:

`GAP_FILL` example:

```bash
python Pro_HLS_GPT_application_v4_7.py \
  --tile_id 14TNP \
  --start_date 2023001 \
  --end_date 2024001 \
  --hls_data_dir /path/to/hls_data \
  --output_dir /path/to/output \
  --dem_dir /path/to/dem_tiles \
  --reconstructed_dates 2023140,2023160,2023180
```

### Arguments

- `--tile_id`: HLS tile ID, for example `14TNP`.
- `--start_date`: Start date of the input time series, included in `[start_date, end_date)`, using `YYYYDOY` format.
- `--end_date`: End date of the input time series, excluded from `[start_date, end_date)`, using `YYYYDOY` format.
- `--hls_data_dir`: Directory containing the input HLS time series.
- `--output_dir`: Directory where output GeoTIFF files will be saved.
- `--dem_dir`: Directory containing HLS-tiled DEM files. Required by the command-line interface.
- `--reconstructed_dates`: Comma-separated `YYYYDOY` dates. Required for `GAP_FILL` and `FUEL_MOISTURE`; not allowed for `SOIL_MOISTURE`, `CROP_MAPPING`, or `CROP_DAMAGE`.
- `--chunk_size`: Spatial chunk size. Default: `1220`.
- `--img_width`: Output tile width in pixels. Default: `3660`.
- `--img_height`: Output tile height in pixels. Default: `3660`.
- `--batch_size`: TensorFlow inference batch size. Default: `2048`.
- `--is_evaluation`: Boolean flag for GAP_FILL evaluation mode. When true, observations on requested reconstruction dates are masked before inference. Default: `False`.

Note for GAP_FILL, SOIL_MOISTURE, and FUEL_MOISTURE: --start_date and --end_date should be exactly one year apart, beginning on any date.
Note for CROP_MAPPING: --start_date should be January 1 of the year to be mapped. --end_date can be any date after --start_date, but should be less than one year later. 
	For more meaningful output, we recommend setting --end_date at least six months after --start_date.
Note for CROP_DAMAGE: --start_date should be March 1 of the year to be mapped. --end_date can be any date after --start_date, but should be before October 15 of the same year. 
    Crop damage mapping is limited to the summer growing season because the model was trained and applied only to corn and soybean.

## Output Format

The last band in `GAP_FILL`, `SOIL_MOISTURE`, and `FUEL_MOISTURE` outputs is a valid-observation indicator:

- `1`: the pixel has at least one valid HLS observation for that date.
- `0`: the pixel has no valid HLS observation for that date, is invalid, or is unrelated.

### GAP_FILL

For each requested `reconstructed_dates` date, Terra-GPT saves separate Landsat and Sentinel-2 GeoTIFF files.

- Landsat output: 7 reflectance bands plus 1 valid-observation indicator band.
- Sentinel-2 output: 11 reflectance bands plus 1 valid-observation indicator band.
- Missing reflectance values are replaced with Terra-GPT reconstructed values.
- The indicator band refers to the original sensor observation on that date.

### SOIL_MOISTURE

For each HLS acquisition date in the selected date range, Terra-GPT saves one three-band GeoTIFF:

- Band 1: soil moisture.
- Band 2: retrieval uncertainty derived from the Gaussian negative log-likelihood output.
- Band 3: valid-observation indicator.

If a pixel has no valid Landsat or Sentinel-2 observation on that date, bands 1 and 2 are set to `-9999`, and band 3 is set to `0`.

### FUEL_MOISTURE

For each requested `reconstructed_dates` date, Terra-GPT saves one three-band GeoTIFF:

- Band 1: live fuel moisture content (LFMC).
- Band 2: retrieval uncertainty derived from the Gaussian negative log-likelihood output.
- Band 3: valid-observation indicator.

### CROP_MAPPING

Terra-GPT saves one single-band crop classification GeoTIFF for the input date range. The legend follows the IBM-NASA multi-temporal crop classification dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification.

### CROP_DAMAGE

Terra-GPT saves one single-band binary crop damage GeoTIFF for the input date range.

# 5. Citation

More details can refer to:

- Li, J., Zhang, H. K., and Roy, D. P. (2026). HLS-GPT: A Generative Pretrained Transformer (GPT) Model for Accurate Harmonized Landsat and Sentinel-2 (HLS) Reflectance Time Series Reconstruction. In review.
- Zhang, H. K.*, Li, J., Camps-Valls, G., Subedi, S., Maimaitijiang, M., Roberts, D., & Roy, D. P. (2026). Earth observation foundation model enables near-real-time land monitoring. In review.
