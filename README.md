# Terra-GPT
Terra-GPT is based on HLS-GPT and is a generative pretrained Transformer for  
- harmonized Landsat and Sentinel-2 reflectance data reconstruction
- daily (gapless) live fuel moisture content (LFMC) retrieval 
- near-real-time soil moisture retrieval
- within-season crop mapping across US (note this is only used to compare with Prithvi model with training samples and 13 classes defined by Prithvi team, for a more comprehensive model please see https://github.com/hankui/In-season-crop-type-mapper)
- near-real-time crop damage mapping 

All are applied to the CONUS region for any time. 

## Notes before running the codes  
- Please first download the models located in https://doi.org/10.5281/zenodo.21084486 to local computers and change the `MODEL_PATHS` variable in the config.py file to reflect your local model path. 
- In config.py, switch among different tasks using the variable `TASK`, switch among different inputs using the variable `INPUT_BANDS`
- If `INPUT BANDS=THERMAL_DEM`, the model needs 30m DEM, slope (in degrees, 0 means flat) and aspect (in degrees, 0/260 indicate north) in one geotiff file. We have put the HLS-tiled DEM geotiff files in https://zenodo.org/records/21117230. You can download the files to local computer and update the `DEM_DIR` variable in the Pro_HLS_GPT_application_v4_6.py

| TASK          | INPUT BANDS | MODEL NAME                                                                                  |
| ------------- | ----------- | ------------------------------------------------------------------------------------------- |
| GAP_FILL      | DEFAULT     | `best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_32.h5`       |
| GAP_FILL      | THERMAL     | `best_model_hls2026.layer4.METHOD2.BATCH1024.LR0.0005.EPOCH40.L20.1.GAPS0.5.v7_33.h5`       |
| GAP_FILL      | THERMAL_DEM | `best_model_hls2026.layer4.METHOD2.BATCH256.LR0.0001.EPOCH40.L20.1.GAPS0.5use_bt_dem.v7_35` |
| FUEL_MOISTURE | THERMAL_DEM | `MC_v6_4.SMmodel.B0004.r0.00001.e10.L0.10000.i5.h5`                                         |
| SOIL_MOISTURE | THERMAL_DEM | `v4_5.SMmodel.B0256.r0.00001.e30.L0.10000.U064.i4.h5`                                       |
| CROP_MAPPING  | THERMAL_DEM | `v3_9.layer4.METHOD2.BATCH128.LR0.0001.EPOCH20.L20.1.FT1.use_bt_dem.i0.model.h5`            |
| CROP_DAMAGE   | THERMAL     | `v5_0.layer4.METHOD2.BATCH2048.LR1e-06.EPOCH30.L20.1.FT1.i0.model 1.h5`                     |

The input band configurations are defined as follows:
- DEFAULT: This configuration does not use the L30 thermal bands or auxiliary topographic variables, including DEM, slope, and aspect.
- THERMAL: This configuration uses the L30 thermal bands but does not use auxiliary topographic variables, including DEM, slope, and aspect.
- THERMAL_DEM: This configuration uses both the L30 thermal bands and auxiliary topographic variables, including DEM, slope, and aspect.

For crop damage, the model was fine-tuned using growing season time series from Mar 1st to any date in growing season up to Oct 1st. 

## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`

## Included files
1. `Pro_HLS_GPT_application_v4_6.py` 
-This is the main script for applying the pretrained Terra-GPT model to reconstruct HLS tiles and downstream tasks.
2. `HLS_io_chunks.py` 
-Class defined for reading HLS tiles.
3. `config.py`
-Band metadata and constants.
4. `transformer_encoder44.py`
-HLS pretrained model definition
5. `multi_head_from_ChatGPT.py`
-multi head attention function 
6. `CONUS_scale60_all_tiles_v2_1.train.mean.02.02.2026.csv`
-csv file storing the mean and standard deviation values for each band used for normalization
## Usage
```
python Pro_HLS_GPT_application_v4_6.py \
  <tile_id> <start_date> <end_date> <hls_data_dir> \
  <output_dir> <reconstructed_dates>(optinonal) 
```
### Arguments
 - **tile_id**: The HLS tile name, e.g., '14TNP'.
 - **start_date**: Start date of the input time series, included in the interval [start_date, end_date). The format is YYYYDOY, where YYYY is the year and DOY is the day of year. For example, 2023001 represents January 1, 2023.
 - **end_date**: End date of the input time series, excluded from the interval [start_date, end_date). The format is YYYYDOY, where YYYY is the year and DOY is the day of year. For example, 2024001 represents January 1, 2024.
 - **hls_data_dir**: The input HLS time series directory.
 - **output_dir**: The output directory.
 - **reconstructed_dates (optinonal)**: used for time series reconstruction and LFMC retrival only to indicate which dates the model will generate results, by default, it will generate results for the dates with HLS files only 
	Use format year+DOY, e.g., '2023140'. If there are multiple dates, separate them with commas. 
	Note for the recconstruction the model only reconstructs reflectance for pixels with no good-quality observations on the reconstruction dates.
The pathes for the pretrained Transformer models for different masks are hardcoded.

   
### Output format
 - **Time series refletance reconstruction**  
	A time series of GeoTIFF files on both the original HLS observation dates and the reconstruction dates. Each output includes Landsat 7-band and Sentinel-2 11-band reflectance data, with one additional band indicating whether the pixel has a cloud-free observation. 
 - **LFMC**  
	A time series of three-band GeoTIFF files on both the original HLS observation dates and the reconstruction dates. The first band contains live fuel moisture content (LFMC), 
	the second band indicate the retreival uncertaity derived using the Gaussian Negative Log-Likelihood (NLL) loss, and the third band indicates whether the pixel has a cloud-free observation. 
 - **Soil moisture**  
	A time series of three-band GeoTIFF files on the original HLS observation dates. The first band contains soil moisture, 
	the second band indicate the retreival uncertaity derived using the Gaussian Negative Log-Likelihood (NLL) loss, and the third band indicates whether the pixel has a cloud-free observation. 
 - **Crop mapping**  
	A single-band GeoTIFF image indicating the crop type. The legend follows the IBM–NASA multi-temporal crop classification dataset: https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification. 
 - **Crop damage**  
	A single-band GeoTIFF image indicating whether crop damage occurred up to the end date. 
	


## Citation
More details can refer to the paper:  
 - Li, J., Zhang, H. K., and Roy, D. P. (2026). HLS-GPT: A Generative Pretrained Transformer (GPT) Model for Accurate Harmonized Landsat and Sentinel-2 (HLS) Reflectance Time Series Reconstruction. In review.  
 - Zhang, H. K.*, Li, J., Camps-Valls, G., Subedi, S., Maimaitijiang, M., Roberts, D., & Roy, D. P. (2026). Earth observation foundation model enables near-real-time land monitoring. In Review.