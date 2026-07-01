# Terra-GPT
Terra-GPT is based on HLS-GPT and is a generative pretrained Transformer for 
	harmonized Landsat and Sentinel-2 reflectance data reconstruction 
	daily (gapless) live fuel moisture content (LFMC) retrieval
	near-real-time soil moisture retrieval  
	within-season crop mapping across US (note this is only used to compare with Prithvi model with training samples and 13 classes defined by Prithvi team, for a more comprehensive model please see https://github.com/hankui/In-season-crop-type-mapper)  
	near-real-time crop damage mapping 

All are applied to the CONUS region for any time. 

## Notes before running the codes  
Please first download the models located in https://doi.org/10.5281/zenodo.21084486 to local computers and change the MODEL_PATHS variable in the config.py file to reflect your local model path. 
In config.py, switch among different tasks using the variable TASK
The model need 30 m DEM, slope (in degrees, 0 means flat) and aspect (in degrees, 0/260 indicate north) in one geotiff file. We have put the HLS-tiled geotiff files in WHERE. 
You can download the files to local computer and update the DEM_DIR variable in the Pro_HLS_GPT_application_v4_5.py

## Requirements
- **Programming Languages**: Python 3.7+
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `rasterio`

## Included files
1. `Pro_HLS_GPT_application_v4_6.py` 
-This is the main script for applying the pretrained HLS model to reconstruct HLS tiles.
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
python Pro_HLS_GPT_application_v4_5.py \
  <tile_id> <start_date> <end_date> <hls_data_dir> \
  <output_dir> <reconstructed_dates>(optinonal) 
```
### Arguments
 - tile_id: The HLS tile name, e.g., '14TNP'.
 - start_date: Define the start date of the input time series. The model uses observations from [start_date, end_date]. e.g., ''2023152' means the input time series is 06/01/2023-06/01/2024  
 - end_date: Define the end date of the input time series. The model uses observations from [start_date, end_date]. e.g., ''2023152' means the input time series is 06/01/2023-06/01/2024  
 - hls_data_dir: The input HLS time series directory.
 - output_dir: The output directory.
 - reconstructed_dates (optinonal): used for time series reconstruction and LFMC retrival only to indicate which dates the model will generate results, by default, it will generate results for the dates with HLS files only 
	Use format year+DOY, e.g., '2023140'. If there are multiple dates, separate them with commas. 
	Note for the recconstruction the model only reconstructs reflectance for pixels with no good-quality observations on the reconstruction dates.
The pathes for the pretrained Transformer models for different masks are hardcoded.

### Output format
Time series refletance reconstruction:  
	Time series (on the HLS file dates and reconstruction dates) of Landsat (7-band) and Sentinel-2 (11-band) geotiff files on the reconstruction dates with one additional bands to indicate whether the pixel is . 
LFMC: 
	Time series (on the HLS file dates and reconstruction dates) of the two bands (first band LFMC, second band incidate whether the pixel has cloud-free observations) geotiff files. 
Soil moisture: 
	Time series (on the HLS file dates) of the two bands (first band soil moisture, second band incidate whether the pixel has cloud-free observations) geotiff files. 
Crop type: 
	A single-band tif image indicate the crop type (legend refer to ??). 
Crop damage: 
	A single-band tif image indicate there is crop damage up to the end date. 
	


## Citation
More details can refer to the paper: 
Li, J., Zhang, H. K., and Roy, D. P. (2026). HLS-GPT: A Generative Pretrained Transformer (GPT) Model for Accurate Harmonized Landsat and Sentinel-2 (HLS) Reflectance Time Series Reconstruction. In review.
1.	Zhang, H. K.*, Li, J., Camps-Valls, G., Subedi, S., Maimaitijiang, M., Roberts, D., & Roy, D. P. (2026). Earth observation foundation model enables near-real-time land monitoring. In Review.