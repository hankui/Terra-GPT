# Algorithm: Terra-GPT inference workflow

## Inputs

- HLS tile ID, start date, end date
- HLS input directory, output directory, DEM directory
- task type: `GAP_FILL`, `SOIL_MOISTURE`, `FUEL_MOISTURE`, `CROP_MAPPING`, or `CROP_DAMAGE`
- input band configuration: `DEFAULT`, `THERMAL`, or `THERMAL_DEM`
- optional reconstructed dates for `GAP_FILL` and `FUEL_MOISTURE`
- pretrained Terra-GPT model weights
- band-wise normalization statistics

## 1. Read model and task configuration

- Load `TASK` and `INPUT_BANDS` from the configuration file.
- Set the number of Landsat and Sentinel-2 input bands.
- Set output scaling factors, fill value, model path, and maximum sequence length.
- Validate the input date range.
- Require reconstructed dates for `GAP_FILL` and `FUEL_MOISTURE`.
- Disallow reconstructed dates for `SOIL_MOISTURE`, `CROP_MAPPING`, and `CROP_DAMAGE`.

## 2. Build the HLS input file list

- Recursively search the HLS input directory for Fmask files.
- Select files matching the requested tile and date range.
- Extract all unique HLS acquisition dates.
- Sort the dates to form the global date list `all_dates`.

## 3. Load normalization statistics

- Read the training mean and standard deviation for each input band.
- Construct separate normalization arrays for Landsat and Sentinel-2.
- If DEM inputs are used, include DEM, slope, `sin(aspect)`, and `cos(aspect)`.

## 4. Initialize output arrays

- If `TASK` is `GAP_FILL`:
  - Create Landsat and Sentinel-2 output arrays for `reconstructed_dates`.
- If `TASK` is `SOIL_MOISTURE`:
  - Create output arrays for `all_dates`.
- If `TASK` is `FUEL_MOISTURE`:
  - Create output arrays for `reconstructed_dates`.
- If `TASK` is `CROP_MAPPING` or `CROP_DAMAGE`:
  - Create a single-band output map.
- For `GAP_FILL`, `SOIL_MOISTURE`, and `FUEL_MOISTURE`:
  - Initialize the last band as a valid-observation indicator with value `0`.

## 5. Process the tile by spatial chunks

- For each image chunk:
  - Determine row and column boundaries.
  - Initialize chunk-level Landsat and Sentinel-2 time-series arrays.
  - Initialize chunk-level quality masks.

## 6. Assemble chunk-level HLS time series

- For each date in `all_dates`:
  - Locate the Landsat and Sentinel-2 HLS files for the date.
  - Store normalized day-of-year information for both sensor sequences.
  - If `INPUT_BANDS` is `THERMAL_DEM`:
    - Read DEM, slope, and aspect for the chunk.
    - Convert aspect to `sin(aspect)` and `cos(aspect)`.
    - Append topographic variables to the input features.
  - If a Landsat file exists:
    - Read Landsat reflectance, thermal bands if required, and quality mask.
    - Store valid observations in the Landsat time-series array.
  - If a Sentinel-2 file exists:
    - Read Sentinel-2 reflectance and quality mask.
    - Store valid observations in the Sentinel-2 time-series array.
  - If neither sensor has valid observations for the entire chunk on this date:
    - Remove this date from the chunk sequence unless it is required for reconstruction.

## 7. Select valid pixels for inference

- Combine Landsat and Sentinel-2 quality masks.
- Select pixels with more than the minimum number of valid observations.
- Reshape the chunk from image format to pixel-sequence format.
- Keep only selected valid pixels.
- Normalize input bands using the training mean and standard deviation.

## 8. Load and adapt the Terra-GPT model

- Build the Transformer architecture for the selected task.
- Load pretrained model weights.
- If the current sequence length differs from the pretrained length:
  - Build a model with the current sequence length.
  - Copy compatible layer weights from the pretrained model.

## 9. Run model inference

- Split valid pixels into batches.
- For each batch:
  - Predict task-specific outputs using Terra-GPT.
- Store predictions for all valid pixels in the current chunk.

## 10. Post-process predictions

- If `TASK` is `GAP_FILL`:
  - Replace missing Landsat observations with predicted Landsat reflectance.
  - Replace missing Sentinel-2 observations with predicted Sentinel-2 reflectance.
  - Extract outputs for `reconstructed_dates`.
  - Append a valid-observation indicator:
    - `1` = the original sensor observation was valid on that date
    - `0` = the original sensor observation was missing, invalid, or unrelated.

- If `TASK` is `SOIL_MOISTURE`:
  - Use the first model output as soil moisture.
  - Transform the second model output using `exp()` to obtain uncertainty.
  - For each pixel and date:
    - If Landsat or Sentinel-2 has a valid observation:
      - Keep soil moisture and uncertainty.
      - Set valid-observation indicator to `1`.
    - Otherwise:
      - Set soil moisture and uncertainty to the fill value.
      - Set valid-observation indicator to `0`.
  - Insert the chunk result into the output array at dates matching `used_dates`.

- If `TASK` is `FUEL_MOISTURE`:
  - Extract predictions for `reconstructed_dates`.
  - Convert predicted LFMC using:
    ```text
    LFMC = prediction * Y_SCALE + Y_OFFSET
    ```
  - Convert predicted uncertainty using:
    ```text
    uncertainty = exp(predicted_uncertainty) * Y_SCALE
    ```
  - Append a valid-observation indicator:
    - `1` = Landsat or Sentinel-2 has a valid observation on that date
    - `0` = both sensors are missing or the date is unrelated.

- If `TASK` is `CROP_MAPPING`:
  - Apply `argmax` to the model output.
  - Convert zero-based class labels to map labels using `Y_OFFSET`.
  - Store the crop type map.

- If `TASK` is `CROP_DAMAGE`:
  - Identify the last time step with at least one valid HLS observation.
  - Extract the crop damage prediction at that time step.
  - Apply sigmoid activation and threshold at `0.5`.
  - Store the binary crop damage map.

## 11. Save outputs as GeoTIFF files

- Use an example HLS file to copy geospatial metadata.
- Set output data type to `float32`, nodata value to `-9999`, and compression to `deflate`.

- For `GAP_FILL`:
  - Save separate Landsat and Sentinel-2 reconstructed reflectance GeoTIFFs.

- For `SOIL_MOISTURE`:
  - Save one three-band GeoTIFF for each output date:
    - band 1 = soil moisture
    - band 2 = uncertainty
    - band 3 = valid-observation indicator

- For `FUEL_MOISTURE`:
  - Save one three-band GeoTIFF for each reconstructed date:
    - band 1 = LFMC
    - band 2 = uncertainty
    - band 3 = valid-observation indicator

- For `CROP_MAPPING`:
  - Save one crop classification GeoTIFF.

- For `CROP_DAMAGE`:
  - Save one binary crop damage GeoTIFF.

## 12. Release resources

- Clear the TensorFlow/Keras session after each chunk.
- Run garbage collection before processing the next chunk.
