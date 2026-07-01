# true colour display 
# Hankui Jul 11 2020 
# import true_color_noC

import os
import math
import subprocess
import datetime
import numpy as np
import rasterio

## ************************************************************************
## convert from tif to ENVI
stretch_cmd = os.getenv("HOME")+"/mycode/v3.0.classification/stretch/truecolor.log.asTIFF"

WORK_DIR = "./"
if not os.path.exists('./tmp'):
    os.makedirs('./tmp')


## new stretch after Andy's 2000 epoch
b_low=200
b_high=2800

## Landsat-8 stretch
b_low=148
b_high=4915

## image is the three band image in the order of blue, green and red
def log_stretch(image,b1_low,b1_high,b2_low,b2_high,b3_low,b3_high,true_color_file):
    
    _ddd = np.zeros(image.shape, dtype=np.uint8) # output image name
    ## process blue band
    band = 0
    _low = math.log(b1_low)
    _top = math.log(b1_high)
    _dat = image[band,:,:].copy()
    _dat[np.where(_dat <= 0)] = 1
    log_dat = np.log(_dat.astype(np.float32))
    _ddd[band, log_dat <= _low] = 0
    _ddd[band, log_dat >= _top] = 255    
    _idx_1 = (log_dat > _low) & (log_dat < _top)
    _ddd[2-band,_idx_1] = np.ceil((log_dat[_idx_1] - _low) * (255.0 / (_top - _low)))
    # _ddd[band,_idx] = 0

    ## process green band
    band = 1
    _low = math.log(b2_low)
    _top = math.log(b2_high)
    _dat = image[band,:,:].copy()
    _dat[np.where(_dat <= 0)] = 1
    log_dat = np.log(_dat.astype(np.float32))
    _ddd[band, log_dat <= _low] = 0
    _ddd[band, log_dat >= _top] = 255    
    _idx_1 = (log_dat > _low) & (log_dat < _top)
    _ddd[2-band,_idx_1] = np.ceil((log_dat[_idx_1] - _low) * (255.0 / (_top - _low)))
    
    ## process red band
    band = 2
    _low = math.log(b3_low)
    _top = math.log(b3_high)
    _dat = image[band,:,:].copy()
    _dat[np.where(_dat <= 0)] = 1
    log_dat = np.log(_dat.astype(np.float32))
    _ddd[band, log_dat <= _low] = 0
    _ddd[band, log_dat >= _top] = 255    
    _idx_1 = (log_dat > _low) & (log_dat < _top)
    _ddd[2-band,_idx_1] = np.ceil((log_dat[_idx_1] - _low) * (255.0 / (_top - _low)))
    
    ## output geotif
    naip_meta=rasterio.profiles.DefaultGTiffProfile()
    naip_meta['count']  = _ddd.shape[0] 
    naip_meta['width']  = _ddd.shape[2] # a bug fixed on Jul 25 2021 
    naip_meta['height'] = _ddd.shape[1] 
    naip_meta['dtype'] = 'uint8'
    # if patch_toa1.dtype==np.uint16: 
        # naip_meta['dtype'] = 'uint16'
    # Write your the ndvi raster object
    with rasterio.open(true_color_file, 'w', **naip_meta) as dst:
        dst.write(_ddd)    
    return 0

## ABI stretch
# b_low=300
# b_high=10000

## Landsat-8 DN stretch
# b_low=2000
# b_high=10000

def true_color (input_file, win=0, blue=0, green=1, red=2):
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)+str(os.getpid())
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    base1 = os.path.basename(input_file)
    ture_color_tif_name = WORK_DIR+"True_color."+base1
    
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    with rasterio.open(input_file) as src:
        # print(src1.profile)
        if win==0:
            win = rasterio.windows.Window(0, 0, src.width, src.height)
        
        image = src.read(window=win)
        dims = image.shape
        
        image[blue ,:,:].tofile(filenames[0])
        image[green,:,:].tofile(filenames[1])
        image[red  ,:,:].tofile(filenames[2])
        
        nrow = dims[1]
        ncol = dims[2]
        subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(b_low), str(b_high), 
            filenames[1], str(b_low), str(b_high), filenames[2], str(b_low), str(b_high), ture_color_tif_name ])
    
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])


## numbers copied from d:\mycode\mycode.wylde\mk.jpg\truecolor.jpg.sh
b1_low=700
b1_high=2500

b2_low=500
b2_high=2300

b3_low=400
b3_high=2200

## numbers made up by Hank for Landsat-8 according to above 
# b1_low=700
# b1_high=2500*2

# b2_low=500
# b2_high=2300*2

# b3_low=400
# b3_high=2200*2

def true_color_from_image_cheatTOA (image, ture_color_tif_name, blue=0, green=1, red=2,factor_high=2, factor_low=1 ):
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)+str(os.getpid())
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    # base1 = os.path.basename(input_file)
    # ture_color_tif_name = WORK_DIR+"True_color."+base1
    
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    # with rasterio.open(input_file) as src:
        # print(src1.profile)
        # if win==0:
            # win = rasterio.windows.Window(0, 0, src.width, src.height)
        
        # image = src.read(window=win)
    dims = image.shape
    
    image[blue ,:,:].tofile(filenames[0])
    image[green,:,:].tofile(filenames[1])
    image[red  ,:,:].tofile(filenames[2])
    
    nrow = dims[1]
    ncol = dims[2]
    subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(int(b1_low*factor_low)), str(int(b1_high*factor_high)), 
        filenames[1], str(int(b2_low*factor_low)), str(int(b2_high*factor_high)), filenames[2], str((b3_low*factor_low)), str(int(b3_high*factor_high)), ture_color_tif_name ])
    
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])

def true_color_from_image (image, ture_color_tif_name, blue=0, green=1, red=2, b_low=b_low, b_high=b_high):
    ## ************************************************************************
    ## convert from tif to ENVI and tif image
    prefix = str(datetime.datetime.now().second) + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour)+str(os.getpid())
    filenames = [WORK_DIR+"tmp/"+prefix+".blue", WORK_DIR+"tmp/"+prefix+".green", WORK_DIR+"tmp/"+prefix+".red"]
    dims = image.shape    
    image[blue ,:,:].tofile(filenames[0])
    image[green,:,:].tofile(filenames[1])
    image[red  ,:,:].tofile(filenames[2])    
    nrow = dims[1]
    ncol = dims[2]
    subprocess.run([stretch_cmd, str(nrow), str(ncol), filenames[0], str(b_low), str(b_high), 
        filenames[1], str(b_low), str(b_high), filenames[2], str(b_low), str(b_high), ture_color_tif_name ])
    ## ************************************************************************
    ## delete temporary files 
    if os.path.isfile(filenames[0]): 
        os.remove(filenames[0])
    if os.path.isfile(filenames[1]): 
        os.remove(filenames[1])
    if os.path.isfile(filenames[2]): 
        os.remove(filenames[2])
    
    ## ************************************************************************
    ## without using c codes
    # log_stretch(image,b_low,b_high,b_low,b_high,b_low,b_high,ture_color_tif_name)
 


def true_color_from_image_noc (image, ture_color_tif_name, blue=0, green=1, red=2, b_low=b_low, b_high=b_high):    
    ## ************************************************************************
    ## without using c codes
    log_stretch(image,b_low,b_high,b_low,b_high,b_low,b_high,ture_color_tif_name)
 

