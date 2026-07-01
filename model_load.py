# model_load.py
import transformer_encoder44
from config import MAX_LANDSAT,MAX_SENTINEL2
import importlib
importlib.reload (transformer_encoder44)
from config import TASK, L8_model_bands_N, S2_model_bands_N, INPUT_BANDS

is_dem = INPUT_BANDS == "THERMAL_DEM"

def load_model(model_path, periods):
    if TASK=="GAP_FILL":
        active = "sigmoid"
        is_reflectance = True 
        n_out = 11
    elif TASK=="SOIL_MOISTURE":
        active = "sigmoid"
        is_reflectance = 2 
        n_out = 2
    elif TASK=="FUEL_MOISTURE":
        active = "linear"
        is_reflectance = 3 
        n_out = 2
    elif TASK=="CROP_MAPPING":
        active = None
        is_reflectance = False
        n_out = 13
    elif TASK=="CROP_DAMAGE":
        is_reflectance = 4
        n_out = 1
        active = None
    else:
        raise ValueError(f"{TASK} not defined")
    
    print (TASK)
    model_basic = transformer_encoder44.get_transformer_reflectance(MAX_LANDSAT=MAX_LANDSAT, MAX_SENTINEL2=MAX_LANDSAT, L8_bands_n=L8_model_bands_N,
                                                                  S2_bands_n=S2_model_bands_N,n_out=n_out, is_dem=is_dem,
                                                                  layern1=3, layern2=4, units=256,
                                                                  n_head=8, drop=0.1, is_day_input=1,
                                                                  is_sensor=True, is_xy=False, is_reflectance=is_reflectance, active=active,
                                                                  concat=4)

    model_basic.load_weights(model_path)
    if periods == MAX_LANDSAT:
        return model_basic
    
    model_long = transformer_encoder44.get_transformer_reflectance(MAX_LANDSAT=periods, MAX_SENTINEL2=periods, L8_bands_n=L8_model_bands_N,
                                                                  S2_bands_n=S2_model_bands_N,n_out=n_out, is_dem=is_dem,
                                                                  layern1=3, layern2=4, units=256,
                                                                  n_head=8, drop=0.1, is_day_input=1,
                                                                  is_sensor=True, is_xy=False, is_reflectance=is_reflectance, active=active,
                                                                  concat=4)
    
    embedding_name = ""
    for il, ilayer in enumerate(model_basic.layers):
        ilayer1 = model_basic.layers[il]
        ilayer2 = model_long.layers[il]
        # if (model_drop==0 and 'dropout' not in ilayer2.name) or model_drop>0: # to handle one model has dropout while the other does no
        # il1=il1+1
        # else:
        # continue
        # ilayer1 = model    .layers[il1]
        name_cls = ''.join([ic for ic in ilayer1.name if not ic.isdigit() and ic != '_'])
        name_ref = ''.join([ic for ic in ilayer2.name if not ic.isdigit() and ic != '_'])
        if "embedding" in name_cls:
            embedding_name = ilayer1.name
        if name_cls == name_ref and ilayer1.trainable and ilayer2.trainable and not not ilayer1.weights and not not ilayer2.weights:
            # print ("\t"+ilayer.name, end=" ")
            model_long.layers[il].set_weights(model_basic.layers[il].get_weights())
    
    print('using long model...')
    return model_long

