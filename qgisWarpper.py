import pickle
import sys,os
_path_full_list = [
    'D:/Program Files/QGIS 3.16/apps/qgis-ltr/./python', 
    'C:/Users/JC/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python', 
    'C:/Users/JC/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python/plugins', 
    'D:/Program Files/QGIS 3.16/apps/qgis-ltr/./python/plugins', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\Scripts', 
    'D:\\Program Files\\QGIS 3.16\\bin\\python37.zip', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\DLLs', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\lib', 
    'D:\\Program Files\\QGIS 3.16\\bin', 
    'C:\\Users\\JC\\AppData\\Roaming\\Python\\Python37\\site-packages', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\lib\\site-packages', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\lib\\site-packages\\win32', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\lib\\site-packages\\win32\\lib', 
    'D:\\Program Files\\QGIS 3.16\\apps\\Python37\\lib\\site-packages\\Pythonwin', 
    'C:/Users/JC/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python', 
    'D:/Data/Forestry/Landsat2/1986',

    'C:\\Users\\JC\\AppData\\Roaming\\Python\\Python37\\site-packages', 
    'D:\\Program Files\\Python\\Python37\\lib\\site-packages', 
    'D:\\Program Files\\Python\\Python37\\lib\\site-packages\\win32', 
    'D:\\Program Files\\Python\\Python37\\lib\\site-packages\\win32\\lib', 
    'D:\\Program Files\\Python\\Python37\\lib\\site-packages\\Pythonwin'
    ]
sys.path = _path_full_list


from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QFileDialog
from qgis.core import *
from osgeo import gdal
from processing.core.Processing import Processing
from qgis import *
import qgis.core
import processing


_mbx_angle = 122.308720

def start_app():
    QgsApplication.setPrefixPath("D:/Program Files/QGIS 3.16/apps/qgis-ltr", True)
    Processing.initialize()

def start_app_grass():
    QgsApplication.setPrefixPath("D:/Program Files/QGIS 3.16/apps/qgis-ltr-grass7", True)
    Processing.initialize()

def _file_check(input_raster, output_raster,overwrite):
    
    if os.path.isfile(output_raster):
        if overwrite:
            os.remove(output_raster)
        else:
            raise FileExistsError("Output raster exist, %s, no overwite"% output_raster)
    

    if not os.path.isfile(input_raster):
        raise FileNotFoundError("Inuput raster not found, %s"%input_raster)


def clip_raster_by_vector(input_raster, input_vector, output_raster, overwrite=False):
    _file_check(input_raster, output_raster, overwrite)
    params = {'INPUT': input_raster,
              'MASK': input_vector,
              'NODATA': 0,
              'ALPHA_BAND': False,
              'CROP_TO_CUTLINE': True,
              'KEEP_RESOLUTION': True,
              'OPTIONS': 'COMPRESS=LZW',
              'DATA_TYPE': 0,  # Byte
              'OUTPUT': output_raster,
              }

    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'gdal:cliprasterbymasklayer'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)
    print(result)
    return result
        

def fill_gap(input_raster, mask, output_raster, overwrite=False):
    _file_check(input_raster, output_raster, overwrite)
    params = {
        'INPUT': input_raster,
        'BAND': 1,
        'DISTANCE': 10,
        'ITERATIONS':0,
        'NO_MASK':False,
        'MASK_LAYER': mask,
        'OPTIONS':'',
        'EXTRA':None,
        'OUTPUT': output_raster,
              }
    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'gdal:fillnodata'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)
    return result


def merge_rasters(input_rasters, output_raster):
    # _file_check(input_raster, output_raster, overwrite)
    params = {
        'GRIDS': input_rasters,
        'Name': "mosaic",
        'TYPE': 6,
        'RESAMPLING':2,
        'OVERLAP':6,
        'BLEND_DIST': 10,
        'MATCH':0,
        'TARGET_USER_SIZE':100,
        'TARGET_USER_FITS':0,
        'TARGET_OUT_GRID': output_raster,
              }
    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'saga:mosaicrasterlayers'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)


def convert_to_tiff(input_raster, output_raster,overwrite=True):
    _file_check(input_raster, output_raster, overwrite)
    params = {
        'INPUT': input_raster,
        'DATA_TYPE': 1,
        'OUTPUT': output_raster,
    }
    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'gdal:translate'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)

def cloud_mask(input_raster, output_raster,overwrite=True):
    _file_check(input_raster, output_raster, overwrite)
    params = {
        'a': input_raster,
        'expression': '(int(A&14143)<=5378)*0 + (int(A&14143)>5378)*255',
        'output': output_raster,
    }
    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'grass7:r.mapcalc.simple'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)

def merge_masks(input_rasters, output_raster):
    # _file_check(input_raster, output_raster, overwrite)
    params = {
        'INPUT': input_rasters,
        'SEPARATE': False,
        'DATA_TYPE': 0,
        'NODATA_INPUT':0,
        'OUTPUT': output_raster,
              }
    feedback = qgis.core.QgsProcessingFeedback()
    alg_name = 'gdal:merge'
    # print(processing.algorithmHelp(alg_name))
    result = processing.run(alg_name, params, feedback=feedback)
    return result

def _print_all():
    for alg in QgsApplication.processingRegistry().algorithms():
        print(alg.id(), "->", alg.displayName())

def __clip_raster_by_vectorp_test():
    start_app()
    input_raster = r"D:/Data/Forestry/Landsat/Desert-Oasis/1986/LT05_L1TP_133033_19860727_20170221_01_T1_B1.TIF"
    input_vector = r"D:\Data\Forestry\Landsat\Desert-Oasis\Boundary\Boundary_2015_Fixed.shp"
    output_raster = r"D:\Projects\2021\qgis\test2.TIF"
    result = clip_raster_by_vector(input_raster, input_vector, output_raster, overwrite=False)
    print('result =', result)

def __fill_gap_test():
    start_app()
    input_raster = r"D:\Data\Forestry\Landsat\Desert-Oasis\2010\LE07_L1TP_133033_20100907_20161212_01_T1_B1.TIF"
    mask = r"D:\Data\Forestry\Landsat\Desert-Oasis\2010\gap_mask\LE07_L1TP_133033_20100907_20161212_01_T1_GM_B1.TIF.gz"
    output_raster = r"D:\Projects\2021\qgis\filled.TIF"
    result = fill_gap(input_raster, mask, output_raster)
    print('result =', result)

def __cloud_mask_test():
    start_app_grass()
    cloud_mask(r'E:\Data\QL\2001\raw\LE07_L1TP_132034_20010721_20200917_02_T1_QA_PIXEL.TIF', r'D:\Projects\2021\qgis\CloudMask.TIFF')

if __name__ == "__main__":
    # __fill_gap_test()
    # contrast()

    __cloud_mask_test()

    # start_app_grass()
    # _print_all()
    # cloud_mask()
    # start_app()
    # merge_rasters(
    #     [ 
    #     r'E:\Data\QL\2002\filter\133033B3.TIF', 
    #     r'E:\Data\QL\2002\filter\133034B3.TIF', 
    #     r'E:\Data\QL\2002\filter\134033B3.TIF', 
    #     r'E:\Data\QL\2002\filter\135033B3.TIF',
    #     r'E:\Data\QL\2002\filter\132034B3.TIF'
    #     ], r'D:\Projects\2021\qgis\merge')
    # convert_to_tiff(r'D:\Projects\2021\qgis\merge', r'D:\Projects\2021\qgis\merge.TIF')
    # clip_raster_by_vector(r'D:\Projects\2021\qgis\merge.TIF',  r"E:\Data\Shapes\QMNP_clip.shp", r'D:\Projects\2021\qgis\merge_cliped.TIF', overwrite=True)