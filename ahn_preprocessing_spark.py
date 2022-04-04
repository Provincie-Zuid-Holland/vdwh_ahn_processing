# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## This notebook does the preprocessing for the height data or AHN data of the .laz format.
# MAGIC ### LAZ files are quite large a lot of ram is needed to process them 250 gb ram is not unusual

# COMMAND ----------

import os
import rasterio
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rasterio.crs import CRS
from affine import Affine
from rasterio import windows
from rasterio import mask
from rasterio import merge
import math
import pandas as pd
from glob import glob
import pylas
import geopandas as gpd
# Spark functionalities.
from pyspark.sql.functions import array, array_sort, floor, col, size
from pyspark.sql import Column
from pyspark.sql.functions import lag  
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from pyspark.sql.functions import lead  
from pyspark.sql.functions import when
import shutil

# COMMAND ----------

def calc_median(p, *args):
    """
    Pyspark formula for calculating the median.
  
    @param p: The percentile of which to take the median.
    @param args: values of which to take a median, these should be python columns.
    """
    def col_(c):
        if isinstance(c, Column):
            return c
        elif isinstance(c, str):
            return col(c)
        else:
            raise TypeError("args should str or Column, got {}".format(type(c)))

    xs = array_sort(array(*[col_(x) for x in args]))
    n = size(xs)
    h = (n - 1) * p
    i = floor(h).cast("int")
    x0, x1 = xs[i], xs[i + 1]
    return x0 + (h - i) * (x1 - x0)


def normalize_parameters(src):
     """
        Normalise rgb values.
     """
     print("Performing channel normalization...")
     channel_normalisation = {}
     combined_channel = []
     assert(len(src.indexes) == 4) # Expects input tif to only have RGBI (4) channels
     for index in tqdm(src.indexes):
            # print(f"\r{index} of {len(src.indexes)}", end="")
            channel = src.read(index).flatten()
            #combined_channel.extend(list(filter(lambda x: x > 0, channel))) # Remove black values or not.
            combined_channel.extend(list(channel))
            
     combined_channel.sort()
     bottom_percentile = int(len(combined_channel)*.005)
     top_percentile = int(len(combined_channel)*.995)

     for index in src.indexes:
            channel_normalisation[index] = {
                "min": combined_channel[bottom_percentile], 
                "max": combined_channel[top_percentile]
            }
     print("\nDONE\n")
     return channel_normalisation

def normalise(tile, norm, w, h):
        newtile = []
        for i in tqdm(range(len(tile))):
            n = norm[i+1]
            d = n["max"]-n["min"]
            if i == 3:
                newtile.append(np.array([[255] * w]* h, dtype=np.uint8))
            newtile.append(
                np.array(
                    list(map( lambda x: 
                        list(map(lambda y: 
                            0 if y < 0 else 255 if y > 255 else y, x
                        )), 
                        [list(map(lambda x: int(round((float(x)-n["min"])/d*255)), row)) for row in tile[i]]
                    )
                ), dtype=np.uint8))
        return newtile
    
def generate_vegetation_height_channel(vegetation_height_data, vegetation_height_transform, target_transform, target_width, target_height):
      
        print("Generating vegetation height channel...")
        channel = np.array([[0] * target_width] * target_height, dtype=np.uint8)
        src_height, src_width = vegetation_height_data.shape[0], vegetation_height_data.shape[1]
        for y in tqdm(range(target_height)):
            for x in range(target_width):
                rd_x, rd_y = rasterio.transform.xy(target_transform, y, x)
                vh_y, vh_x = rasterio.transform.rowcol(vegetation_height_transform, rd_x, rd_y)
                if vh_x < 0 or vh_x >= src_width or \
                    vh_y < 0 or vh_y >= src_height:
                    continue
                # print(rd_x, rd_y, x, y, vh_x, vh_y)
                channel[y][x] = vegetation_height_data[vh_y][vh_x]
        return channel

def generate_ndvi_channel(tile):
        print("Generating NDVI channel...")
        red = tile[2]
        nir = tile[3]
        ndvi = []
        for i in tqdm(range(len(red))):
            ndvi.append((nir[i]-red[i])/(nir[i]+red[i]+1e-10)*255)
        return np.array(ndvi, dtype=np.uint8)
    
def merge_raster(input1,bounds, res, nodata, precision):
    import warnings
    warnings.warn("Deprecated; Use rasterio.merge instead", DeprecationWarning)
    return rasterio.merge.merge(input1, bounds, res, nodata, precision)
  
def merge_tif_files(files, out_put_name, shape_file):
  
  bounds=None
  res=None
  nodata=None
  precision=None
    
  for file in range(0,len(files)):
    
    if file == 0:
      # For the first 2 files.
      dataset1 = rasterio.open(files[0])
      dataset2 = rasterio.open(files[1])
          
      dest, output_transform=merge_raster([dataset1,dataset2], bounds=bounds, res=res, nodata=nodata, precision=precision)

      with rasterio.open(files[0]) as src:
              out_meta = src.meta.copy()    
      out_meta.update({"driver": "GTiff",
                       "height": dest.shape[1],
                       "width": dest.shape[2],
                       "transform": output_transform})
      with rasterio.open("/home/mergedRasters.tif", "w", **out_meta) as dest1:
              dest1.write(dest)
          
    if file >= 2:
      # Merge the file with the already done ones.
      dataset1 = rasterio.open("/home/mergedRasters.tif")
      dataset2 = rasterio.open(files[file])

      dest, output_transform=merge_raster([dataset1,dataset2], bounds=bounds, res=res, nodata=nodata, precision=precision)

      with rasterio.open("/home/mergedRasters.tif") as src:
              out_meta = src.meta.copy()    
      out_meta.update({"driver": "GTiff",
                       "height": dest.shape[1],
                       "width": dest.shape[2],
                       "transform": output_transform})
      with rasterio.open("/home/mergedRasters.tif", "w", **out_meta) as dest1:
              dest1.write(dest)

  # Crop the combined file with the geojson.
  __make_the_crop(shape_file,"/home/mergedRasters.tif", "/home/mergedRasters_corrupted.tif")
  
  
  shutil.move("/home/mergedRasters_corrupted.tif", out_put_name)

def run_laz_to_tif(path_laz, path_output_tif, fill_up = True):
  
  if os.path.isdir("/home/tmp") != True:
    os.mkdir("/home/tmp")
    
  temp_path_tif = "/home/tmp/"+path_laz.split("/")[-1]
  path_parquet = path_laz.replace(".LAZ",".parquet")
  
  # Make a parquet file from a laz file if it does not exist.
  if os.path.isfile(path_parquet) != True:
    print("Parquet file not found converting laz file to parquet file.")
    laz = pylas.read(path_laz)

    cloud_df = pd.DataFrame({
              "x":laz.x, 
              "y":laz.y, 
              "z":laz.z,
              "classification": laz.classification
          })

    del laz

    for k in ["x", "y"]:
          cloud_df[k] = cloud_df[k].round()

    # Extract maaiveld from vegetatie.
    maaiveld_df = cloud_df[cloud_df["classification"] == 2]
    vegetation_df = cloud_df[cloud_df["classification"] == 1]  
    maaiveld_df = maaiveld_df.groupby(['x', 'y', "classification"]).agg({
                  'z': 'min'
              })

    maaiveld_df = maaiveld_df.reset_index()


    vegetation_df = vegetation_df.groupby(['x', 'y', "classification"]).agg({
                  'z': 'max'
              })

    vegetation_df = vegetation_df.reset_index()
    cloud_df = pd.merge(vegetation_df, maaiveld_df, how='outer', on=['x', 'y'], suffixes=["_vegetation", "_maaiveld"])
    del maaiveld_df
    del vegetation_df

    cloud_df["height"] = list(map(lambda x: 0 if pd.isnull(x[0]) else x[0] - x[1], np.array([cloud_df["z_vegetation"], cloud_df["z_maaiveld"]]).T))
    sparkDF = spark.createDataFrame(cloud_df)
    sparkDF.write.format("delta").mode("overwrite").save(path_parquet)
    del cloud_df
  else:
    print("Parquet file found")
    
  #Checkpoint.
  sparkDF = spark \
  .read \
  .format("delta") \
  .load(path_parquet)
  
  # Fill empty values with 0.
  sparkDF = sparkDF.fillna(0)
  
  if fill_up == True:
    print("Proceding to filling empty data.")
    # Fill up empty files based on it's nearest neighbour.
    windowX  = Window.orderBy(sparkDF.x)
    windowY = Window.orderBy(sparkDF.y)
    count_zeros = sparkDF.where(sparkDF.height.isNull()).count()
    # Get the median of the nearest neighbours 
    change_count = 0
    print("Number of empty values:")
    print(count_zeros)
    while  count_zeros > 0 and change_count < 3:


      sparkDF = sparkDF.withColumn("height_lag_x1",lag("height",1).over(windowX)).withColumn("height_lag_x-1",lead("height",1).over(windowX))\
      .withColumn("height_lead_y1",lag("height",1).over(windowY)).withColumn("height_lead_y-1",lead("height",1).over(windowY))

      sparkDF = sparkDF.withColumn("height_lag_x1y1",lag("height_lag_x1").over(windowY)).withColumn("height_lag_x1y-1",lead("height_lag_x1").over(windowY)) \
           .withColumn("height_lag_x-1y1",lag("height_lag_x-1").over(windowY)) \
           .withColumn("height_lag_x-1y-1",lead("height_lag_x-1").over(windowY))

      sparkDF = sparkDF.withColumn('height_median', calc_median(0.5,sparkDF["height_lag_x1"], sparkDF["height_lag_x-1"], sparkDF["height_lead_y1"], sparkDF["height_lead_y-1"],\
                                                               sparkDF["height_lag_x1y1"], sparkDF["height_lag_x1y-1"], sparkDF["height_lag_x-1y1"],sparkDF["height_lag_x-1y-1"]))
      sparkDF = sparkDF.select("x","y",
               when( sparkDF.height.isNull(), sparkDF.height_median ).otherwise(sparkDF.height).alias('height')
              )

      sparkDF.write.format("delta").mode("overwrite").save(path_parquet)
      count_zeros_cur = sparkDF.where(sparkDF.height.isNull()).count()
      if count_zeros_cur == count_zeros:
        change_count = change_count+1  
      else:
        change_count=0

      count_zeros = count_zeros_cur
      print("Number of empty values:")
      print(count_zeros)
    
    
    # Write to a numpy array.  
    min_x = sparkDF.agg({"x": "min"}).collect()[0][0]
    min_y = sparkDF.agg({"y": "min"}).collect()[0][0]
    max_x = sparkDF.agg({"x": "max"}).collect()[0][0]
    max_y = sparkDF.agg({"y": "max"}).collect()[0][0]
    res = 0.5
    pixel_data = []

    for x in range(int(((max_y - min_y)/res+2))):
            pixel_data.append([0] * int(((max_x - min_x)/res+2)))
        
    ahn_data = sparkDF.toPandas()
    res = .5
    ahn_data['value'] = sparkDF.rdd.map(lambda x:int(round((abs(x['height']*2550)**.5)))).map(lambda x:0 if x < 0 else 254 if x > 254 else x).collect()
    ahn_data['x_index'] = sparkDF.rdd.map(lambda x:int(round((x["x"]-min_x)/res)) ).collect()
    ahn_data['y_index'] = sparkDF.rdd.map(lambda x: int(round((max_y - x["y"])/res))).collect() 
    
    print("Filling array:")

    def fill_pixel_data(row):
        pixel_data[int(row["y_index"])][int(row["x_index"])] = row["value"] + 1
        pixel_data[int(row["y_index"])+1][int(row["x_index"])] = row["value"] + 1
        pixel_data[int(row["y_index"])][int(row["x_index"])+1] = row["value"] + 1
        pixel_data[int(row["y_index"])+1][int(row["x_index"])+1] = row["value"] + 1
        
    ahn_data.apply(lambda x:fill_pixel_data(x), axis=1)
    
    pixel_data = np.array([pixel_data], dtype=np.uint8)
    print(pixel_data.shape)

    meta = {
            'driver': 'GTiff', 
            'dtype': 'uint8', 
            'nodata': 0.0, 
            'width': pixel_data.shape[2], 
            'height': pixel_data.shape[1], 
            'count': 1, 
            'crs': CRS.from_epsg(28992), 
            'transform': Affine(.5, 0.0, min_x, 0.0, -.5, max_y)
    }

    with rasterio.open(temp_path_tif, 'w', **meta) as outds:        
            outds.write(pixel_data)
        
    
    shutil.move(temp_path_tif, path_output_tif) 

    
def __make_the_crop(load_shape, raster_path, raster_path_cropped):
    """
        This crops the sattelite image with a chosen shape.

        TODO: Make this accept a object of geopandas or shapely and crs independant.
        @param load_schape: path to a geojson shape file.
        @param raster_path_wgs: path to the raster .tiff file.
        @param raster_path_cropped: path were the cropped raster will be stored.
    """
    geo_file = gpd.read_file(load_shape)
    src = rasterio.open(raster_path)

    # Change the crs to rijks driehoek, because all the satelliet images are in rijks driehoek
    if geo_file.crs != 'epsg:28992':
        geo_file = geo_file.to_crs(epsg=28992)

    out_image, out_transform = rasterio.mask.mask(src,geo_file['geometry'], crop=True, filled=True)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(raster_path_cropped, "w", **out_meta) as dest:
            dest.write(out_image)
            dest.close()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Convert Height from .laz to layer in a .tif file.

# COMMAND ----------

!pip install pylas[lazrs]

# COMMAND ----------

# Arguments fill in for you local enviroment.
inpath_laz_file = "/dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/C_24HZ2.LAZ"
inpath_tif_file = "/dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/C_24HZ2_test_method.tif"

# COMMAND ----------

for file in glob("/dbfs/mnt/actueelhoogtebestand/ahn4/*[!HN1].LAZ"):
  print(file)

# COMMAND ----------

for file in glob("/dbfs/mnt/actueelhoogtebestand/ahn4/*"):
  print(file)

# COMMAND ----------

run_laz_to_tif("/dbfs/mnt/actueelhoogtebestand/ahn4/C_24HZ1.LAZ","/dbfs/mnt/actueelhoogtebestand/ahn4/C_24HZ1.tif")

# COMMAND ----------

for file in glob("/dbfs/mnt/actueelhoogtebestand/ahn4/*[!HN1].LAZ"):
  print(file)
  run_laz_to_tif(file, file.replace(".LAZ",".tif"))

# COMMAND ----------

# Done laz to .tif files merge into one big .tif file.
for file in glob("/dbfs/mnt/actueelhoogtebestand/ahn4/*.tif"):
  print(file)

# COMMAND ----------

files_append = []
for file in glob("/dbfs/mnt/actueelhoogtebestand/ahn4/*[!empty].tif"):
  print(file)
  files_append.append(file)

# COMMAND ----------

files_append.pop()

# COMMAND ----------

# Merge all .tif files into one.
merge_tif_files(files_append,"/dbfs/mnt/actueelhoogtebestand/ahn4/ahn4_waterleiding_duinen.tif", "/dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/waterleidingduin_aaneensluitende_polygon.geojson")

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC mkdir /home/temp

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC mv /home/temp/Coepelduynen_merge.tif /dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/ahn3_waterleidingduinen.tif

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Add height to satellite image.

# COMMAND ----------

ahn_output = '/dbfs/mnt/actueelhoogtebestand/ahn4/C_30EN2_fill_coepelduynen.tif'

# COMMAND ----------

os.mkdir("/home/tmp/")

# COMMAND ----------

with rasterio.open(ahn_output, 'r') as inds:        
  vegetation_height_data = inds.read(1)
  vegetation_height_transform = inds.meta["transform"]

# COMMAND ----------

for file in glob("/dbfs/mnt/e34a505986aa74678a5a0e0f_satellite-images-nso/coepelduynen/202[0|1]*cropped.tif"):
  print(file)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/mnt/satellite-images-nso/natura2000-geojson-gebieden/fotos/Coepelduynen/

# COMMAND ----------

for file in glob("/dbfs/mnt/e34a505986aa74678a5a0e0f_satellite-images-nso/coepelduynen/202[0|1]*"):

  with rasterio.open(file) as inds:  
          #channel_normalisation = normalize_parameters(inds)
          meta = inds.meta
          meta.update(count = 6)   
          tile = inds.read() # TODO is this behaviour similar to inds.read() ? MW: yes
          ndvi = generate_ndvi_channel(tile)
          #normalized_tile = np.array(normalise(tile, channel_normalisation, meta["width"], meta["height"]))            
          heightChannel = generate_vegetation_height_channel(vegetation_height_data, vegetation_height_transform, inds.meta["transform"], meta["width"], meta["height"])
          tile = np.append(tile, [heightChannel], axis=0)
          tile = np.append(tile, [ndvi], axis=0)
          
          file_to = "/dbfs/mnt/satellite-images-nso/natura2000-geojson-gebieden/fotos/Coepelduynen/"+file.replace(".tif","_ndvi_height.tif").split("/")[-1]
          file_local = "/home/tmp/"+file_to.split("/")[-1]
          with rasterio.open(file_local, 'w', **meta) as outds:        
                outds.write_band(1,tile[0])
                outds.write_band(2,tile[1])
                outds.write_band(3,tile[2])
                outds.write_band(4,tile[3])
                outds.write_band(5,ndvi)
                outds.write_band(6,heightChannel)
                
          shutil.move(file_local, file_to)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC mv /home/temp/* /dbfs/mnt/satellite-images-nso/coepelduynen/

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Add only NDVI

# COMMAND ----------

for file in glob("/dbfs/mnt/e34a505986aa74678a5a0e0f_satellite-images-nso/waterleidingduinen/*.tif"):
  print(file)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC ls /dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/ahn3_waterleidingduinen.tif

# COMMAND ----------

file = "/dbfs/mnt/e34a505986aa74678a5a0e0f_satellite-images-nso/waterleidingduinen/20190422_111333_SV1-01_50cm_RD_11bit_RGBI_Noordwijkerhout_waterleidingduin_aaneensluitende_polygon_cropped.tif"

with rasterio.open(file) as inds:  
          #channel_normalisation = normalize_parameters(inds)
          meta = inds.meta
          meta.update(count = 1)   
          tile = inds.read() # TODO is this behaviour similar to inds.read() ? MW: yes
          ndvi = generate_ndvi_channel(tile)
          #normalized_tile = np.array(normalise(tile, channel_normalisation, meta["width"], meta["height"]))            
          #heightChannel = generate_vegetation_height_channel(vegetation_height_data, vegetation_height_transform, inds.meta["transform"], meta["width"], meta["height"])
          #tile = np.append(tile, [heightChannel], axis=0)
          #tile = np.append(tile, [ndvi], axis=0)
          
          file_to = file.replace(".tif","_ndvi_only.tif")
          file_local = "/home/"+file_to.split("/")[-1]
          with rasterio.open(file_local, 'w', **meta) as outds:        
                outds.write_band(1,ndvi)
                
          shutil.move(file_local, file_to)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Merge ahn and NDVI .tif

# COMMAND ----------

file_tif_ahn = "/dbfs/mnt/actueelhoogtebestand/ahn3/Waterleidingduinen/ahn3_waterleiding_duinen.tif"
file_tif_ndvi = "/dbfs/mnt/e34a505986aa74678a5a0e0f_satellite-images-nso/waterleidingduinen/20190729_111526_SV1-04_50cm_RD_11bit_RGBI_Lisse_waterleidingduin_aaneensluitende_polygon_cropped_ndvi_only.tif"

# COMMAND ----------

inds_ndvi = rasterio.open(file_tif_ndvi) 

# COMMAND ----------

with rasterio.open(file_tif_ahn, 'r') as inds:        
  vegetation_height_data = inds.read()
  vegetation_height_transform = inds.meta["transform"]

# COMMAND ----------

vegetation_height_data = vegetation_height_data.reshape(vegetation_height_data.shape[1],vegetation_height_data.shape[2])

# COMMAND ----------

with rasterio.open(file_tif_ndvi, 'r') as inds:      
  meta = inds.meta
  meta.update(count = 2)   
  tile = inds.read() # TODO is this behaviour similar to inds.read() ? MW: yes
  #tile2 = rasterio.open(file_tif_ahn).read() 

            #normalized_tile = np.array(normalise(tile, channel_normalisation, meta["width"], meta["height"]))            
  heightChannel = generate_vegetation_height_channel(vegetation_height_data, vegetation_height_transform, inds.meta["transform"], meta["width"], meta["height"])
            #tile = np.append(tile, [heightChannel], axis=0)


  file_to = file_tif_ndvi.replace("ndvi_only.tif","_ndvi_ahn.tif")
  file_local = "/home/"+file_to.split("/")[-1]
  
  with rasterio.open(file_local, 'w', **meta) as outds:        
        outds.write_band(1,tile.reshape(tile.shape[1],tile.shape[2]))
        outds.write_band(2,heightChannel)

  shutil.move(file_local, file_to)

# COMMAND ----------

tile.reshape(tile.shape[1],tile.shape[2]).shape

# COMMAND ----------

tile.shape

# COMMAND ----------

heightChannel.shape

# COMMAND ----------



# COMMAND ----------


