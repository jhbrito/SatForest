# Rasterize a portugal layer, using the rasterize field as pixel burn value.
# This expects the cos to be projected to SRID 3857 and perfectly aligned with
# the extent provided in the command.
gdal_rasterize\
  -init 99\
  -a rasterize\
  -a_srs EPSG:3857\
  -te -1061557.4488245286047459 4432124.6480876598507166 -687321.7583403056487441 5185487.9988663569092751\
  -ts 39168 78848\
  -ot byte\
  -co COMPRESS=DEFLATE\
  -co PREDICTOR=2\
  -co NUM_THREADS=3\
  -co BIGTIFF=YES\
  -co TILED=YES\
  -a_nodata 99\
  -l cos_pt_2015_l1_3857\
  /home/tam/Documents/work/projects/tesselo/projects/celpa/cos/2015/cos_pt_2015_l1_3857.shp\
  /home/tam/Documents/work/projects/tesselo/projects/celpa/cos/2015/cos_pt_2015_l1_3857.tif

# Add overviews for faster rendering.
gdaladdo -r nearest /home/tam/Documents/work/projects/tesselo/projects/celpa/cos/2015/cos_pt_2015_l1_3857.tif 2 4 8 16 32 64 128
