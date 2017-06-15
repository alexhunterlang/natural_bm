# Documentation for Utils (utils.py)

Utility functions 

## functions

### dict\_2\_OrderedDict
```py

def dict_2_OrderedDict(d)

```



Convert a standard dict to a sorted, OrderedDict 


### merge\_OrderedDicts
```py

def merge_OrderedDicts(d1, d2)

```



Merge two OrderedDicts into a new one 


### scale\_to\_unit\_interval
```py

def scale_to_unit_interval(ndar, eps=1.1920929e-07)

```



Scales all values in the ndarray ndar to be between 0 and 1 


### standard\_save\_folders
```py

def standard_save_folders(save_folder, overwrite=True)

```



Creates a standard set of folders expected by the callbacks and<br />returns a dictionary of the folders


### tile\_raster\_images
```py

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0), scale_rows_to_unit_interval=True, output_pixel_vals=True)

```



Copied from http://deeplearning.net/tutorial/code/utils.py ~ <br /><br />Transform an array with one flattened image per row, into an array in<br />which images are reshaped and layed out like tiles on a floor.<br /><br />This function is useful for visualizing datasets whose rows are images,<br />and also columns of matrices for transforming those rows<br />(such as the first layer of a neural net).<br /><br />:type X: a 2-D ndarray or a tuple of 4 channels, elements of which can<br />be 2-D ndarrays or None;<br />:param X: a 2-D array in which every row is a flattened image.<br /><br />:type img_shape: tuple; (height, width)<br />:param img_shape: the original shape of each image<br /><br />:type tile_shape: tuple; (rows, cols)<br />:param tile_shape: the number of images to tile (rows, cols)<br /><br />:param output_pixel_vals: if output should be pixel values (i.e. int8<br />values) or floats<br /><br />:param scale_rows_to_unit_interval: if the values need to be scaled before<br />being plotted to [0,1] or not<br /><br /><br />:returns: array suitable for viewing as an image.<br />(See:`Image.fromarray`.)<br />:rtype: a 2-d array with same dtype as X.

