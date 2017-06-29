"""Utility functions """

#%%
import os
import shutil
import numpy as np
from collections import OrderedDict

import natural_bm.backend as B


#%%
def dict_2_OrderedDict(d):
    """Convert a standard dict to a sorted, OrderedDict """

    od = OrderedDict()
    if d is not None:
        keys = list(d.keys())
        keys.sort()
        for k in keys:
            od[k] = d[k]

    return od


#%%
def merge_OrderedDicts(d1, d2):
    """Merge two OrderedDicts into a new one """

    od = OrderedDict()
    for d in [d1, d2]:
        for k, v in d.items():
            od[k] = v

    return od


#%%
def scale_to_unit_interval(ndar, eps=B.epsilon()):
    """Scales all values in the ndarray ndar to be between 0 and 1 """
    return ndar/ (np.max(ndar) - np.min(ndar) + eps)


#%%
def standard_save_folders(save_folder, overwrite=True):
    """Creates a standard set of folders expected by the callbacks and
    returns a dictionary of the folders
    """
        
    save_folder = os.path.normpath(save_folder)
    
    if os.path.exists(save_folder) and not overwrite:
        raise OSError('Folder already exists: {}'.format(save_folder))
    elif os.path.exists(save_folder) and overwrite:
        shutil.rmtree(save_folder)
    
    try:
        os.mkdir(save_folder)
    except OSError:
        basefolder = os.path.abspath(os.path.join(save_folder, '..'))        
        os.mkdir(basefolder)
        os.mkdir(save_folder)
    
    save_dict = {}
    save_dict['base'] = save_folder
    save_dict['csv'] = os.path.join(save_folder, 'history.txt')
    
    weight_folder =  os.path.join(save_folder, 'weights')
    os.mkdir(weight_folder)
    save_dict['weights'] = os.path.join(weight_folder, 'weights.{epoch:04d}.hdf5')

    opt_weights = os.path.join(save_folder, 'opt_weights')
    os.mkdir(opt_weights)
    save_dict['opt_weights'] = os.path.join(opt_weights, 'opt_weights.{epoch:04d}.hdf5')
    
    plots =  os.path.join(save_folder, 'plots')
    os.mkdir(plots)
    save_dict['plots']  = plots
    
    return save_dict


#%%
def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Copied from http://deeplearning.net/tutorial/code/utils.py    
    
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp
                 for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    this_img = this_x.reshape(img_shape)
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(this_img)
                    
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                        
                    row_start = tile_row * (H + Hs)
                    row_end = row_start + H
                    col_start = tile_col * (W + Ws)
                    col_end = col_start + W
                    out_array[row_start : row_end,
                              col_start : col_end] = this_img * c
                              
        return out_array
