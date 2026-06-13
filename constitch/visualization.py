import sys
import os
import base64
import PIL
import io
import numpy as np


def image_info(image):
    return dict(
        shape = image.shape,
        dtype = image.dtype.name,
    )

def encode_image(image):
    """ Converts an image with a datatype larger than uint8 into a 3 channel uint8
    image that can be saved to a png/jpeg for web transmission
    """
    if image.ndim >= 3:
        image = image.reshape(-1, image.shape[-1])

    #num_channels = 3 if image.itemsize <= 3 else 4
    num_channels = 3
    newimage = np.zeros((image.shape[0], image.shape[1], num_channels), np.uint8)

    if np.issubdtype(image.dtype, np.integer):
        if np.issubdtype(image.dtype, np.signedinteger):
            image = image.astype(np.uint16)
        for i in range(num_channels):
            newimage[:,:,i] = (image >> (i * 8)) & 0xFF

    elif np.issubdtype(image.dtype, np.floating):
        mantissa, exponent = np.frexp(image)
        mantissa = np.round(mantissa * (np.iinfo(np.int16).max + 1)).astype(np.uint16)
        newimage[:,:,0] = exponent
        for i in range(num_channels - 1):
            newimage[:,:,i+1] = (mantissa >> (i * 8)) & 0xFF

    return newimage

def decode_image(image, shape=None, dtype=None, out=None):
    if out is None:
        outimage = np.zeros(shape, dtype)
    else:
        outimage, shape, dtype = out, out.shape, out.dtype

    num_channels = image.shape[2]

    if len(shape) >= 3:
        outimage = outimage.reshape(-1, image.shape[-2])
        assert outimage.shape[0] == image.shape[0]

    if np.issubdtype(dtype, np.integer):
        for i in range(num_channels):
            outimage |= image[:,:,i].astype(dtype) << (i * 8)
        if np.issubdtype(dtype, np.signedinteger):
            kjsdlfkjsl

    elif np.issubdtype(dtype, np.floating):
        outimage[...] = (image[:,:,1] | (image[:,:,2].astype(np.uint16) << 8)).astype(np.int16)
        outimage /= np.iinfo(np.int16).max + 1
        outimage *= 2.0 ** image[:,:,0].astype(np.int8)

    return outimage.reshape(shape)

def image_to_base64(image, format='png'):
    if image.ndim == 2:
        pass

    pilimage = PIL.Image.fromarray(image)
    image_data = io.BytesIO()
    pilimage.save(image_data, format=format)
    image_data.seek(0)

    img = ET.SubElement()
    encoded = 'data:image/{};base64,'.format(format) + base64.b64encode(image_data.getvalue()).decode()
    return encoded






