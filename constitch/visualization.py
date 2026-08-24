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





def encode_image2(image, extra_channel=1):
    """ Encodes a 16bit image into a 3 channel jpeg image. Only allows 2d images, as opposed to
    encode_image, and is designed to support jpeg compression
    """
    assert image.ndim == 2 and image.dtype == np.uint16

    newimage = np.zeros(image.shape + (3,), np.uint8)
    bits = np.zeros(image.shape, np.uint16)

    for srcbit in range(16):
        channel = (extra_channel + srcbit) % 3
        destbit = 7 - (15 - srcbit) // 3
        #bits = image & (1 << srcbit)
        np.bitwise_and(image, 1 << srcbit, out=bits)
        #print (bin(bits[0,0]), bin(image[0,0] & (1 << srcbit)))

        if srcbit > destbit:
            bits >>= srcbit - destbit
            #print (srcbit - destbit)
        else:
            bits <<= destbit - srcbit
            #print (destbit - srcbit)

        #print (bin(bits[0,0]))

        newimage[:,:,channel] |= bits
        #print (bin(image[0,0]), image[0,0], list(map(bin, newimage[0,0])), newimage[0,0], srcbit, channel, destbit)

    return newimage

def round_uint8(image, power, out=None):
    roundto = 1 << power
    out = np.clip(image, 0, 255 - roundto // 2, out=out)
    np.add(out, roundto // 2, out=out)
    np.bitwise_and(out, ~np.uint8(roundto - 1), out=out)
    return out

def decode_image2(image, extra_channel=1):
    assert image.ndim == 3 and image.shape[-1] == 3 and image.dtype == np.uint8

    round_uint8(image[:,:,extra_channel], 2, out=image[:,:,extra_channel])
    round_uint8(image[:,:,(extra_channel+1)%3], 3, out=image[:,:,(extra_channel+1)%3])
    round_uint8(image[:,:,(extra_channel+2)%3], 3, out=image[:,:,(extra_channel+2)%3])

    newimage = np.zeros(image.shape[:2], np.uint16)
    bits = np.zeros(image.shape[:2], np.uint16)

    for destbit in range(16):
        channel = (extra_channel + destbit) % 3
        srcbit = 7 - (15 - destbit) // 3
        np.bitwise_and(image[:,:,channel], 1 << srcbit, out=bits)

        if srcbit > destbit:
            bits >>= srcbit - destbit
        else:
            bits <<= destbit - srcbit

        newimage |= bits
        #print (list(map(bin, image[0,0])), image[0,0], bin(newimage[0,0]), newimage[0,0], srcbit, channel, destbit)

    return newimage



