from __future__ import division
import numpy as np

import cv2

import skimage.measure

def crop_image(img, threshold=0):
    rows = np.where(np.max(img, 0) > threshold)[0]
    cols = np.where(np.max(img, 1) > threshold)[0]
    img = img[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    return img

def get_square_image(img):
    if img.shape[0] > img.shape[1]:
        pad_up = (img.shape[0] - img.shape[1]) // 2
        pad_bottom = np.ceil((img.shape[0] - img.shape[1]) / 2).astype('int32')
        img = np.pad(img, ((0, 0), (pad_up, pad_bottom)), 'constant', constant_values=0)
        return img
    else:
        pad_left = (img.shape[1] - img.shape[0]) // 2
        pad_right = np.ceil((img.shape[1] - img.shape[0]) / 2).astype('int32')
        img = np.pad(img, ((pad_left, pad_right), (0, 0)), 'constant', constant_values=0)
        return img

def add_padding(img, paddding_size=4):
    return np.pad(img, ((paddding_size, paddding_size), (paddding_size, paddding_size)), 'constant', constant_values=0)

# Aspect Ratio Adaptive Normalization
def get_new_aspect_ratio(img):
    if img.shape[0] > img.shape[1]:
        return np.sqrt(np.sin((img.shape[1] / img.shape[0]) * (np.pi / 2)))
    else:
        return np.sqrt(np.sin((img.shape[0] / img.shape[1]) * (np.pi / 2)))

def linear_normalization(img):
    aspect_ratio = get_new_aspect_ratio(img)
    new_size = 20
    width = new_size if img.shape[0] > img.shape[1] else int(aspect_ratio * new_size)
    height = new_size if img.shape[1] > img.shape[0] else int(aspect_ratio * new_size)
    x_points = np.round(np.array(range(width)) * (float(img.shape[0]-1) / (width-1))).astype('int32')
    y_points = np.round(np.array(range(height)) * (float(img.shape[1]-1) / (height-1))).astype('int32')
    new_img = img[np.ix_(x_points, y_points)]
    return new_img

def lagrange_mapping(position, centroid, centroid2, orig_size, delta, delta2, deltacentroid):
    lag = ((2 * orig_size) * (position**2 - 2*position*centroid + centroid2 + (delta*position - deltacentroid)/2) - (4*centroid) * (position**2 + centroid2 - (delta2/4) - 2*position*centroid)) / delta2
    return lag

def moment_normalization(img):
    # moments
    moments = skimage.measure.moments(img, 2)
    centroid = [moments[0, 1] / moments[0, 0], moments[1, 0] / moments[0, 0]]
    central_moments = skimage.measure.moments_central(img, centroid[0], centroid[1], 2)
    delta_y = 4 * np.sqrt(central_moments[0, 2])
    delta_x = 4 * np.sqrt(central_moments[2, 0])
    delta_y2 = delta_y ** 2
    delta_x2 = delta_x ** 2
    delta_cent_y = delta_y * centroid[1]
    delta_cent_x = delta_x * centroid[0]
    centroid2 = np.power(centroid, 2)

    # Calculate new width and height
    aspect_ratio = get_new_aspect_ratio(img)
    new_size = 20
    width = new_size if img.shape[0] > img.shape[1] else int(aspect_ratio * new_size)
    height = new_size if img.shape[1] > img.shape[0] else int(aspect_ratio * new_size)

    x_points = np.array(range(width))
    y_points = np.array(range(height))
    xx = ((delta_x * (x_points - (width-1)/2)) / (width-1)) + centroid[0]
    yy = ((delta_y * (y_points - (height-1)/2)) / (height-1)) + centroid[1] 
    xx = np.round(lagrange_mapping(xx, centroid[0], centroid2[0], img.shape[0]-1, delta_x, delta_x2, delta_cent_x)).astype('int32')
    yy = np.round(lagrange_mapping(yy, centroid[1], centroid2[1], img.shape[1]-1, delta_y, delta_y2, delta_cent_y)).astype('int32')
    xx = np.clip(xx, 0, img.shape[0]-1)
    yy = np.clip(yy, 0, img.shape[1]-1)
    new_img = img[np.ix_(xx, yy)]
    return new_img

def normalize_shape(img):
    img = crop_image(img)
    img = moment_normalization(img)
    img = get_square_image(img)
    img = add_padding(img)
    return img

def preprocess_image(img):
    img = normalize_shape(img)
    #img = cv2.GaussianBlur(img, (1, 1), 0)
    return img