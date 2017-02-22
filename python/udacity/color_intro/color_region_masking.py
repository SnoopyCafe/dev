__author__ = 'admin'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def color_thresholds(orig_image):
    # Define color selection criteria
    ###### MODIFY THESE VARIABLES TO MAKE YOUR COLOR SELECTION
    red_threshold = 200
    green_threshold = 200
    blue_threshold = 200
    ######

    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Do a boolean or with the "|" character to identify
    # pixels below the thresholds
    return (orig_image[:, :, 0] < rgb_threshold[0]) \
                 | (orig_image[:, :, 1] < rgb_threshold[1]) \
                 | (orig_image[:, :, 2] < rgb_threshold[2])


def region_thresholds(XX, YY):
    # Define a triangular region of interest
    # Keep in mind the origin (x=0, y=0) is in the upper left in image processing
    # left_bottom = [140, 539]
    # right_bottom = [800, 539]
    # apex = [460, 316]

    left_bottom = [0, 539]
    right_bottom = [900, 539]
    apex = [475, 320]

    # Fit lines (y=Ax+B) to identify the  3 sided region of interest
    # np.polyfit() returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    return (YY > (XX * fit_left[0] + fit_left[1])) & \
            (YY > (XX * fit_right[0] + fit_right[1])) & \
            (YY < (XX * fit_bottom[0] + fit_bottom[1]))


# Read in the image and print some stats
image = mpimg.imread('test.jpg')
print('This image is: ', type(image),
         'with dimesions:', image.shape)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select_img = np.copy(image)
line_image = np.copy(image)


# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

# Mask color selection
color_select_img[color_thresholds(image) | ~region_thresholds(XX,YY)] = [0,0,0]

# Find where image is both colored right and in the region
line_image[~color_thresholds(image) & region_thresholds(XX, YY)] = [255,0,0]

plt.imshow(image)

#x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
#y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(XX, YY, 'r--', lw=4)

mpimg.imsave("test-color-after.jpg", color_select_img)
mpimg.imsave("test-region-after.jpg", line_image)