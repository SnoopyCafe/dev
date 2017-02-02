
# coding: utf-8

# # **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, see [this forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?spaceKey=CAR&id=29496372&questionTitle=finding-lanes---import-cv2-fails-even-though-python-in-the-terminal-window-has-no-problem-with-import-cv2) for more troubleshooting tips.**

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[1]:

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import collections
import math
get_ipython().magic('matplotlib inline')


# In[2]:

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
  
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
      
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    draw_avg_ext_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# In[3]:

# NOTE: I did not write this function (draw_avg_ext_lines).  
# I copied it pretty much verbatim from the author below in order to submit this project on time.
# However, I am reviewing it to get a better understanding of the processing.
#
# Author: Juan Marcos Ottonello
# Site: http://ottonello.gitlab.io/selfdriving/nanodegree/lane/lines/2016/12/15/lane-lines-submitted.html
#

debug=False
# We'll average information between frames
prev_lines = collections.deque([], 5)

def reset_prev_lines():
    prev_lines.clear()
    
def draw_avg_ext_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    Here I'm applying a couple of the suggested techniques, first I split the lines into
    left and right according to their slopes, then and to iterate only once, I take a moving
    average of the slopes and discard the lines with a slope outside of the average.
    I tried both including all lines into the average and only the ones passing the slope
    average test, and including everything seems to work better.
    """
    debug=False
    lSlopeAvg=0
    rSlopeAvg=0
    slope_tolerance=.1
    slope_tolerance_from_zero=.5
    bottom_y = img.shape[0]
    top_y = int(bottom_y /1.6)
    
    lLines = []
    rLines = []
    l = 1
    r = 1
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if np.absolute(slope) == np.inf or np.absolute(slope) < slope_tolerance_from_zero:
                continue
            if debug:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            if slope < 0:
                lSlopeAvg = lSlopeAvg + (slope - lSlopeAvg) / l
                l += 1
                if np.absolute(lSlopeAvg - slope) < slope_tolerance :
                    lLines.append((x1,y1))
                    lLines.append((x2,y2))
            else:
                rSlopeAvg = rSlopeAvg + (slope - rSlopeAvg) / r
                r += 1
                if np.absolute(rSlopeAvg - slope) < slope_tolerance :
                    rLines.append((x1,y1))
                    rLines.append((x2,y2))               
    
    """
    After having split the lines, I use cv2.fitline to fit the sets of points to a single line.
    cv2.fitline gives back a unit vector and a point in the line, both of which we can use
    to calculate the slope and intercept of the line function.
    """
    if len(lLines) > 0 and len(rLines) > 0  :
        [left_vx,left_vy,left_x,left_y] = cv2.fitLine(np.array(lLines, dtype=np.int32), cv2.DIST_L2,0,0.01,0.01)      
        left_slope = left_vy / left_vx
        left_b = left_y - (left_slope*left_x)

        [right_vx,right_vy,right_x,right_y] = cv2.fitLine(np.array(rLines, dtype=np.int32), cv2.DIST_L2,0,0.01,0.01)    
        right_slope = right_vy / right_vx
        right_b = right_y - (right_slope*right_x)

        # Average this line with previous frames       
        prev_lines.append((left_b, left_slope, right_b, right_slope))
    
    if len(prev_lines) > 0: 
        avg = np.sum(prev_lines, -3) /len(prev_lines)
        left_b = avg[0]
        left_slope = avg[1]
        right_b = avg[2]
        right_slope = avg[3]

        """
        Having the slope and intercept enables us to calculate the x coordinates of our desired
        start and end points at the top and the bottom of the road.
        """
        ltop_x = (top_y - left_b) / left_slope
        lbottom_x = (bottom_y - left_b) / left_slope

        rtop_x = (top_y - right_b) / right_slope
        rbottom_x = (bottom_y - right_b) / right_slope

        cv2.line(img, (lbottom_x, bottom_y), (ltop_x, top_y), color, thickness)
        cv2.line(img, (rbottom_x, bottom_y), (rtop_x, top_y), color, thickness)
        


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# run your solution on all test_images and make copies into the test_images directory).

# In[4]:

import os

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
#reading in an image

def lane_lines(image):
    color_select= np.copy(image)
    gray = grayscale(color_select)
    #plt.imshow(gray, cmap='gray')

    # Define a kernel size and apply Gaussian smoothing
    # before running Canny, which suppresses noise and spurious gradients by averaging
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap='gray') 

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(390,340), (510, 330), (imshape[1],imshape[0])]], dtype=np.int32)

    # Define a mask region
    masked_edges = region_of_interest(edges, vertices)
 
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 30
    min_line_len = 60
    max_line_gap = 70

    return hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)


# In[5]:


# Test on the static images 
in_path = 'test_images/'
out_path = 'test_soln/'
test_images = os.listdir(in_path) 

for img in test_images:
    filename, ext = os.path.splitext(img)
    if ext != ".jpg":
        continue
           
    test_image = mpimg.imread(in_path + img)
    lines_image = lane_lines(test_image)
   
    # Draw the lines on the edge image
    combo_image = weighted_img(lines_image, test_image) 
    fig = plt.figure()
    fig.suptitle(img)
    plt.imshow(combo_image)
    mpimg.imsave(out_path + img, combo_image, cmap='Greys_r')


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an `import error` when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt. Also, check out [this forum post](https://carnd-forums.udacity.com/questions/22677062/answers/22677109) for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://carnd-forums.udacity.com/display/CAR/questions/26218840/import-videofileclip-error) for more troubleshooting tips across operating systems.**

# In[6]:

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    # Draw the lines on the edge image
    lines_image = lane_lines(image)
    return weighted_img(lines_image, image)


# Let's try the one with the solid white lane on the right first ...

# In[7]:

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4", audio=False)

white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[8]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[9]:

yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().magic('time yellow_clip.write_videofile(yellow_output, audio=False)')


# In[10]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Reflections
# 
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
# 
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
# 

# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[11]:

challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().magic('time challenge_clip.write_videofile(challenge_output, audio=False)')


# In[12]:

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:



