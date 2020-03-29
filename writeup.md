## Project 2: Advanced Lane Finding

---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/original_vs_corrected.jpg "Undistorted"
[image2]: ./test_images/straight_lines1.jpg "Road Transformed"
[image3]: ./examples/binary_example.jpg "Binary Example"
[image4]: ./examples/transformation_comparison.jpg "Transformation Example"
[image5]: ./examples/window_fit.jpg "Window Fit"
[image6]: ./examples/final.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

---

### Reference
#### examples/example.ipynb
This notebook was for organizing and developing each feature separately. I could use this area to try different filtering techniques as well as develop the requirements for objects and interfaces, before being put to use to process the pictures/images in the project. The most useful portion of this notebook is the tuning section, where I utilize IPython Widgets to tune the camera pipeline filtering parameters in real time. These parameters are then applied as defaults to the camera class.

#### Project.ipynb
This notebook was used to test the class structure and interfaces as well as to generate the final output of the project. Here I created the output images and videos.

#### Line.py
The class is designed to store coefficients of lane polynomials, the x values for given lanes, the averages over a set window.

`Line.update_fit()` - This function takes an array of x_values and an array of coefficients and updates the most recent x_value, most recent coefficients, window of `Line.n` (default 5) x_values and coefficients, and  the average x_values and coefficients over the `Line.n` window.

`Line.update_radius()` - This function updates and returns  the radius value `Line.radius_of_curvature` in the line object. This uses a default pixels-to-meters conversion factor that can be modified by the user if the camera value changes.

#### Camera.py
This class contains the following useful functions:  
`Camera.calibrate()` - This takes a set of image file paths and uses the opencv chessboard calibration method to compute the distortion coefficients for the camera. These parameters are used later to undistort each image.  

`Camera.calibrate_perspective()` - This takes an image and draws the points used to transform (warp) the image. It also computes the transformation matrix needed for `Camera.transform_image`. It returns an image with the transformation points drawn (as lines) for visual inspection of the placement of the lines. This is useful for determining the proper points for transformation.

`Camera.dir_threshold()`, `Camera.mag_thresh()`, `Camera.hls_select()` - Various filtering tools added to detect lane lines. Not all of these are used in the pipeline, but were covered in the course and added for convenience in case they are needed.

`Camera.pipeline()` - This function applies the pipeline of filters to get a binary image for lane line detection. Details on the pipeline are described in detail later in this writeup.

`Camera.undistort_image()` - This function applies the `cv2.undistort()` function with all calibrated parameters.

`Camera.transform_image()` - This function applies the `cv2.warpPerspective()` function with the transformation matrix computed by `Camera.calibrate_perspective`.

`Camera.get_top_down()` - This function combines the `Camera.undistort_image()` and `Camera.transform_image()` to conveniently provide a undistorted and transformed image for processing.

`Camera.display_lane()` - This takes the original image with both lane objects and returns the image with a semi-transparent line transformed and drawn on to the image to represent the area of the lane. This is useful for a real-world visualization of the output of the entire lane detection pipeline.

#### LaneFinder.py
The `LaneFinder` class contains the `Camera` and `Line` classes used for processing images and video. It's main function is to combine all camera, line, and lane finding operations into easy to use functions. The `LaneFinder.process_image()` method is the main method used to apply the distortions, transformations, filtering, lane detection, and annotations to the image before outputting.

---


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

### Camera Calibration
#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
See the description below. The process used to calibrate the camera is the same as what was used for processing individual lane images and videos

### Pipeline (single images)

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First, we use the `camera.calibrate()` method defined in Camera.py in the root of this project. This method takes a list of paths to calibration images that contain chessboards. It also takes an optional argument for the chessboard size (the default 9 by 6 is used in the project for convenience.  

To compute the camera matrix and distortion coefficients we begin by detecting chessboard corners on each image in the list. From here we create a list of points and coordinates (`objpoints` and `imgpoints`) in a set of data that can be used as inputs `cv2.calibrateCamera()`. The parameters returned from this function are returned as properties to the `Camera` class for later use in the `camera.undistort_image()`.  

The use of the `Camera` class was helpful in this project to manage the necessary parameters in the distortion removal operation that is used on each frame of the video, as well as the perspective transform used later in the pipeline. It is feasible that an autonomous vehicle will use multiple cameras to process data in different locations. This class could be reused to calibrate each camera individually and effectively manage those parameters.  

![alt text][image1]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The image filtering pipeline is a tuned version with structure like the pipeline developed in the lessons. The steps for creating the binary image are as follows:

1. The HSL image and corresponding H, S, and L channels are created.
2. We compute the derivative (using the Sobel operation) in the x direction across the l_channel. Ideally finding increases/decreases in lightness in the x direction. This is then scaled to 255 so it can be displayed as a full saturation color chanel later.
3. This image is then thresholded to user specifications. The default values in the `Camera.pipeline()` method are used, which were tuned by the interactive gui on the `examples/example.ipynb` notebook.
4. The saturation channel is also thresholded (without Sobel operation) per tuned parameters (using the same notebook tuning setup above).
5. These two images are stacked and can be used later for processing.

As the video shows, we are able to obtain good results in uniform lighting conditions in this case. However image `output_images/shadow1.jpg` (pulled from the project_video.mp4 file) shows how compromised lighting conditions cause this filtering to fail. More work is needed to utilize other channels or image filters to better handle these situations. The filter tuning portion of the `example.ipynb` was used to start to process this image and determine better parameters or design changes. Given more time, this could be greatly improved.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code to transform the perspective can be found in the `Camera` class. Here I use the `Camera.calibrate_perspective()` class to compute the transformation matrix and draw the source points onto the test image. This is helpful for debugging the placement of the points. No automatic source point detection is used to compute these points, however more advanced systems may take this approach as this method is fairly manual. The `test_images/straight_lines1` image was used to calibrate the parameters.

```python
horizon_px = 470
mid = int(self.img_size[0]/2) #1280/2 = 640
tl = [mid - 80, horizon_px]
tr = [mid + 80, horizon_px]
bl = [200, self.img_size[1]-25]
br = [1100, self.img_size[1]-25]

src = np.float32([tl, tr, bl, br])
dst = np.float32([[200,0],
                  [self.img_size[0] - 200,0],
                  [200,self.img_size[1]],
                  [self.img_size[0] - 200,self.img_size[1]]])
```

This resulted in the following source and destination points (image size (720, 1280))

| Source        | Destination   |
|:-------------:|:-------------:|
| 560, 470      | 200, 0        |
| 720, 470      | 1080, 720     |
| 200, 720      | 200, 720      |
| 1100, 720     | 1080, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto the test image and its warped counterpart to verify that the lines appear parallel in the warped image.  

The example below shows the transformation on a curved road. The curvature appears subtle in the original image, however the transformation reveals the true curvature of the lane.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane-line detection has two main algorithms. The `LaneFinder.start_lane_find()` method is used first, when no previous lane data is used or later in the detection process when other methods are not able to properly fit a polynomial. This method "starts from scratch", starting at the bottom of the image and uses a sliding window. Each window step in the y direction searches for activated pixels within the window. It then shifts this window based on the center of the last window. The below image illustrates the search windows (green rectangles), the activated "lane" pixels, and the fit line.

![alt text][image5]

Once the first polynomial has been found, I switch to the `LaneFinder.search_around_poly()` method which uses the previous fit polynomial to look for activated pixels around it within a certain margin. This assumes the maximum change in curvature is constrained. This is a safe assumption on highways, but could fail quickly in subdivisions or secondary roads where many lanes diverge frequently. If this search method fails, the frame is not processed and we switch back to the previous `LaneFinder.start_lane_find()` method for the next frame. More advanced error handling could, for example, broaden the margin used to search to attempt to find lane markers. A retry count could be implemented where, given a number of attempts, the algorithm reverts to the `LaneFinder.start_lane_find()` method.
![alt text][image6]
#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
Here the `Line` class has the method `update_radius` that computes the radius of the last averaged line. I chose to compute the radius on the average line coefficients because this value will be required for the control systems to properly direct the vehicle within the turn. The instantaneous value may be too noisy or erroneous for the control system to handle. A key design factor in self-driving cars (or any large robotic system) is to properly distribute tasks. The lane finding algorithms are tasked to determine the lane lines, their position in relation to the vehicle, and their radius. Therefore if we can handle these errors sooner, there will be less remediation required for systems later in the self-driving software stack. Here, we use averaging as a basic method for reducing error and obtaining smoother lane following.  

The `LaneFinder` class contains the method `LaneFinder.get_vehicle_lane_position`. This method assumes the center of the frame is the center of the vehicle. Using the bottom point of the last averaged polynomial, I compute the center of the lane lines, and then compare that to the center of the frame. This, when modified by the pixel-to-meter ratio provided in the previous lesson, allows us to compute the offset in meters.  

The image in step 6. below shows these computations on the frame.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `Camera` class provides the method `Camera.display_lane()` that plots the lane within the bounds of the two lane polynomials. Here, the `cv2.fillPoly()` method is used to create the green lane seen on the image below. The semi-transparent overlay is an excelent visual to determine performance, and will be helpful as we start to evalutate the performance of the control system to maintain lane position.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The video pipeline is the same as the image pipeline, however here we can utilize the `LaneFinder.search_around_poly()` method after first finding the lanes with the window method. The video also tests the error handling portion of the pipeline, where we can revert to simpler lane detection schemes or reset the lane properties when the `LaneFinder.search_around_poly()` is unable to get a proper lane line fit.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  
At the end of the video, we see the lane finding algorithm fail. I used the `debug` section of the `Project.ipynb` notebook. Here I was able to extract an image that clearly made the algorithm fail. Once extracted, I used the `example.ipynb` to analyze this frame. Given more time, I would use this frame to change the `Camera.pipeline()` design. Filtering different channels with additional sobel filtering could help this situation.

#### 2. What could you do to make it more robust?
In this video, we see the lane finding algorithm fail after exposure to shadows in the lane. I implemented basic "retry" logic, where, if the polynomial doesn't fit, it will revert to the previous "windowed" technique for finding the lane. Other constraints could be applied (such as high opposing curvature of the lanes, crossing lanes, extreme curvature, etc) to trigger this reset to the `LaneFinder.start_lane_find()` algorithm.
