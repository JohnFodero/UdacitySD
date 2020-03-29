import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from Line import Line
from Camera import Camera

class LaneFinder():
    def __init__(self, cal_files=[]):
        self.camera = Camera()
        self.camera.calibrate(cal_files)
        temp = cv2.imread(cal_files[0])
        _, _ = self.camera.calibrate_perspective(temp)
        self.img_shape = temp.shape
        self.left_lane = Line(self.img_shape)
        self.right_lane = Line(self.img_shape)
    
    def start_lane_find(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            ### TO-DO: Identify the nonzero pixels in x and y within the window ###

            good_left_inds = ((nonzerox < win_xleft_high) & (nonzerox > win_xleft_low) 
                                      & (nonzeroy < win_y_high) & (nonzeroy > win_y_low)).nonzero()[0]
            good_right_inds = ((nonzerox < win_xright_high) & (nonzerox > win_xright_low) 
                                      & (nonzeroy < win_y_high) & (nonzeroy > win_y_low)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if good_left_inds.shape[0] > minpix:
                histogram = np.sum(binary_warped[(nwindows-(window+1))*window_height:(nwindows-window)*window_height, win_xleft_low:win_xleft_high], axis=0)
                leftx_current = np.argmax(histogram) + win_xleft_low
            if good_right_inds.shape[0] > minpix:
                histogram = np.sum(binary_warped[(nwindows-(window+1))*window_height:(nwindows-window)*window_height, win_xright_low:win_xright_high], axis=0)
                rightx_current = np.argmax(histogram) + win_xright_low

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError as e:
            # Avoids an error if the above is not implemented fully
            print(e)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        # Color detected pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
        margin = 5
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([left_line_pts]), (255,255, 0))
        cv2.fillPoly(out_img, np.int_([right_line_pts]), (255,255, 0))

        return out_img

    def search_around_poly(self, binary_warped):
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # USING PREV fit not AVG fit for search
        left_fit = self.left_lane.current_fit
        right_fit = self.right_lane.current_fit
        left_fit_old = left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]
        right_fit_old = right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]
        left_lane_inds = ((nonzerox < left_fit_old+margin) & (nonzerox > left_fit_old-margin)).nonzero()[0]
        right_lane_inds = ((nonzerox < right_fit_old+margin) & (nonzerox > right_fit_old-margin)).nonzero()[0]

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fitx, right_fitx, ploty = self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        #plt.plot(left_fitx, ploty, color='yellow')
        #plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##

        return result

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
        ### Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, deg=2)
        right_fit = np.polyfit(righty, rightx, deg=2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        ### Calc both polynomials using ploty, left_fit and right_fit ###
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty
            self.left_lane.detected = False
            self.right_lane.detected = False
        else:
            self.left_lane.detected = True
            self.right_lane.detected = True
            self.left_lane.update_fit(left_fitx, left_fit)
            self.right_lane.update_fit(right_fitx, right_fit)

        return left_fitx, right_fitx, ploty

    def get_lines(self, img):
        binary_warped = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Find our lane pixels first
        if not self.left_lane.detected or not self.right_lane.detected:
            print('starting lane find')
            out_img = self.start_lane_find(binary_warped)
        else:
            out_img = self.search_around_poly(binary_warped)

        return out_img
    
    def get_vehicle_lane_position(self, px_to_m=3.7/700):
        center = self.left_lane.bestx[-1] + ((self.right_lane.bestx[-1] - self.left_lane.bestx[-1])/2)
        veh_pos = self.img_shape[1]/2
        return px_to_m*(veh_pos - center)
    
    def reset_lines(self):
        self.left_lane = Line(self.img_shape)
        self.right_lane = Line(self.img_shape)
        
    def process_image(self, image):
        self.top_down = self.camera.get_top_down(image)
        self.binary, self.s, self.sx = self.camera.pipeline(self.top_down)
        self.binary_lines = self.get_lines(self.binary)
        left_curv = self.left_lane.update_radius()
        right_curv = self.right_lane.update_radius()
        position = self.get_vehicle_lane_position()
        self.annotated = self.camera.show_lane_data(self.camera.display_lane(image, self.left_lane, self.right_lane), left_curv, right_curv, position)
        return self.annotated


