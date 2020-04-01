import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from Line import Line

class Camera():
    def __init__(self):
        self.M = None
        self.M_inv = None
        self.img_size = None
        
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        
        self.M = None
        self.Minv = None
    
    def calibrate(self, images, chessboard=(9,6)):
        img = cv2.imread(images[0])
        self.img_size = (img.shape[1], img.shape[0])
        objp = np.zeros((chessboard[1]*chessboard[0],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        out_imgs = []
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, chessboard, None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # Draw and display the corners
                #cv2.drawChessboardCorners(img, chessboard, corners, ret)
                out_imgs.append(img)
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size,None,None)
        
    def calibrate_perspective(self, img):
        
        horizon_px = 470
        mid = int(self.img_size[0]/2)
        tl = [mid - 80, horizon_px]
        tr = [mid + 80, horizon_px]
        bl = [200, self.img_size[1]-25]
        br = [1100, self.img_size[1]-25]
        src = np.float32([tl, tr, bl, br])
        
        color = (0, 255, 0) 
        thickness = 9
        oimg = np.array(img, copy=True)
        oimg = self.undistort_image(oimg)
        cv2.line(oimg, tuple(tl), tuple(tr), color, thickness) 
        cv2.line(oimg, tuple(tr), tuple(br), color, thickness) 
        cv2.line(oimg, tuple(br), tuple(bl), color, thickness) 
        cv2.line(oimg, tuple(bl), tuple(tl), color, thickness) 
 
        dst = np.float32([[200,0],[self.img_size[0] - 200,0],[200,self.img_size[1]],[self.img_size[0] - 200,self.img_size[1]]])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        
        warped = self.transform_image(img)
        cv2.line(warped, tuple(dst[0]), tuple(dst[1]), color, thickness) 
        cv2.line(warped, tuple(dst[1]), tuple(dst[3]), color, thickness) 
        cv2.line(warped, tuple(dst[3]), tuple(dst[2]), color, thickness) 
        cv2.line(warped, tuple(dst[2]), tuple(dst[0]), color, thickness) 
        
        return oimg, warped
    
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary = np.zeros_like(grad_dir)
        binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary
    
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
    
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude 
        sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*sobelxy/np.max(sobelxy))
        # 5) Create a binary mask where mag thresholds are met
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary
    
    def hls_select(self, img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]
        # 2) Apply a threshold to the S channel
        binary = np.zeros_like(S)
        binary[(S > thresh[0]) & (S <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary
    
    def pipeline(self, img, s_thresh=(90, 120), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:,:,0]
        l_channel = hls[:,:,1]
        s_channel = hls[:,:,2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        #s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #s_binary = np.zeros_like(s_channel)
        #s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        s_binary = self.pipeline2(img)
        color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
        return color_binary.astype('uint8'), s_binary, sxbinary
    
    def thresh(self, img, thresh_min, thresh_max):
        ret = np.zeros_like(img)
        ret[(img >= thresh_min) & (img <= thresh_max)] = 1
        return ret
    
    def pipeline2(self, img):
        b = np.zeros((img.shape[0],img.shape[1]))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        V = hsv[:,:,2]

        R = img[:,:,0]
        G = img[:,:,1]
        B = img[:,:,2]

        t_yellow_H = self.thresh(H,10,30)
        t_yellow_S = self.thresh(S,50,255)
        t_yellow_V = self.thresh(V,150,255)

        t_white_R = self.thresh(R,225,255)
        t_white_V = self.thresh(V,230,255)

        #b[(t_yellow_H==1) & (t_yellow_S==1) & (t_yellow_V==1)] = 1
        b[(t_white_R==1)|(t_white_V==1)] = 1
        
        return b
     
    def undistort_image(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
    
    def transform_image(self, img):
        return cv2.warpPerspective(img, self.M, self.img_size, flags=cv2.INTER_LINEAR)
        
    def get_top_down(self, img):
        return self.transform_image(self.undistort_image(img))
    
    def display_lane(self, original_img, left_lane, right_lane):
        binary_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        img_shape = original_img.shape
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        pts_left = np.array([np.transpose(np.vstack([left_lane.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (original_img.shape[1], original_img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
        return result
    
    def show_lane_data(self, img, left_curvature, right_curvature, position):
        font = cv2.FONT_HERSHEY_SIMPLEX
        spacing = 60
        start = 60
        scale = 2
        oimg = np.array(img, copy=True)
        cv2.putText(oimg,'Left Curvature = ' + str(left_curvature),(50,start), font, scale,(255,255,255),2)
        cv2.putText(oimg,'Right Curvature = ' + str(right_curvature),(50,start+spacing), font, scale,(255,255,255),2)
        cv2.putText(oimg,'Lane Center Offset = ' + str(round(position, 3)),(50,start+(2*spacing)), font, scale,(255,255,255),2)
        return oimg
        
        
        
        