import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line():
    def __init__(self, img_shape):
        # was the line detected in the last iteration?
        self.img_shape = img_shape
        self.detected = False  
        self.n = 5
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients over the last n iterations
        self.fits = [np.array([False])]  
        # last set of coefficients
        self.current_fit = None
        # avg coeffs of last n iterations
        self.avg_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
    def update_fit(self, x_values, coeffs):
        if self.detected:
            self.recent_xfitted.append(x_values)
            if len(self.recent_xfitted) > self.n:
                self.recent_xfitted = self.recent_xfitted[len(self.recent_xfitted)-self.n:]
            self.bestx = sum(self.recent_xfitted)/len(self.recent_xfitted)
            
            self.current_fit = coeffs
            self.fits.append(coeffs)
            if len(self.fits) > self.n:
                self.fits = self.fits[len(self.fits) - self.n:]
            self.avg_fit = sum(self.fits)/len(self.fits)
    
    def update_radius(self, ym_per_pix=30/720, xm_per_pix=3.7/700):
        left_fit_cr = self.avg_fit
        y_eval = self.img_shape[0]
        A = xm_per_pix/np.power(ym_per_pix, 2)
        B = xm_per_pix/ym_per_pix
        self.radius_of_curvature = (np.power((1+np.power(((2*A*left_fit_cr[0]*y_eval*ym_per_pix) + B*left_fit_cr[1]), 2)), 1.5))//np.abs((2*A*left_fit_cr[0]))
        return self.radius_of_curvature
             