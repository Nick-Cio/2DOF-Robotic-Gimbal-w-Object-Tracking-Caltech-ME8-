# Import the necessary packages
import atexit, select, sys, termios

# Import the testing packages
from time import sleep

# Import useful packages
import hebi
import numpy as np              # For future use
import matplotlib.pyplot as plt

# Import OpenCV
import cv2

from math import pi, sin, cos, asin, acos, atan2, sqrt, inf
from time import sleep, time


def detector(shared):


    # Set up video capture device (camera).  Note 0 is the camera number.
    # If things don't work, you may need to use 1 or 2?
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise Exception("Could not open video device: Maybe change the cam number?")

    # Change the frame size and rate.  Note only combinations of
    # widthxheight and rate are allowed.  In particular, 1920x1080 only
    # reads at 5 FPS.  To get 30FPS we downsize to 640x480.
    camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS,           30)

    WIDTH = 640
    HEIGHT = 480
    MINIMUM_AREA = 200

    ###################
    # CAMERA SETTINGS #
    ###################

    # Change the camera settings, likely turning off the auto-features.
    #camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # Enable autofocus
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # Disable autofocus
    camera.set(cv2.CAP_PROP_FOCUS, 0)            # 0 - 255

    #camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)        # Auto mode
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)        # Manual mode
    camera.set(cv2.CAP_PROP_EXPOSURE, 111)      # 3 - 2047
    camera.set(cv2.CAP_PROP_GAIN, 30)          # 0 - 255

    #camera.set(cv2.CAP_PROP_AUTO_WB, 1.0)            # Enable auto white balance
    camera.set(cv2.CAP_PROP_AUTO_WB, 0.0)            # Disable auto white balance
    camera.set(cv2.CAP_PROP_WB_TEMPERATURE, 4581)      # 2000 - 6500

    camera.set(cv2.CAP_PROP_BRIGHTNESS, 70)  # 0 - 255
    camera.set(cv2.CAP_PROP_CONTRAST, 37)    # 0 - 255
    camera.set(cv2.CAP_PROP_SATURATION, 78)  # 0 - 255
    camera.set(cv2.CAP_PROP_SHARPNESS,  29)   # 0 - 255

    from cameracontrol import CameraControl
    # Alternatively, create sliders to control these settings.
    #CameraControl(camera)

    from hsvthresholds import HSVThresholds
    # Adjustable thresholds
    # limits = HSVThresholds()
    # which is the same as
    # limits = HSVThresholds([[0,179],[0,255],[0,255]], sliders=True)
    # or fixed thresholds
    # limits = HSVThresholds([[Hmin,Hmax],[Smin,Smax],[Vmin,Vmax]], sliders=False)


    #############
    # FUNCTIONS #
    #############

    # input: (x,y) coordinate center of object
    # output: required tilt/pan angle motor needs to make to center camera on center of object
    def object_pan_tilt_theta(x_object, y_object, camerapan, cameratilt):
        dist_from_center_x = x_object - x_center
        dist_from_center_y = y_object - y_center
        pan_theta = dist_from_center_x * scale_pan + camerapan
        tilt_theta = dist_from_center_y * scale_tilt + cameratilt
        return (pan_theta, tilt_theta)
    
    def max_min_pan_tilt(camerapan, cameratilt):
        max_pan = camerapan + (WIDTH // 2)*(-scale_pan)
        min_pan = camerapan - (WIDTH // 2)*(-scale_pan)
        max_tilt = cameratilt + (HEIGHT // 2)*(-scale_tilt)
        min_tilt = cameratilt - (HEIGHT // 2)*(-scale_tilt)
        #print(f"from func: {max_pan}")
        #print(f"from func: {min_pan}")
        return max_pan, min_pan, max_tilt, min_tilt


    ###########################
    # VARIABLE INITIALIZATION #
    ###########################

    # experimentally determined threshold
    limits = HSVThresholds([[77,88],[50,255],[50,255]], sliders=False)

    # experimentally determined conversion factor from pixels to motor rad.
    scale_pan = -0.0014 #rad/pix 
    scale_tilt = -0.0015 #rad/pix 
    x_center = WIDTH//2
    y_center = HEIGHT//2
    

    ###################
    # MAIN EVENT LOOP #
    ###################

    # Keep scanning, until 'q' hit IN IMAGE WINDOW.
    count = 0
    while True:
        # Grab an image from the camera.  Often called a frame (part of sequence).
        ret, image = camera.read()
        count += 1

        # Grab and report the image shape.
        (H, W, D) = image.shape
        #print(f"Frame #{count:3} is {W}x{H} pixels x{D} color channels.")

        # Convert the BGR image to RGB or HSV.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
        
        # input: an image converted to HSV values
        # for each pixel checks if passes threshold values
        # if pass threshold white, else black
        binary = cv2.inRange(hsv, limits[:,0], limits[:,1])
        
        # Grab the actual motor angles showing where the camera is pointing.
        # Based on where camera pointing can figure out the pan/tilt max/min for the FOV (share this)
        if shared.lock.acquire():
            camerapan = shared.motorpan
            cameratilt = shared.motortilt
            
            max_pan, min_pan, max_tilt, min_tilt = max_min_pan_tilt(camerapan, cameratilt)
            shared.max_pan = max_pan
            shared.min_pan = min_pan
            shared.max_tilt = max_tilt
            shared.min_tilt = min_tilt
                
            shared.lock.release()
        
        
        ###################
        # NOISE REDUCTION # 
        ###################
        
        # dilate then erode: rid holes
        binary = cv2.dilate(binary, None, iterations=4)
        binary = cv2.erode(binary, None, iterations=4)
        
        # erode then dilate: rid side noise
        binary = cv2.erode(binary, None, iterations=2)
        binary = cv2.dilate(binary, None, iterations=2)
        
        
        ############
        # CONTOURS #
        ############
        
        # get list of all contours: draws continiuos outline around white sections in binary
        (contours, hierarchy) = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # initialize list to track all contours that pass screening
        valid_contours_area = []
    
        # FILTER THROUGH CONTOURS AND ONLY CHOOSE VALID ONES
        for contour in contours:
            
            # fill in irregularities: increases robustness when covering (ex. finger/pencil)
            hull = cv2.convexHull(contour)
            # area enclosed by convexed contour
            area = cv2.contourArea(hull)
            # perimeter of hull contour
            perimeter = cv2.arcLength(hull, True)
            #4 % of perimenter
            epsilon = 0.04 * perimeter
            # forces to approximate perimeter of contour with 96% accuracy using limited lines
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # if able to approx. with only 4 lines --> rect.
            if len(approx) == 4 and area > 300:
                valid_contours_area.append((contour, area))
        
        
        # DRAW BIGGEST CONTOUR/HULL & CENTER OF OBJECT
        if valid_contours_area:
            biggest_valid_contour, _ = max(valid_contours_area, key=lambda x: x[1])
            biggest_hull = cv2.convexHull(biggest_valid_contour)
            
            M = cv2.moments(biggest_hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                cv2.drawContours(image, [biggest_valid_contour], 0, (255,0,0), 1)
                cv2.drawContours(image, [biggest_hull], 0, (0,0,0), 1)
                # motor theta commands to get to the center of the contour enclosing the biggest area
                # thus motor will track the largest object
                theta_pan1, theta_tilt1 = object_pan_tilt_theta(cX, cY, camerapan, cameratilt)

                
        # GATHER DATA ABOUT POSITIONS OF ALL OBJECTS
        objects_data = []
                
        for contour, area_val in valid_contours_area:
            hull = cv2.convexHull(contour)
            
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                theta_pan = object_pan_tilt_theta(cX, cY, camerapan, cameratilt)[0]
                theta_tilt = object_pan_tilt_theta(cX, cY, camerapan, cameratilt)[1]
                objects_data.append((theta_pan, theta_tilt))
            
                
        # Share:
        # 1) angles to get to the biggest object (for tracking)
        # 2) share position of all objects even smaller ones (for scanning)
        if shared.lock.acquire():
            
            try:
                shared.object_pan = theta_pan1
                shared.object_tilt = theta_tilt1
                    
            except NameError:
                shared.object_pan = None
                shared.object_tilt = None
                    
            shared.objects_data = objects_data.copy()
                    
            shared.new_data = True
            shared.lock.release()
        
                        
        # Draw crosshair: helpful user visual
        cv2.line(image, (WIDTH//2,7*HEIGHT//16), (WIDTH//2,9*HEIGHT//16), (255,255,255), 3)
        cv2.line(image, (7*WIDTH//16,HEIGHT//2), (9*WIDTH//16,HEIGHT//2), (255,255,255), 3)
        
        
        
        # Show the processed image with the given title.  Note this won't
        # actually appear (draw on screen) until the waitKey(1) below.
        cv2.imshow('Processed Image', image)

        # Check for a key press IN THE IMAGE WINDOW: waitKey(0) blocks
        # indefinitely, waitkey(1) blocks for at most 1ms.  If 'q' break.
        # This also flushes the windows and causes it to actually appear.
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
        
        # Check if the main thread signals this loop to end.
        if shared.lock.acquire():
            stop = shared.stop
            shared.lock.release()
            if stop:
                break

    # Close everything up.
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detector(None)