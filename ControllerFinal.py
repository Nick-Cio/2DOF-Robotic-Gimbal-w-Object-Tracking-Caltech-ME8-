###############################
#        IMPORTS & SETUP      # 
###############################

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

from enum import Enum
class Traj(Enum):
    HOLD = 0
    SPLINE = 1
    SINE = 2
 # Keeps a constant position, zero velocity
 # Computes a cubic spline, with finite movetime
 # Computes sinusoidal position/velocity

class Mode(Enum):
    POINTING = 0
    TRACKING = 1
    SCANNING = 2
 # Go to a specific position and hold
 # Track the primary object of interest
 # Scan the entire field of view (w/o tracking)

def controller(shared):
    #
    #  The following changes the terminal into canonical mode, where it
    #  makes individual key presses available.  Note we return the
    #  terminal to the original settings on exit.
    #

    # Set up a handler, so the terminal returns to normal on exit.
    stdattr = termios.tcgetattr(sys.stdin.fileno())
    def reset_attr():
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, stdattr)
    atexit.register(reset_attr)

    # Switch the terminal to canonical mode: do not wait for <return>
    # presses.  Also prevent the keys from echoing.
    newattr    = termios.tcgetattr(sys.stdin.fileno())
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSAFLUSH, newattr)


    #KEY PRESS MONITORING AND RETREIVAL

    # Report True/False if a key has been pressed, awaiting retrieval.
    def kbhit():
        return sys.stdin in select.select([sys.stdin], [], [], 0)[0]

    # Grab the character of the last key pressed.  To avoid blocking,
    # check to make sure a character is indeed waiting to be retrieved.
    def getch():
        if kbhit():
            return sys.stdin.read(1)
        else:
            return None


    #  HEBI INITIALIZATION - don't worry about this!
    
    #  Create the motor group, and pre-allocate the command and feedback
    #  data structures.  Remember to set the names list to match your
    #  motor.
    names = ['9.9', '1.1']
    group = hebi.Lookup().get_group_from_names(['robotlab'], names)
    if group is None:
      print("Unable to find both motors " + str(names))
      raise Exception("Unable to connect to motors")

    command  = hebi.GroupCommand(group.size)
    feedback = hebi.GroupFeedback(group.size)
                 
            

    ####################################
    #  VARIABLE & LIST INITIALIZATION  # 
    ####################################

    # INITIAL POSITION
    feedback = group.get_next_feedback(reuse_fbk=feedback)
    pinit_pan = feedback.position[0]
    pinit_tilt = feedback.position[1]

    # LISTS FOR DATA

    #both
    Time = []
    historyofobjects = []
    known_objects = []

    #pan
    PAct_pan = []
    PCmd_pan = []
    VAct_pan = []
    VCmd_pan = []
    Error_vel_pan = []
    Error_pos_pan = []

    #tilt
    PAct_tilt = []
    PCmd_tilt = []
    VAct_tilt = []
    VCmd_tilt = []
    Error_vel_tilt = []
    Error_pos_tilt = []
    #Object_theta_tilt = []

    # CONSTANTS

    #both
    dt = 0.01     
    Tscan = 20
    pointing_keys = ["u", "d", "l", "r", "c"]    # up, down, left, right, center
    dist_threshold = 0.2                         # distance radius
    attention_span = 3                           # how long focuses on object for
    attention_span_over = False 
    known_objects_index = 0 #-1 
    
    #confidence level
    C_match = 20
    C_missing = 1
    C_outofview = 0.1
    C_threshold = 60
    C_initial = inf
    
    # interest level
    I_notfocus = 0.005
    I_movementfactor = 2
    I_outofview = 0.015
    I_discovery = 0.05
    interest_switch_bool = False
    max_interest_index = None

    #pan
    vmax_pan = 2.5
    amax_pan = vmax_pan / 0.6
    phold_pan = pinit_pan
    Apan = (72 * pi) / 180
    Npan = 3
    PHIpan = 0

    #tilt
    vmax_tilt = 2.5
    amax_tilt = vmax_tilt / 0.6
    phold_tilt = pinit_tilt
    Atilt = (15 * pi) / 180
    Ntilt = 4
    PHItilt = 0
    center_tilt = 0

    # CHANGING VARIABLES
    
    #precompute spline
    p0_pan = pinit_pan
    pf_pan = p0_pan
    v0_pan = 0.0
    vf_pan = 0.0

    p0_tilt = pinit_tilt
    pf_tilt = p0_tilt
    v0_tilt = 0.0
    vf_tilt = 0.0

    #times
    t = 0.0
    t0 = 0.0
    tf = 3

    #mode/traj
    traj = Traj.HOLD
    mode = Mode.POINTING
    
    ###############################
    #          FUNCTIONS          # 
    ###############################

    # MOVE TIME HELPER FUNCTION (given: p_init, p_final, and vmax output: time to complete movement)
    def move_time(p0, pf, vmax, v0, amax, vf=0):
        tv0_extra = abs(v0) / amax
        tvf_extra = abs(vf) / amax
        t_extra = tv0_extra + tvf_extra
        T = ((3/2) * (abs(pf-p0) / vmax)) + t_extra
        if T > 0.4:
            return T
        else:
            return 0.4


    # SPLINE PARAMS HELPER FUNCTION (outputs the coefficients of the spline function)
    def spline_params(t0, tf, p0, pf, v0, vf):
        a = p0
        b = v0
        c = 3 * (pf - p0) * (1 / (tf - t0)**2) - (vf) * (1 / (tf - t0)) - 2 * (v0) * (1 / (tf - t0))
        d = -2 * (pf - p0) * (1 / (tf - t0)**3) + (vf) * (1 / (tf - t0)**2) + (v0) * (1 / (tf - t0)**2)
        return (a, b, c, d)


    # SPLINE COMMAND HELPER FUNCTION (outputs the vmcd and pcmd for each dt)
    def spline_command(a, b, c, d, t, t0):
        pcmd = a + b * (t - t0) + c * (t - t0)**2 + d * (t - t0)**3
        vcmd = b + 2 * c * (t - t0) + 3 * d * (t - t0)**2
        return (pcmd, vcmd)
        
        
    # SINE COMMAND HELPER FUNCTION
    def sine_command(A, N, t, t0, Tscan, PHI, center=0):
        pcmd = A * sin(2*pi * N * (t - t0) / Tscan + PHI) + center
        vcmd = A * 2*pi * N / Tscan * cos(2*pi * N * (t - t0) / Tscan + PHI)
        return (pcmd, vcmd)
        
    
    # INPUT: LIST OF ALL DETECTED OBJECTS, LIST OF ALL PREVIOUSLY KNOWN OBJECTS
    # FIGURES OUT WHAT OBJECTS CURRENTLY IN FOV
    # UPDATES KNOWN OBJECTS IF NO MATCH
    # UPDATES CONFIDENCE & INTEREST LEVELS BASED ON MATCHING/NOT MATCHING & IN/NOT-IN FOV & IF MOVING
    def known_objects_to_detected_objects_mapper(known_objects, detected_objects, dist_threshold, known_objects_index):
                
        # CHECKS IF NEW DATA MATCHES TO ANY PREVIOUSLY KNOWN OBJECT LOCATION
        # yes: 
            # 1) updates position of known object to position of detection
            # 2) adds to list of known_objects that matched (for later confidence update)
        # no:
            # 1) appends data to known_objects
        
        known_objects_that_matched_indices = []
                
        # loop through each object
        for obj in detected_objects:
            obj = list(obj)
            # reset for each object
            match_found = False
            i = 0
            # compare until find a match or run out of known objects to compare to
            while not match_found and i != len(known_objects):
                dist_btw_objs = sqrt((obj[0] - known_objects[i][0])**2 + (obj[1] - known_objects[i][1])**2)
                if dist_btw_objs <= dist_threshold:
                    
                        new_confidence = known_objects[i][2] + C_match
                        obj.append(new_confidence)
                        new_interest = known_objects[i][3] + dist_btw_objs * I_movementfactor
                        obj.append(new_interest)
                        known_objects[i] = obj
                        known_objects_that_matched_indices.append(i)
                        match_found = True
                i+=1
            # if no match after search ends append new
            if not match_found:
                # append confidence level to [x, y]
                obj.append(C_initial + C_missing)
                # append interest level to [x, y, c]
                obj.append(I_discovery)
                known_objects.append(obj)
                print(f"new object added: {known_objects}")
                
         
        # LIST OF OBJECTS CURRENTLY IN FOV
        objects_in_FOV_indices = []
        for i in range(len(known_objects)):
            
            if objects_in_FOV(known_objects[i], max_pan, min_pan, max_tilt, min_tilt):
                objects_in_FOV_indices.append(i)
                print(known_objects_index)
                if i != known_objects_index:
                    known_objects[i][3] += I_notfocus
            else:
                known_objects[i][3] += I_outofview

        # CONFIDENCE UPDATE
        # update confidence depending on if matched or no
        for i in range(len(known_objects)):
            
            if i in objects_in_FOV_indices:
                if i not in known_objects_that_matched_indices:
                    known_objects[i][2] -= C_missing
            else:
                known_objects[i][2] -= C_outofview
                
        # CONFIDENCE BOUNDING
        i = 0
        while i < len(known_objects):
            if known_objects[i][2] > 100:
                known_objects[i][2] = 100
            if known_objects[i][2] <= 0:
                print("deleted object")
                known_objects.pop(i)
            else:
                i += 1
        
        # INTEREST NORMALIZATION
        interest_sum = 0
        for obj in known_objects:
            interest_sum += obj[3]
        if interest_sum > 0:
            for obj in known_objects:
                obj[3] = obj[3] / interest_sum
        
        # BOOL THAT DECIDES TO SWITCH TO NEXT OBJECT OR NOT
        interest_switch_bool = False
        max_interest_index = None
        
        if known_objects:
            # Returns tuple: (index, object)
            max_interest_index, best_obj = max(enumerate(known_objects), key=lambda x: x[1][3])
            
            if known_objects_index >= len(known_objects):
                        known_objects_index = 0
            else:
                I_switch = 1 / len(known_objects)
                if known_objects[max_interest_index][3] - known_objects[known_objects_index][3] >= I_switch:
                    interest_switch_bool = True
                else:
                    interest_switch_bool = False
        
        return known_objects, interest_switch_bool, max_interest_index
        
    
    # RETURNS BOOL IF AN OBJ IN FOV
    def objects_in_FOV(obj, max_pan, min_pan, max_tilt, min_tilt):

        if obj[0] > min_pan and obj[0] < max_pan and obj[1] > min_tilt and obj[1] < max_tilt:
            return True
        else:
            return False
    
    # CHECKS IF OBJECT IS ABOVE NEEDED CONFIDENCE LEVEL
    def valid_threshold(known_objects):
        for obj in known_objects:
            if obj[2] > 60:
                return True
        return False
    

    ####################################
    #          MAIN EVENT LOOP         # 
    ####################################

    while True:
        
        #DEPENDING ON TRAJECTORY STATE MOVE WITH WITH A CERTAIN BEHAVIOUR
        if traj is Traj.HOLD:
            pcmd_pan = phold_pan
            vcmd_pan = 0.0
            
            pcmd_tilt = phold_tilt
            vcmd_tilt = 0.0
            
        elif traj is Traj.SPLINE:
            (pcmd_pan, vcmd_pan) = spline_command(a_pan, b_pan, c_pan, d_pan, t, t0)
            (pcmd_tilt, vcmd_tilt) = spline_command(a_tilt, b_tilt, c_tilt, d_tilt, t, t0)

        elif traj is Traj.SINE:
            (pcmd_pan, vcmd_pan) = sine_command(Apan, Npan, t, t0, Tscan, PHIpan, center=0)
            (pcmd_tilt, vcmd_tilt) = sine_command(Atilt, Ntilt, t, t0, Tscan, PHItilt, center=center_tilt)
             
        else:
            raise ValueError(f'Bad trajectory type {traj}')

         
        # SEND POS/VEL COMMANDS TO MOTOR
        command.position = [pcmd_pan, pcmd_tilt]
        command.velocity = [vcmd_pan, vcmd_tilt]
        group.send_command(command)

        # READ ACTUAL POS/VEL
        feedback = group.get_next_feedback(reuse_fbk=feedback)
        pact_pan = feedback.position[0]
        pact_tilt = feedback.position[1]
        vact_pan = feedback.velocity[0]
        vact_tilt = feedback.velocity[1]

        # STORE ALL DATA: commands vs. actual
        Time.append(t)
        
        PAct_pan.append(pact_pan)
        PCmd_pan.append(pcmd_pan)
        VAct_pan.append(vact_pan)
        VCmd_pan.append(vcmd_pan)
        Error_vel_pan.append(vact_pan - vcmd_pan)
        Error_pos_pan.append(pact_pan - pcmd_pan)
        
        PAct_tilt.append(pact_tilt)
        PCmd_tilt.append(pcmd_tilt)
        VAct_tilt.append(vact_tilt)
        VCmd_tilt.append(vcmd_tilt)
        Error_vel_tilt.append(vact_tilt - vcmd_tilt)
        Error_pos_tilt.append(pact_tilt - pcmd_tilt)

        # ACQUIRE LOCK: 
        # 1) initialize pan/tilt angles needed to move camera to object
        # 2) get max/min tilt/pan for FOV
        # 3) see if reciveved new data (decision to spline to object depends on this) - then set to false in shared memory
        # 4) check newly recieved data with list of known objects
        # 5) share motors current tilt/pan angles so detector can normalize object angles relative to motor
        if shared.lock.acquire():
            
            object_pan_theta = shared.object_pan
            object_tilt_theta = shared.object_tilt
            
            max_pan = shared.max_pan
            min_pan = shared.min_pan
            max_tilt = shared.max_tilt
            min_tilt = shared.min_tilt
            
            new_data_bool = shared.new_data

            # for cluster graph: if got new data add it to the cluster
            #if shared.objects_data:
            if new_data_bool:
                if shared.objects_data:
                    historyofobjects.extend(shared.objects_data)
                # make a copy so don't screw up lock
                objects_data_copy = shared.objects_data.copy()
                # reset to empty each time
                shared.objects_data = []
                known_objects, interest_switch_bool, max_interest_index = known_objects_to_detected_objects_mapper(known_objects, objects_data_copy, dist_threshold, known_objects_index)

            shared.new_data = False
        
            shared.motorpan = pcmd_pan
            shared.motortilt = pcmd_tilt
            
            shared.lock.release()
    

        # MONITOR WHAT LETTER WAS PRESSED
        # a, b, c, d, e, z --> pointing & spline
        # t --> tracking & spline
        # s --> scanning & sine
        # q --> quit + graph
        letter = getch()
        
        if letter in pointing_keys:
           
            mode = Mode.POINTING
            traj = Traj.SPLINE
            
            if (letter == "l"):
                
                #pan
                pf_pan = pi/6
                vf_pan = 0
                p0_pan = pcmd_pan
                v0_pan = vcmd_pan
                
                #tilt
                pf_tilt = 0
                vf_tilt = 0
                p0_tilt = pcmd_tilt
                v0_tilt = vcmd_tilt
            
            if (letter == "d"):
    
                #pan
                pf_pan = 0
                vf_pan = 0
                p0_pan = pcmd_pan
                v0_pan = vcmd_pan
                
                #tilt
                pf_tilt = -pi/6
                vf_tilt = 0
                p0_tilt = pcmd_tilt
                v0_tilt = vcmd_tilt

            if (letter == "r"):

                #pan
                pf_pan = -pi/6
                vf_pan = 0
                p0_pan = pcmd_pan
                v0_pan = vcmd_pan
                
                #tilt
                pf_tilt = 0
                vf_tilt = 0
                p0_tilt = pcmd_tilt
                v0_tilt = vcmd_tilt
            
            if (letter == "u"):
                
                #pan
                pf_pan = 0
                vf_pan = 0
                p0_pan = pcmd_pan
                v0_pan = vcmd_pan
                
                #tilt
                pf_tilt = pi/6
                vf_tilt = 0
                p0_tilt = pcmd_tilt
                v0_tilt = vcmd_tilt
            
            if (letter == "c"):
                
                #pan
                pf_pan = 0
                vf_pan = 0
                p0_pan = pcmd_pan
                v0_pan = vcmd_pan
                
                #tilt
                pf_tilt = 0
                vf_tilt = 0
                p0_tilt = pcmd_tilt
                v0_tilt = vcmd_tilt
            
            #both
            t0 = t
            tf = t0 + max(move_time(p0_pan, pf_pan, vmax_pan, v0_pan, amax_pan), move_time(p0_tilt, pf_tilt, vmax_tilt, v0_tilt, amax_tilt))
            a_pan, b_pan, c_pan, d_pan = spline_params(t0, tf, p0_pan, pf_pan, v0_pan, vf_pan)
            a_tilt, b_tilt, c_tilt, d_tilt = spline_params(t0, tf, p0_tilt, pf_tilt, v0_tilt, vf_tilt)
        
        
        if (letter == "t"):
            mode = Mode.TRACKING
        
        
        if (letter == "s"):
            
            mode = Mode.SCANNING
            traj = Traj.SPLINE
            scan_finished = False
            historyofobjects = []
            known_objects = []
                
            #create spline to the start of the sine scanner pos/vel
                
            # pan 
            pf_pan = 0
            vf_pan = Apan * 2*pi * Npan / Tscan
            p0_pan = pcmd_pan
            v0_pan = vcmd_pan
                
            # tilt
            pf_tilt = center_tilt
            vf_tilt = Atilt * 2*pi * Ntilt / Tscan
            p0_tilt = pcmd_tilt
            v0_tilt = vcmd_tilt
                
            #both
            t0 = t
            tf = t0 + max(move_time(p0_pan, pf_pan, vmax_pan, v0_pan, amax_pan, vf_pan), move_time(p0_tilt, pf_tilt, vmax_tilt, v0_tilt, amax_tilt, vf_tilt))
            a_pan, b_pan, c_pan, d_pan = spline_params(t0, tf, p0_pan, pf_pan, v0_pan, vf_pan)
            a_tilt, b_tilt, c_tilt, d_tilt = spline_params(t0, tf, p0_tilt, pf_tilt, v0_tilt, vf_tilt)
            

        if (letter =="q"):
            
            # Create a plot of position and velocity, actual and command
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(8, 10))

            ax1.plot(Time, PAct_pan, color='blue', linestyle='-',  label='Act_pan')
            ax1.plot(Time, PCmd_pan, color='green', linestyle='--', label='Cmd_pan')
            ax2.plot(Time, VAct_pan, color='blue', linestyle='-',  label='Act_pan')
            ax2.plot(Time, VCmd_pan, color='green', linestyle='--', label='Cmd_pan')
            ax3.plot(Time, PAct_tilt, color='blue', linestyle='-',  label='Act_tilt')
            ax3.plot(Time, PCmd_tilt, color='green', linestyle='--', label='Cmd_tilt')
            ax4.plot(Time, VAct_tilt, color='blue', linestyle='-',  label='Act_tilt')
            ax4.plot(Time, VCmd_tilt, color='green', linestyle='--', label='Cmd_tilt')
            ax1.plot(Time, Error_pos_pan, color='red', linestyle='-',  label='PosErr_pan')
            ax2.plot(Time, Error_vel_pan, color='red', linestyle='-',  label='VelErr_pan')
            ax3.plot(Time, Error_pos_tilt, color='red', linestyle='-',  label='PosErr_tilt')
            ax4.plot(Time, Error_vel_tilt, color='red', linestyle='-',  label='VelErr_tilt')

            ax1.set_title('Pan Motor CMD vs ACT')
            ax1.set_ylabel('Position (rad)')
            ax2.set_ylabel('Velocity (rad/s)')
            ax2.set_xlabel('Time (s)')
            ax1.grid()
            ax2.grid()
            ax1.legend()
            ax2.legend()
            
            ax3.set_title('Tilt Motor CMD vs ACT')
            ax3.set_ylabel('Position (rad)')
            ax4.set_ylabel('Velocity (rad/s)')
            ax4.set_xlabel('Time (s)')
            ax3.grid()
            ax4.grid()
            ax3.legend()
            ax4.legend()
            
            # scatter plot for object clustering
            if historyofobjects:
                pan_data, tilt_data = zip(*historyofobjects)
                plt.figure()
                plt.scatter(pan_data, tilt_data, marker='x', color='r')
                plt.xlim(-1.7, 1.7)
                plt.ylim(-1.5, 1.0)
                plt.xlabel('Pan (rad)')
                plt.ylabel('Tilt (rad)')
                plt.title('Map of Detected Objects')
                plt.grid(True)
                plt.show() 
            plt.tight_layout()
            plt.show()
            break
            
        
        # MOVEMENT AND ACTION AFTER COMPLETION DEPENDING ON SET MODE

        if mode is Mode.POINTING:
                        
            # once get to location hold there
            if (t + dt > tf):
                phold_pan = pf_pan
                phold_tilt = pf_tilt
                traj = Traj.HOLD
                

        elif mode is Mode.TRACKING:
            
            # if no known objects
            if not known_objects:
                traj = Traj.HOLD
                phold_pan = pcmd_pan 
                phold_tilt = pcmd_tilt
                
                attention_span_over = False
                tracking_active = False
                known_objects_index = 0
                
            # if at least one known object
            else: 
                
                # check if any objects above confidence level thresh
                if valid_threshold(known_objects):
                    
                   # check that index isn't out of bounds due to removing objects
                    if known_objects_index >= len(known_objects):
                        known_objects_index = 0
                    
                    # MOVE 1
                    # MOVE TO LOCATION OF KNOWN OBJECT
                    if not attention_span_over:
                        
                        attention_span_over = True
                        tracking_active = False
                        
                        traj = Traj.SPLINE
                        
                        #pan
                        pf_pan = known_objects[known_objects_index][0]
                        vf_pan = 0
                        p0_pan = pcmd_pan
                        v0_pan = vcmd_pan
                                            
                        #tilt
                        pf_tilt = known_objects[known_objects_index][1]
                        vf_tilt = 0
                        p0_tilt = pcmd_tilt
                        v0_tilt = vcmd_tilt
                                
                        #both
                        t0 = t
                        tf = t0 + max(move_time(p0_pan, pf_pan, vmax_pan, v0_pan, amax_pan), move_time(p0_tilt, pf_tilt, vmax_tilt, v0_tilt, amax_tilt))
                        a_pan, b_pan, c_pan, d_pan = spline_params(t0, tf, p0_pan, pf_pan, v0_pan, vf_pan)
                        a_tilt, b_tilt, c_tilt, d_tilt = spline_params(t0, tf, p0_tilt, pf_tilt, v0_tilt, vf_tilt)
                    
                    # transition between move 1 and move 2
                    if not tracking_active:
                        if t + dt >= tf:
                            tracking_active = True
                        else:
                            pass
                        
                    # MOVE 2
                    # if get new data & movement complete from one object to the other --> move to new data (tracking)
                    if tracking_active:
                        
                        if new_data_bool:
                                        
                            traj = Traj.SPLINE

                            #pan
                            pf_pan = known_objects[known_objects_index][0] #object_pan_theta
                            vf_pan = 0
                            p0_pan = pcmd_pan
                            v0_pan = vcmd_pan
                                
                            #tilt
                            pf_tilt = known_objects[known_objects_index][1] #object_tilt_theta
                            vf_tilt = 0
                            p0_tilt = pcmd_tilt
                            v0_tilt = vcmd_tilt
                                                    
                            #both
                            t0 = t
                            tf = t0 + max(move_time(p0_pan, pf_pan, vmax_pan, v0_pan, amax_pan), move_time(p0_tilt, pf_tilt, vmax_tilt, v0_tilt, amax_tilt))
                            a_pan, b_pan, c_pan, d_pan = spline_params(t0, tf, p0_pan, pf_pan, v0_pan, vf_pan)
                            a_tilt, b_tilt, c_tilt, d_tilt = spline_params(t0, tf, p0_tilt, pf_tilt, v0_tilt, vf_tilt)
                        
                        elif (t + dt > tf):
                            traj = Traj.HOLD
                            phold_pan = pcmd_pan 
                            phold_tilt = pcmd_tilt
                    
                        
                    # if no object in location (confidence level too low) or interest level decreased ("bored")
                    if known_objects[known_objects_index][2] < C_threshold: 
                        known_objects_index += 1
                        known_objects_index = known_objects_index % len(known_objects)
                        attention_span_over = False
                        tracking_active = False
                    if interest_switch_bool:
                        known_objects_index = max_interest_index
                        interest_switch_bool = False

                else:
                    traj = Traj.HOLD
                    phold_pan = pcmd_pan 
                    phold_tilt = pcmd_tilt
                
                    attention_span_over = False
                    tracking_active = False
                    known_objects_index = 0
                
               
        elif mode is Mode.SCANNING:
            
            # time final can represent two things: first end of spline, then end of scan
            if t + dt > tf: 
            
                if scan_finished:
                    mode = Mode.TRACKING
                
                t0 = t
                tf = Tscan + t0
                traj = Traj.SINE
                scan_finished = True
    
        else:
            raise ValueError(f'Bad Operation mode {mode}')
            
        # Advance the time.
        t += dt


if __name__ == "__main__":
    controller(None)