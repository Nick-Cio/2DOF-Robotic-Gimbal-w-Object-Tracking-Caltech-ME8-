# 2DOF-Robotic-Gimbal-w-Object-Tracking-Caltech-ME8-
Final code following completion of Caltech ME8 class. Code got pretty messy by the end of the project.
Split into 3 files: detector, controller, runner.
  - detector: object detection
  - controller: robot movement
  - runner: manages multi-threaded execution, running 'detector' and 'controller' in parallel & stores lock-protected data 
