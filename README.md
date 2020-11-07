# Hand signs detection and hand tracking

Python program demonstrates hand signs detection
and hand tracking. Hopefully this script can help in research and 
development of assistive technology for Autistic and Deaf-mute 
people.

Principle. Hand geometry model rather than image set trained one, 
is projected and correlated with sample Image. Natively 
accelerated by graphics card without direct programming. 

Install dependencies:
- *`pip install opencv-python PyOpenGL PyOpenGL_accelerate pygame numpy imutils Pillow==6.1.0`*

Clone and run:
- *`git clone https://github.com/mmilann/hand_project.git`*
- *`cd hand_project`*
- *`python hand_project.py`*

Tune hand model with input arguments:
 - *`python hand_project.py [lenRatio widthRatio thickRatio]`*
    - lenRatio: fingers_length/palm_length ratio
    - widthRatio: palm_width/palm_length ratio
    - thickRatio: fingers_thickness/palm_thickness
 - example:
    - *`python hand_project.py 1.05 0.8 0.95`*
    
![picture alt](https://github.com/mmilann/hand_project/blob/main/Sign_projections.jpg)
