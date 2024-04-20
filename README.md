# Binocular Detection and Ranging System Intro

`Binocular Detection and Ranging System` (2022), 'BDRS' for short, is a vision system for a robot. BDRS aims to detect target objects and measure the distance from the target object to the camera. The control system and hardware implementation of the robot are handled by two master's students and one doctoral student. YOLOv5 was adopted as the detection method in BDRS. Finally, the accuracy of ranging within 15cm reached 90%, and the accuracy of identifying objects reached 80%. 
You can run the `camera.py` to see the demonstration.

Ps: I have just changed it to yolov8 and trained a new model for the demonstration because I deleted the yolov5 tool after yolov8 came out, but the code calling yolov5 is still in the comments. Considering that there may not be a camera to use, I used a pair of binocular images for training as the image source, and it should now be able to perform a basic demonstration!

## Research thoughts
**Preliminary work**
+ Checking the equipment 
    + The binocular camera was smaller than my finger because it needed to fit into the end of the execution. It therefore had a particularly poor imaging effect, the two images were not even parallel, so it was necessary to `rectify the camera`. However, the opposite was also beneficial, with the camera on the lens, `the accuracy of 3d ranging increases` as it got closer to the target.
+ Trying and practicing
    + At that time, out of my lack of confidence in myself, I even used a ruler to measure the depth distance during the implementation of the ranging function for a while(lol), because the ranging was the horizontal and vertical displacement calculated by the depth first.
 
**Algorithm development**
+ Basic idea
    + The functions in BDRS cannot be realized synchronously, because it was a gradual process from rectification to detection to ranging. After considering the possible impact of image quality on detection, I decided to use rectified images for model training.
    + BDRS needed to transmit the analyzed data to the operating system by a socket,while waiting for instructions from the operating system at any time.


## Methods

**Img preprocessing**

+ Rectifying images
    + Remapped by calculating the mapping matrix and reprojection matrix
        + Using the parameters of the camera (obtained through Matlab calibration toolbox).


**Detection**
+ Object detection
    + Using model to predict objects.
    + Converting results into objects that are easy to understand.

``` python
        for xyxycc in xyxyccs:
            if xyxycc[4] < 0.5 :#for demonstration
                continue  # confidence
            x1 = int(xyxycc[0])
            y1 = int(xyxycc[1])
            x2 = int(xyxycc[2])
            y2 = int(xyxycc[3])
            confidence = float(xyxycc[4])
            classID = int(xyxycc[5])
            class_name = LABELS[int(xyxycc[5])]
            #DetectResult is a class used to save the result
            detect_results.append(DetectResult(
                class_name, confidence, [x1, y1, x2, y2]))
```
+ Corner detection
    +  Useing A variety of methods to determine whether there are corner points, such as key point extraction, simplified line intersection extraction, corner matching.

**Ranging**

+ Matching
    + Iterate over the two image lists, matching based on the difference and category of the midpoint coordinates of each identifier box.

+ Calculating
    + For the matched detection boxes, their `pixel coordinate difference` in the left and right images is calculated, called parallax, which is related to the `depth of the object` from the camera.   Multiplying the parallax by a scaling factor that is related to the focal length of the camera(obtained by camera calibration) to calculate the three-dimensional coordinates of an object (X, Y, Z)（the unit is millimeters）.
``` python
        for index_l, result_l in enumerate(self.detect_results_left):
            for index_r, result_r in enumerate(self.detect_results_right):
                if the type is not the same:
                    continue  # skip the long parts of code
                boxl = result_l.box
                boxr = result_r.box
                if conditions: # skip the long parts of code
                    matched_index[index_l] = index_r
                    break
        # locate the matched detection boxes
        locate_results = []
        for index_l in range(len(self.detect_results_left)):  # 
            if index_l not in matched_index:  # if the left result is not matched, failed
               locate_results.append([10000, 10000, 10000])  # failed
            else:
                boxl = self.detect_results_left[index_l].box
                boxr = self.detect_results_right[matched_index[index_l]].box
                # based on the principle of binocular ranging, calculate the coordinates of the X direction edges and the center coordinates
                pix_x_l = (boxl[0]+boxl[2])/2  # middle of the left img
                pix_y_l = boxl[3]#)/2    #as the operation system team asked, change the y to the bottom of the box
                pix_x_r = (boxr[0]+boxr[2])/2  
                pix_y_r = (boxr[1]+boxr[3])/2   
                X, Y, Z = self.StereoLocating(pix_x_l, pix_y_l, pix_x_r)
                locate_results.append([X, Y, Z])
```



## Result Analysis
+ Algorithm accuracy
    + The accuracy of ranging within 15cm reached 90%, and the accuracy of identifying objects reached 80%. 
    
+ Algorithm efficiency
    + The efficiency was 8 fps on the RTX3070.
