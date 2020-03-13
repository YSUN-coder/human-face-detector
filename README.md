Matching of Face Videos using Covariance Matrices
=================================================
The videos for Input Data is a directory in python file named images. You could run the python file video_face_landmarks.py to get the results.
＜/br＞
# Proposed Method
In this work, I applied a pre-trained HOG + Linear SVM object detector for the task of face detection, then used 68 landmarks to visualize the human face as it shows in Figure 1. The result to detect facial landmarks is in realtime percisely as shown in Figure 2.
<div align="center">
<img src="https://raw.github.com/YSUN-coder/human-face-detector/master/report_resource/landmark.png"/>
<center>Figure 1:Visualizing each of the 68 facial coordinate points </center>
</div>
<div align="center">
<img src="https://raw.github.com/YSUN-coder/human-face-detector/master/report_resource/landmark_result.png"/>
<center>Figure 2:  Landmark Result </center>
</div>




#To run
1. add a directory named 'images' and put your image(.jpg/) and video(.mp4/) into the directory
2. python video_face_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --videofile ${images/video.mp4}
(Please change ${} content on command line to your own file directory or name according to example in it.)