# Final year project of BSc: development of multi-person markerless motion capture system

My goal was to create an application that is capable of estimating the three-dimensional poses
of multiple people in real time. At the beginning of my work, I did a research covering the
literature of this subject. I studied the operation of the neural networks for pose estimation. I
looked at several optical motion capture systems. After studying these, I created my own system
design. I divided it into two pieces, the hardware design and the software design.

The result of the hardware design was a camera system consisting of four cameras. Furthermore,
I chose a so-called inference device that runs the neural network. This allowed me to
make my computer independent of the pose estimation speed.

Then I started developing the software. I needed to implement two important functions,
the calibration of the camera system and the conversion of the position estimation result into
a three-dimensional one using the calibrated system. During the calibration I performed the
calibration of the internal parameters of the cameras and the calibration of the position of the
cameras. Then I implemented the reconstruction calculations. Here, as a first step, I determine,
based on the spatial positioning of the pose estimates, which belong to the same person. Once
these were identified, I calculated the three-dimensional poses by triangulation. I created the
software using the application development framework called PyQt.

After the application was created, I tested its functionality. I examined two important
factors, the accuracy of the calibration and the position estimation. Based on the tests, I have
come to the conclusion that the calibration can be performed with much greater accuracy than
the accuracy of the pose estimator, so if we wanted to improve the system further, it could be
achieved by improving the processing of the position estimator output.

In my opinion, the greatest potential of the completed system is interaction with the virtual
reality. Because position estimation is made with image processing, there is no need for wearable
sensors, resulting in a much more user-friendly experience. By improving the reconstruction of
pose estimates, this could be achieved in the near future.
