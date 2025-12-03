**Peg In Hole Task**

The above code utilizes the Gelsight-Mini, a vision based tactile sensor, in order to help robotic systems in insertion tasks.
Image processing on the Gelsight-Mini stream allowed us to view a vector field hilighting the distortion on the surface of the sensor, which corresponds to interruptions in the insertion task.
Using Helmholtz Decomposition, we were able to split the vector field into a divergence free component(Solenoidal) and curl free component(Potential) and an Harmonic Field. 
It was found that using the contact force vector described by the harmonic field alongside the vector concerning the lever arm, an axis of rotation was found such that an robotic system, 
such as the UR_5e robotic arm, would be able to rotate on by some theta in order to avoid the contact and complete the insertion task
