# CUDA-image-processing
# CUDA Image Processing Project

##  Description
This project implements geometric transformations using CUDA.

##  Features
- Scaling (2x)
- Translation (tx=100, ty=50)
- Rotation (90° clockwise)
- Affine Transformation
Can be configured otherwise as well

##  Technologies Used
- CUDA (GPU Programming)
- OpenCV
- C++

##  Concept
Each pixel is processed in parallel using CUDA threads.

##  How to Run
1. Compile:
   nvcc project.cu -o project `pkg-config --cflags --libs opencv4`

2. Run:
   ./project

3. Choose operation:
   1 - Scaling
   2 - Translation
   3 - Rotation
   4 - Affine



