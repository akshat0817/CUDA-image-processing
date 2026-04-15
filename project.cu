#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;

// SCALE 
__global__ void scaleKernel(unsigned char* input, unsigned char* output, int w, int h, int scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int newW = w * scale;
    int newH = h * scale;

    if (x < newW && y < newH)
    {
        int srcX = x / scale;
        int srcY = y / scale;

        output[y * newW + x] = input[srcY * w + srcX];
    }
}

// TRANSLATION 
__global__ void translateKernel(unsigned char* input, unsigned char* output, int w, int h, int tx, int ty)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int newX = x + tx;
        int newY = y + ty;

        if (newX < w && newY < h)
            output[newY * w + newX] = input[y * w + x];
    }
}

// ROTATION 
__global__ void rotateKernel(unsigned char* input, unsigned char* output, int w, int h, float angle)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int cx = w / 2;
        int cy = h / 2;

        float rad = angle * 3.14159 / 180.0;

        int srcX = (int)((x - cx) * cos(rad) + (y - cy) * sin(rad) + cx);
        int srcY = (int)(-(x - cx) * sin(rad) + (y - cy) * cos(rad) + cy);

        if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h)
            output[y * w + x] = input[srcY * w + srcX];
        else
            output[y * w + x] = 0;
    }
}

// AFFINE
__global__ void affineKernel(unsigned char* input, unsigned char* output, int w, int h,
                             float a, float b, float c, float d, float tx, float ty)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h)
    {
        int srcX = (int)(a * x + b * y + tx);
        int srcY = (int)(c * x + d * y + ty);

        if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h)
            output[y * w + x] = input[srcY * w + srcX];
        else
            output[y * w + x] = 0;
    }
}

int main()
{
    Mat img = imread("messi5.jpg", IMREAD_GRAYSCALE);
    if (img.empty())
    {
        printf("Image not found!\n");
        return -1;
    }

    int w = img.cols;
    int h = img.rows;

    int choice;
    printf("\n1. Scaling\n2. Translation\n3. Rotation\n4. Affine\nEnter choice: ");
    scanf("%d", &choice);

    unsigned char *d_input, *d_output;

    cudaMalloc(&d_input, w * h);
    cudaMemcpy(d_input, img.data, w * h, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);

    Mat output;

    if (choice == 1)
    {
        int scale = 2;
        int newW = w * scale;
        int newH = h * scale;

        cudaMalloc(&d_output, newW * newH);

        dim3 blocks2((newW + 15) / 16, (newH + 15) / 16);

        scaleKernel<<<blocks2, threads>>>(d_input, d_output, w, h, scale);
        cudaDeviceSynchronize();  

        output = Mat(newH, newW, CV_8UC1);
        cudaMemcpy(output.data, d_output, newW * newH, cudaMemcpyDeviceToHost);

        imwrite("scaled.jpg", output);
    }

    else if (choice == 2)
    {
        int tx = 100, ty = 50;

        cudaMalloc(&d_output, w * h);
        cudaMemset(d_output, 0, w * h);

        translateKernel<<<blocks, threads>>>(d_input, d_output, w, h, tx, ty);
        cudaDeviceSynchronize();  

        output = Mat(h, w, CV_8UC1);
        cudaMemcpy(output.data, d_output, w * h, cudaMemcpyDeviceToHost);

        imwrite("translated.jpg", output);
    }

    else if (choice == 3)
    {
        cudaMalloc(&d_output, w * h);

        rotateKernel<<<blocks, threads>>>(d_input, d_output, w, h, 90.0);
        cudaDeviceSynchronize(); 

        output = Mat(h, w, CV_8UC1);
        cudaMemcpy(output.data, d_output, w * h, cudaMemcpyDeviceToHost);

        imwrite("rotated.jpg", output);
    }

    else if (choice == 4)
    {
        cudaMalloc(&d_output, w * h);

        float a = 1, b = 0.2, c = 0.2, d = 1, tx = 10, ty = 20;

        affineKernel<<<blocks, threads>>>(d_input, d_output, w, h, a, b, c, d, tx, ty);
        cudaDeviceSynchronize();  

        output = Mat(h, w, CV_8UC1);
        cudaMemcpy(output.data, d_output, w * h, cudaMemcpyDeviceToHost);

        imwrite("affine.jpg", output);
    }

    cudaDeviceSynchronize();  

    cudaFree(d_input);
    cudaFree(d_output);

    printf("Done! Check output image.\n");

    return 0;
}