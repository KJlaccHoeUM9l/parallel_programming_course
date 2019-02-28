// Copyright 2019 Gladyshev Alexey
#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

typedef struct { int r; int g; int b; } color;
typedef struct { int x; int y; } point;
typedef struct { point start; point finish; } index;

int clamp(int value, int min, int max);
float* createGaussianKernel(int radius, float sigma);
color calculateNewPixelColor(color* sourceImage, int width, int height, float* kernel, int radius, int i, int j);
void processImage(color* sourceImage, color* resultImage, int width, int height, float* kernel, int kernelRadius);
void createRandomPicture(color* arrayImage, int width, int height);

int main() {
    srand(static_cast<unsigned int>(time(NULL)));
    int width = 3123, height = 4967;
    int radius = 1;
    float sigma = 6;
    float* kernel = createGaussianKernel(radius, sigma);
    color* sourceArrayImage = new color[width * height];
    color* resultArrayImage = new color[width * height];
    double start, linearTotal;

    createRandomPicture(sourceArrayImage, width, height);

    start = omp_get_wtime();
    processImage(sourceArrayImage, resultArrayImage, width, height, kernel, radius);
    linearTotal = omp_get_wtime() - start;

    std::cout << std::endl << "(width, height) = (" << width << ", " << height << ")";
    std::cout << std::endl << "Kernel radius =    " << radius;
    std::cout << std::endl << "Filtering (non-parallel):       " << linearTotal * 1000 << " (ms)" << std::endl;

    delete[]sourceArrayImage;
    delete[]resultArrayImage;
    delete[]kernel;

    return 0;
}

int clamp(int value, int min, int max) {
    if (value < min)    return min;
    if (value > max)    return max;
    return value;
}

float* createGaussianKernel(int radius, float sigma) {
    int size = 2 * radius + 1;
    float* kernel = new float[size * size];
    float norm = 0;

    for (int i = -radius; i <= radius; i++)
        for (int j = -radius; j <= radius; j++) {
            kernel[(i + radius) * size + (j + radius)] = static_cast<float>((exp(-(i * i + j * j) / (sigma * sigma))));
            norm += kernel[(i + radius) * size + (j + radius)];
        }

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            kernel[i * size + j] /= norm;

    return kernel;
}

color calculateNewPixelColor(color* sourceImage, int width, int height, float* kernel, int radius, int i, int j) {
    float resultR = 0;
    float resultG = 0;
    float resultB = 0;

    for (int l = -radius; l <= radius; l++) {
        for (int k = -radius; k <= radius; k++) {
            int idX = clamp(i + l, 0, height - 1);
            int idY = clamp(j + k, 0, width - 1);

            color neighborColor = sourceImage[idX * width + idY];

            resultR += neighborColor.r * kernel[(k + radius) * radius + (l + radius)];
            resultG += neighborColor.g * kernel[(k + radius) * radius + (l + radius)];
            resultB += neighborColor.b * kernel[(k + radius) * radius + (l + radius)];
        }
    }

    color result;
    result.r = clamp(static_cast<int>(resultR), 0, 255);
    result.g = clamp(static_cast<int>(resultG), 0, 255);
    result.b = clamp(static_cast<int>(resultB), 0, 255);
    return result;
}

void processImage(color* sourceImage, color* resultImage, int width, int height, float* kernel, int kernelRadius) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            resultImage[i * width + j] = calculateNewPixelColor(sourceImage, width, height, kernel, kernelRadius, i, j);
}

void createRandomPicture(color* arrayImage, int width, int height) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            arrayImage[i * width + j].r = std::rand() % 256;
            arrayImage[i * width + j].g = std::rand() % 256;
            arrayImage[i * width + j].b = std::rand() % 256;
        }
}
