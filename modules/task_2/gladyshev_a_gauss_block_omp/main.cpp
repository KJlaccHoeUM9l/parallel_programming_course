// Copyright 2019 Gladyshev Alexey
#include <omp.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>

typedef struct { int r; int g; int b; } myColor;
typedef struct { int x; int y; } myPoint;
typedef struct { myPoint start; myPoint finish; } myIndex;

int clamp(int value, int min, int max);
float* createGaussianKernel(int radius, float sigma);
myColor calculateNewPixelColor(myColor* sourceImage, int width, int height,
                             float* kernel, int radius, int i, int j);
void createRandomPicture(myColor* arrayImage, int width, int height);

myPoint getDecomposition(int n) {
    //  n = k * m
    // |k - m| -> min
    myPoint result;
    int m, k;

    m = static_cast<int>(sqrt(n));
    while (n % m != 0) m--;
    k = n / m;

    result.x = std::max(k, m);
    result.y = std::min(k, m);
    return result;
}

int* getLength(int length, int qElements) {
    int eachLength = length / qElements;
    int tailLength = length % qElements;
    int* lengthArray = new int[qElements];

    for (int i = 0; i < qElements; i++)
        lengthArray[i] = eachLength;

    int i = 0;
    while (tailLength != 0) {
        lengthArray[i % qElements]++;
        tailLength--;
        i++;
    }

    return lengthArray;
}

void getIndexes(myIndex* indexArray, int width, int height, int threads) {
    myPoint decomposition = getDecomposition(threads);
    int* heightLength = getLength(height, decomposition.x);
    int* widthLength = getLength(width, decomposition.y);

    int currentHeight = 0;
    for (int i = 0; i < decomposition.x; i++) {
        int currentWidth = 0;
        for (int j = 0; j < decomposition.y; j++) {
            myPoint start, finish;
            start.x = currentWidth;
            start.y = currentHeight;

            finish.x = currentWidth + widthLength[j];
            finish.y = currentHeight + heightLength[i];

            indexArray[i * decomposition.y + j].start = start;
            indexArray[i * decomposition.y + j].finish = finish;

            currentWidth += widthLength[j];
        }
        currentHeight += heightLength[i];
    }

    delete[]heightLength;
    delete[]widthLength;
}

void ompProcessImage_block(myColor* sourceImage, myColor* resultImage, int width, int height,
                           float* kernel, int kernelRadius, myIndex* indexArray) {
#pragma omp parallel
        {
            int myid = omp_get_thread_num();
            int startHeight = indexArray[myid].start.y;
            int finishHeight = indexArray[myid].finish.y;
            int startWidth = indexArray[myid].start.x;
            int finishWidth = indexArray[myid].finish.x;

            for (int i = startHeight; i < finishHeight; i++)
                for (int j = startWidth; j < finishWidth; j++)
                    resultImage[i * width + j] = calculateNewPixelColor(sourceImage, width, height, kernel, kernelRadius, i, j);
        }
}

void processImage(myColor* sourceImage, myColor* resultImage, int width, int height, float* kernel, int kernelRadius) {
    //Обход и заполнение всех пикселей
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            resultImage[i * width + j] = calculateNewPixelColor(sourceImage, width, height, kernel, kernelRadius, i, j);
}

int check(myColor* first, myColor* second, int width, int height) {
    int result = 1;

    for (int i = 0; i < width * height; i++)
        if (first[i].r != second[i].r || first[i].g != second[i].g || first[i].b != second[i].b) {
            result = 0;
            break;
        }

    return result;
}

int main() {
    srand(static_cast<unsigned int>(time(NULL)));
    int width = 3123, height = 4967;
    int radius = 1;
    float sigma = 6;
    int threads = 4;
    double start, linearTotal, ompTotal;
    float* kernel = createGaussianKernel(radius, sigma);
    
    myColor* sourceArrayImage = new myColor[width * height];
    myColor* resultArrayImage = new myColor[width * height];
    myColor* ompResultArrayImage = new myColor[width * height];

    createRandomPicture(sourceArrayImage, width, height);

    start = omp_get_wtime();
    processImage(sourceArrayImage, resultArrayImage, width, height, kernel, radius);
    linearTotal = omp_get_wtime() - start;

    omp_set_num_threads(threads);
    myIndex* indexArray = new myIndex[threads];
    getIndexes(indexArray, width, height, threads);
    start = omp_get_wtime();
    ompProcessImage_block(sourceArrayImage, ompResultArrayImage, width, height, kernel, radius, indexArray);
    ompTotal = omp_get_wtime() - start;

    if (!check(resultArrayImage, ompResultArrayImage, width, height)) {
        std::cout << std::endl << "The results of serial and parallel algorithms are NOT identical. Check your code.";
    }
    else {
        std::cout << std::endl << "(width, height) = (" << width << ", " << height << ")";
        std::cout << std::endl << "threads =          " << threads;
        std::cout << std::endl << "Kernel radius =    " << radius;
        std::cout << std::endl << "Filtering (non-parallel):       " << linearTotal * 1000 << " (ms)";
        std::cout << std::endl << "Filtering (parallel-block):     " << ompTotal * 1000 << " (ms)";
        std::cout << std::endl << "Acceleration (block):           " << linearTotal / ompTotal << std::endl;
    }
    

    delete[]sourceArrayImage;
    delete[]resultArrayImage;
    delete[]ompResultArrayImage;
    delete[]kernel;
    delete[]indexArray;

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
            kernel[(i + radius) * size + (j + radius)] =
                static_cast<float>(
                    (std::exp(-(i * i + j * j) / (sigma * sigma))));
            norm += kernel[(i + radius) * size + (j + radius)];
        }

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            kernel[i * size + j] /= norm;

    return kernel;
}

myColor calculateNewPixelColor(myColor* sourceImage, int width, int height,
                             float* kernel, int radius, int i, int j) {
    float resultR = 0;
    float resultG = 0;
    float resultB = 0;

    for (int l = -radius; l <= radius; l++) {
        for (int k = -radius; k <= radius; k++) {
            int idX = clamp(i + l, 0, height - 1);
            int idY = clamp(j + k, 0, width - 1);

            myColor neighborColor = sourceImage[idX * width + idY];

            resultR += neighborColor.r *
                kernel[(k + radius) * radius + (l + radius)];
            resultG += neighborColor.g *
                kernel[(k + radius) * radius + (l + radius)];
            resultB += neighborColor.b *
                kernel[(k + radius) * radius + (l + radius)];
        }
    }

    myColor result;
    result.r = clamp(static_cast<int>(resultR), 0, 255);
    result.g = clamp(static_cast<int>(resultG), 0, 255);
    result.b = clamp(static_cast<int>(resultB), 0, 255);
    return result;
}

void createRandomPicture(myColor* arrayImage, int width, int height) {
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++) {
            arrayImage[i * width + j].r = std::rand() % 256;
            arrayImage[i * width + j].g = std::rand() % 256;
            arrayImage[i * width + j].b = std::rand() % 256;
        }
}
