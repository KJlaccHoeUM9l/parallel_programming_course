#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include <iostream>

using namespace tbb;
using namespace std;

const size_t L = 150;
const size_t M = 225;
const size_t N = 300;

class MatrixMultiplyBody2D {
    float(*my_a)[L];
    float(*my_b)[N];
    float(*my_c)[N];
public:
    void operator()(const blocked_range2d<size_t>& r) const {
        float(*a)[L] = my_a;
        float(*b)[N] = my_b;
        float(*c)[N] = my_c;
        for (size_t i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (size_t j = r.cols().begin(); j != r.cols().end(); ++j) {
                float sum = 0;
                for (size_t k = 0; k<L; ++k)
                    sum += a[i][k] * b[k][j];
                c[i][j] = sum;
            }
        }
    }
    MatrixMultiplyBody2D(float c[M][N], float a[M][L], float b[L][N]) :
        my_a(a), my_b(b), my_c(c)
    {}
};

void ParallelMatrixMultiply(float c[M][N], float a[M][L], float b[L][N]) {
    parallel_for(blocked_range2d<size_t>(0, M, 16, 0, N, 32),
        MatrixMultiplyBody2D(c, a, b));
}
/*
int main() {
    float c[M][N], a[M][L], b[L][N];

    for (int i = 0; i < M; i++)
        for (int j = 0; j < L; j++)
            a[i][j] = (float)(i + j) / 100;
    for (int i = 0; i < L; i++)
        for (int j = 0; j < N; j++)
            b[i][j] = (float)(i + j) / 100;

    ParallelMatrixMultiply(c, a, b);


    cout << "hello world" << endl;
    return 0;
}*/