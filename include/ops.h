#pragma once
#include "tensor.h"

void matmul(const Tensor& A,
            const Tensor& B,
            Tensor& C);


void matmul_parallel(const Tensor& A,
                     const Tensor& B,
                     Tensor& C,
                     int num_threads,
                     bool transB = false);
void add(const Tensor& A, const Tensor& B, Tensor& C);
void relu(const Tensor& A, Tensor& C);
