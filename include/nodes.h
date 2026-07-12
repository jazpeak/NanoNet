#pragma once

#include "graph.h"
#include <thread>
#include "tensor.h"
#include "ops.h"
using namespace std;



class MatMulNode : public Node{
private:
	const Tensor& A;
    const Tensor& B;
    Tensor& C;
    bool transB;
public:
    MatMulNode(const Tensor& A, const Tensor& B, Tensor& C,bool transB = false) : A(A), B(B), C(C) ,transB(transB){}

    void forward() override {
    	int threads = thread::hardware_concurrency();
        if (threads <= 0) threads = 1;
        matmul_parallel(A, B, C, threads, transB);
    }
};

class AddNode : public Node{
private:
	const Tensor& A;
    const Tensor& B;
    Tensor& C;

public:
    AddNode(const Tensor& A, const Tensor& B, Tensor& C) : A(A), B(B), C(C) {}

    void forward() override {
        add(A, B, C);
    }
};
class ReluNode : public Node{
private:
    const Tensor& A;
    Tensor& C;
public:
    ReluNode(const Tensor& A, Tensor& C) : A(A),C(C) {}

    void forward() override {
        relu(A, C);
    }
};

class SoftmaxNode : public Node {
private:
    const Tensor& A;
    Tensor& C;
public:
    SoftmaxNode(const Tensor& A, Tensor& C) : A(A), C(C) {}

    void forward() override {
        softmax(A, C);
    }
};

class FlattenNode : public Node {
private:
    const Tensor& A;
    Tensor& C;

public:
    FlattenNode(const Tensor& A, Tensor& C)
        : A(A), C(C) {}

    void forward() override {
        int total = A.rows() * A.cols();
        assert(C.rows() * C.cols() == total);

        copy(A.data(), A.data() + total, C.data());
    }
};
