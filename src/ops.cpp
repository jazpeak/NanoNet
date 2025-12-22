#include "ops.h"
#include <cassert>
#include <algorithm>
using namespace std;
void add(const Tensor& A, const Tensor& B, Tensor& C){
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());
    assert(C.rows() == A.rows());
    assert(C.cols() == A.cols());

    int size = A.rows() * A.cols();
    const float* a = A.data();
    const float *b = B.data();
    float * c = C.data();
    for( int i=0;i<size;i++){
    	c[i]=a[i]+b[i];
    }
}


void relu(const Tensor& A, Tensor& C) {
    assert(C.rows() == A.rows());
    assert(C.cols() == A.cols());

    int size = A.rows() * A.cols();

    const float* a = A.data();
    float* c = C.data();

    for (int i = 0; i < size; ++i) {
        c[i] = max(0.0f, a[i]);
    }
}
