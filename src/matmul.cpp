#include "ops.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <thread>
using namespace std;

void matmul(const Tensor& A, const Tensor& B , Tensor& C){
	assert(A.cols() == B.rows());
    assert(C.rows() == A.rows());
    assert(C.cols() == B.cols());

    int m = A.rows();
    int k = A.cols();
    /* 
       If transB is not supported in simple matmul, assume false. 
       But current signature doesn't take transB?
       Wait, simple matmul signature is `void matmul(const Tensor& A, const Tensor& B , Tensor& C)`. 
       It implicitly assumes transB=false based on assertions A.cols() == B.rows().
       I should probably leave it or check if it is used.
       MatMulNode uses matmul_parallel.
    */
    int n = B.cols();


    const float* a = A.data();
    const float* b = B.data();
    float * c = C.data();
    fill(c, c + m * n, 0.0f);



    for( int i=0; i < m; i++){
    	for( int l=0;l<k;l++){
    		float ail= a[i*k + l];
    		for ( int j=0;j<n;j++){
    			c[i*n+j]+= ail * b[n*l + j];
    		}
    	}
    }

}

static void matmul_worker(const float* a,
                          const float* b,
                          float* c,
                          int m, int k, int n,
                          int row_start,
                          int row_end,bool transB) {

	if(!transB){
    for (int i = row_start; i < row_end; ++i) {
        for (int p = 0; p < k; ++p) {
            float a_ip = a[i * k + p];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}else{
	for (int i = row_start; i < row_end; ++i) {
        for (int p = 0; p < k; ++p) {
            float a_ip = a[i * k + p];
            for (int j = 0; j < n; ++j) {
                c[i * n + j] += a_ip * b[j * k + p];
            }
        }
    }
}
}


void matmul_parallel(const Tensor& A, const Tensor& B, Tensor& C, int num_threads,bool transB){
    int m = A.rows();
    int k = A.cols();
    int n = transB ? B.rows() : B.cols();

    if (transB) {
        assert(A.cols() == B.cols());
    } else {
        assert(A.cols() == B.rows());
    }
    assert(C.rows() == A.rows());
    assert(C.cols() == n);

    const float* a = A.data();
    const float* b = B.data();
    float* c = C.data();

    fill(c, c + m * n, 0.0f);

    num_threads = min(num_threads, m);
    int chunk = (m + num_threads - 1) / num_threads;

    vector<thread> threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = min(start + chunk, m);

        if (start >= end) break;

        threads.emplace_back(
            matmul_worker,
            a, b, c,
            m, k, n,
            start, end,transB
        );
    }

    for (auto& th : threads) {
        th.join();
    }

}