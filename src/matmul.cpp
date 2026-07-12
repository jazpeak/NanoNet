#include "ops.h"
#include "thread_pool.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <thread>
#include <atomic>
using namespace std;

void matmul(const Tensor& A, const Tensor& B , Tensor& C){
	assert(A.cols() == B.rows());
    assert(C.rows() == A.rows());
    assert(C.cols() == B.cols());

    int m = A.rows();
    int k = A.cols();

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
    const int BLOCK_SIZE = 64;

	if(!transB){
        for (int i = row_start; i < row_end; i += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                for (int j = 0; j < n; j += BLOCK_SIZE) {
                    int i_end = min(i + BLOCK_SIZE, row_end);
                    int p_end = min(p + BLOCK_SIZE, k);
                    int j_end = min(j + BLOCK_SIZE, n);
                    for (int ii = i; ii < i_end; ++ii) {
                        for (int pp = p; pp < p_end; ++pp) {
                            float a_ip = a[ii * k + pp];
                            for (int jj = j; jj < j_end; ++jj) {
                                c[ii * n + jj] += a_ip * b[pp * n + jj];
                            }
                        }
                    }
                }
            }
        }
    }else{
        for (int i = row_start; i < row_end; i += BLOCK_SIZE) {
            for (int p = 0; p < k; p += BLOCK_SIZE) {
                for (int j = 0; j < n; j += BLOCK_SIZE) {
                    int i_end = min(i + BLOCK_SIZE, row_end);
                    int p_end = min(p + BLOCK_SIZE, k);
                    int j_end = min(j + BLOCK_SIZE, n);
                    for (int ii = i; ii < i_end; ++ii) {
                        for (int pp = p; pp < p_end; ++pp) {
                            float a_ip = a[ii * k + pp];
                            for (int jj = j; jj < j_end; ++jj) {
                                c[ii * n + jj] += a_ip * b[jj * k + pp];
                            }
                        }
                    }
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

    atomic<int> completed(0);
    int tasks_submitted = 0;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk;
        int end = min(start + chunk, m);

        if (start >= end) break;

        tasks_submitted++;
        get_thread_pool().enqueue([=, &completed]() {
            matmul_worker(a, b, c, m, k, n, start, end, transB);
            completed++;
        });
    }

    while (completed < tasks_submitted) {
        this_thread::yield();
    }
}