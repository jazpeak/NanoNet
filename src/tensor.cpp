#include "tensor.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;



Tensor::Tensor(int rows, int cols): rows_(rows), cols_(cols),data_(rows*cols,0.0f){}

int Tensor::rows() const{
	return rows_;
}

int Tensor::cols() const{
	return cols_;
}

float* Tensor::data() {
	return data_.data();
}

const float* Tensor::data() const{
	return data_.data();
}

float& Tensor::at(int i , int j){
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
	return data_[i*cols_ + j];
}

const float& Tensor::at(int i, int j) const {
    assert(i >= 0 && i < rows_);
    assert(j >= 0 && j < cols_);
    return data_[i * cols_ + j];
}

bool Tensor::load_from_file(const string& path){
	ifstream in(path);

	if (!in) {
        cerr << "Failed to open " << path << "\n";
        return false;
    }

    int r, c;
    in >> r >> c;

    if (r != rows_ || c != cols_) {
        cerr << "Shape mismatch in " << path << "\n";
        return false;
    }


    for (int i = 0; i < rows_ * cols_; ++i) {
        in >> data_[i];
    }
    return true;

}