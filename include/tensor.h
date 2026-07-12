#pragma once


#include <vector>
#include <string>

#include <stddef.h>
using namespace std;

class Tensor{
public:

	Tensor() : rows_(0), cols_(0) {}
	
	Tensor(int rows,int cols);

	int rows() const;
	int cols() const;

	float* data();
	const float* data() const;

	float& at(int i,int j);
	const float& at(int i, int j) const;

	bool load_from_file(const string& path);

private:
	int rows_;
	int cols_;
	vector<float> data_;

};