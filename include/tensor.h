#pragma once


#include <vector>
#include <string>

#include <stddef.h>

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

	bool load_from_file(const std::string& path);

private:
	int rows_;
	int cols_;
	std::vector<float> data_;

};