#pragma once
#include <unordered_map>
#include <string>
#include "tensor.h"
#include <vector>
#include <memory>


class Node{
	public:
		virtual ~Node() = default;
		virtual void forward() = 0;
};

class Graph{
private:
	std::vector<std::unique_ptr<Node>> nodes;

public:
	Graph(){};
	std::unordered_map<std::string, Tensor> tensors;
	void add(std::unique_ptr<Node> node){
		nodes.push_back(std::move(node));
	}

	void run(){
		for( auto & node : nodes){
			node->forward();
		}
	}

};