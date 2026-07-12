#pragma once
#include <unordered_map>
#include <string>
#include "tensor.h"
#include <vector>
#include <memory>
using namespace std;


class Node{
	public:
		virtual ~Node() = default;
		virtual void forward() = 0;
};

class Graph{
private:
	vector<unique_ptr<Node>> nodes;

public:
	Graph(){};
	unordered_map<string, Tensor> tensors;
	void add(unique_ptr<Node> node){
		nodes.push_back(move(node));
	}

	void run(){
		for( auto & node : nodes){
			node->forward();
		}
	}

};