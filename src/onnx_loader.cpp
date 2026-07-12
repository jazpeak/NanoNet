#include "onnx_loader.h"
#include <fstream>
#include <unordered_map>
#include <onnx/onnx_pb.h>
#include "nodes.h"
#include <filesystem>
#include <cstring>
using namespace std;


Graph ONNXLoader::load(const string& path){
	onnx::ModelProto model;
	
	ifstream in(path,ios::binary);
    if (!model.ParseFromIstream(&in)) {
        throw runtime_error("Failed to load ONNX model");
    }

    const auto & graph_proto = model.graph();

    filesystem::path model_path(path);
    filesystem::path base_dir = model_path.parent_path();

    Graph graph;
    auto& tensors = graph.tensors;

    for( const auto& init : graph_proto.initializer()){
        int rows = 1;
        int cols = 1;
        if (init.dims_size() == 2) {
            rows = init.dims(0);
            cols = init.dims(1);
        } else if (init.dims_size() == 1) {
            // Treat 1D as row vector (1 x N) usually better for broadcasting in many engines,
            // but let's see. Or (N x 1).
            // For Gemm bias (C), it is usually (N).
            // Helper "Tensor" class might be simple.
            // Let's assume (1, N) for now.
            rows = 1;
            cols = init.dims(0);
            cout << "Warning: 1D Tensor " << init.name() << " treated as 1x" << cols << "\n";
        } else {
             // Handle 3D+ or 0D if needed?
             cout << "Warning: unexpected dim size " << init.dims_size() << " for " << init.name() << "\n";
             if (init.dims_size() > 0) rows = init.dims(0);
             if (init.dims_size() > 1) cols = init.dims(1);
        }

        Tensor t(rows, cols);
        size_t expected_bytes = rows * cols * sizeof(float);

        if (init.external_data_size() > 0) {
            string location;
            unsigned long long offset = 0;
            
            for (int i=0; i<init.external_data_size(); ++i) {
                const auto& entry = init.external_data(i);
                if (entry.key() == "location") location = entry.value();
                else if (entry.key() == "offset") offset = stoull(entry.value());
            }

            filesystem::path data_path = base_dir / location;
            ifstream ifs(data_path, ios::binary);
            if (!ifs) {
                 throw runtime_error("Could not open external data: " + data_path.string());
            }
            
            ifs.seekg(offset);
            
            // Handle INT64 specialized loading
            if (init.data_type() == 7) { // INT64
                size_t num_elements = rows * cols;
                vector<int64_t> buffer(num_elements);
                ifs.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(int64_t));
                for(size_t i=0; i<num_elements; ++i) t.data()[i] = (float)buffer[i];
            } else {
                 ifs.read(reinterpret_cast<char*>(t.data()), expected_bytes);
            }
            if (!ifs) {
                 throw runtime_error("Failed to read external data from " + data_path.string());
            }
        }
        else if (!init.raw_data().empty()) {
            const string& raw = init.raw_data();
            
            if (init.data_type() == 7) { // INT64
                 size_t num_elements = rows * cols;
                 if (raw.size() >= num_elements * sizeof(int64_t)) {
                     const int64_t* ptr = reinterpret_cast<const int64_t*>(raw.data());
                     for(size_t i=0; i<num_elements; ++i) t.data()[i] = (float)ptr[i];
                 }
            } else {
                if (raw.size() != expected_bytes) {
                    cerr << "Warning: raw_data size mismatch for " << init.name() 
                              << " (got " << raw.size() << ", expected " << expected_bytes << ")\n";
                }
                memcpy(t.data(), raw.data(), min(raw.size(), expected_bytes));
            }
        }
        else {
            if (init.float_data_size() > 0) {
                 for (int i = 0; i < rows * cols; i++) {
                     if (i < init.float_data_size())
                        t.data()[i] = init.float_data(i);
                 }
            } else {
                 // Might be int32_data etc. but we only support float in this engine
                 cerr << "Warning: no float data found for identifier " << init.name() << "\n";
            }
        }

        tensors[init.name()] = move(t);
    }

	for (const auto& input : graph_proto.input()) {
        const auto& shape = input.type().tensor_type().shape();

        long long rows = 1;
        long long cols = 1;

        if (shape.dim_size() > 0) {
            rows = shape.dim(0).dim_value();
            if (rows < 1) rows = 1; 
        }

        // Flatten all remaining dimensions into columns
        for (int k = 1; k < shape.dim_size(); ++k) {
            cols *= shape.dim(k).dim_value();
        }

        tensors[input.name()] = Tensor(rows, cols);
    }

    for(const auto & node: graph_proto.node()){
    	const string & op = node.op_type();

    	if (op=="Gemm"){
    		Tensor& A = tensors[node.input(0)];
    		Tensor& B = tensors[node.input(1)];
    		
    		Tensor& bias = tensors[node.input(2)];
    		bool transB = false;
		    float alpha = 1.0f;
		    float beta = 1.0f;

		    for (const auto& attr : node.attribute()) {
		        if (attr.name() == "transB") {
		            transB = attr.i() == 1;
		        } else if (attr.name() == "alpha") {
		            alpha = attr.f();
		        } else if (attr.name() == "beta") {
		            beta = attr.f();
		        }
		    }

		  
		    if (alpha != 1.0f || beta != 1.0f) {
		        throw runtime_error("Only alpha=1, beta=1 supported for Gemm");
		    }

		    if (!transB) {
		        throw runtime_error("Only transB=1 supported for Gemm");
		    }

		    int m = A.rows();
		    int n = B.rows();  
		    string temp = node.output(0) + "_mm";

		    tensors[temp] = Tensor(m, n);
		    graph.add(make_unique<MatMulNode>(
		        A,
		        B,                     // MatMulNode should treat B as transposed
		        tensors[temp],
		        /*transB=*/true
		    ));

		    // ---- Final output tensor ----
		    tensors[node.output(0)] = Tensor(m, n);

		    // ---- Add bias ----
		    graph.add(make_unique<AddNode>(
		        tensors[temp],
		        bias,
		        tensors[node.output(0)]
		    ));
		}

		else if (op=="Relu")
		{
			Tensor& A = tensors[node.input(0)];
			tensors[node.output(0)] = Tensor(A.rows(), A.cols());
			graph.add(make_unique<ReluNode>(A,tensors[node.output(0)]));
		}
		else if (op=="Softmax")
		{
			Tensor& A = tensors[node.input(0)];
			tensors[node.output(0)] = Tensor(A.rows(), A.cols());
			graph.add(make_unique<SoftmaxNode>(A,tensors[node.output(0)]));
		}
		else if (op=="Reshape"){
			Tensor& A = tensors[node.input(0)];
            Tensor& shape_tensor = tensors[node.input(1)];
            
            vector<int> new_dims;
            int total_size = A.rows() * A.cols();
            int inferred_idx = -1;
            int product = 1;
            
            int shape_size = shape_tensor.rows() * shape_tensor.cols();
            for(int i=0; i<shape_size; ++i) {
                int d = (int)shape_tensor.data()[i];
                if (d == -1) {
                    inferred_idx = i;
                    new_dims.push_back(-1);
                } else {
                    new_dims.push_back(d);
                    product *= d;
                }
            }
            
            if (inferred_idx != -1) {
                 if (product != 0)
                    new_dims[inferred_idx] = total_size / product;
                 else
                    new_dims[inferred_idx] = 0; // Should not happen for valid reshape
            }
            
            int new_rows = 1;
            int new_cols = 1;
            if (new_dims.size() > 0) new_rows = new_dims[0];
            for(size_t i=1; i<new_dims.size(); ++i) new_cols *= new_dims[i];
            
            tensors[node.output(0)] = Tensor(new_rows, new_cols);
            
            // Reusing FlattenNode since it just copies data
            graph.add(make_unique<FlattenNode>(A, tensors[node.output(0)]));
		}
		else if (op=="Flatten"){
			Tensor& A = tensors[node.input(0)];
			int batch = A.rows();
		    int features = A.cols();

		    tensors[node.output(0)] = Tensor(batch, features);
		        graph.add(make_unique<FlattenNode>(A, tensors[node.output(0)]));
		}else{
			cout << "No op as such: " << op << "\n";
		}
    }

    return graph;
}