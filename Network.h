#pragma once
#include "matrix.h"

class Network
{
public:
	int num_input;
	int num_output;
	int num_hidden;
	int num_layer;

	bool   output_softmax;
	double alpha;
	double bias;

	Vector<matrix>		   w;				// weights
	Vector<int>			   num_neurons;
	Vector<Vector<double>> layer_grad;		// gradients
	Vector<Vector<double>> neurons;
	Vector<unsigned int>   activation_type;	// 0 : sigmoid, 1 : ReLU, 2 : LReLU(default)
	Vector<matrix>		   h;				// for AdaGrad
	Vector<matrix>		   v;				// for momentum

	Network(const int& _num_input, const int& _num_output, const int& _num_hidden);

	void init_network(const int& _num_input, const int& _num_output, const int& _num_hidden);
	void init_network(const Vector<int>& _num_neurons);

	void apply_identity(Vector<double>& v);
	void apply_sigmoid(Vector<double>& v);
	void apply_ReLU(Vector<double>& v);
	void apply_LReLU(Vector<double>& v);
	void apply_softmax(Vector<double>& v);

	inline double identity(const double& x) { return x; }
	inline double sigmoid(const double& x) { return 1.0 / (1.0 + exp(-x)); }
	inline double ReLU(const double& x) { return x > 0.0 ? x : 0.0; }
	inline double LReLU(const double& x) { return x > 0.0 ? x : 1e-2*x; }

	inline double grad_identity(const double& x) { return 1.0; }
	inline double grad_sigmoid(const double& y) { return (1.0 - y)*y; }
	inline double grad_ReLU(const double& y) { return y > 0 ? 1.0 : 0.0; }
	inline double grad_LReLU(const double& y) { return y > 0 ? 1.0 : 1e-2; }

	void setInput(const Vector<double>& v);
	void feedForward();
	void backPropagation(const Vector<double>& v);

	void printOutput();
	Vector<double> gradient_product(const matrix &w, const Vector<double>& layer);
	Vector<double> getOutput();

	void getGradient_MSE(const Vector<double>& v);
	void update_weight_SGD();
	void update_weight_momentum();		// not yet
	void update_weight_AdaGrad();		// not yet
};