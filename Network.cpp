#include "Network.h"
#include<iostream>
#include<windows.h>

namespace act
{
	const int sigmoid = 0;
	const int relu = 1;
	const int lrelu = 2;
}

Network::Network(const int& _num_input, const int& _num_output, const int& _num_hidden)
{
	init_network(_num_input, _num_output, _num_hidden);
}

void Network::init_network(const Vector<int>& _num_neurons)
{
	output_softmax = false;
	activation_type.resize(num_layer, act::relu);

	neurons.resize(num_layer);
	layer_grad.resize(num_layer);
	for (int i = 0; i < num_layer; i++)
	{
		neurons[i].resize(_num_neurons[i], 0.0);
		neurons[i][_num_neurons[i] - 1] = bias;
		layer_grad[i].resize(_num_neurons[i], 0.0);
	}

	v.resize(num_layer - 1);
	w.resize(num_layer - 1);
	h.resize(num_layer - 1);
	for (int i = 0; i < num_layer - 1; i++)
	{
		w[i].init_matrix(_num_neurons[i + 1] - 1, _num_neurons[i]);
		v[i].init_matrix(_num_neurons[i + 1] - 1, _num_neurons[i]);
		h[i].init_matrix(_num_neurons[i + 1] - 1, _num_neurons[i]);

		// initialize to 0
		h[i].assign_random(0.0, 0.0);
		v[i].assign_random(0.0, 0.0);

		w[i].assign_random(0.0, 1.0);							// random initialization
		//w[i].assign_random_n(sqrt(2.0 / _num_neurons[i]));	// Xe initialization
		//w[i].assign_random_n(sqrt(1.0 / _num_neurons[i]));	// Xavier initialization
	}
}

void Network::init_network(const int& _num_input, const int& _num_output, const int& _num_hidden)
{
	num_input = _num_input;
	num_output = _num_output;
	num_hidden = _num_hidden;
	num_layer = num_hidden + 2;

	num_neurons.resize(num_layer);

	for (int i = 0; i < num_layer - 1; i++)	// except output layer
		num_neurons[i] = num_input + 1;
	num_neurons[num_layer - 1] = num_output + 1;

	bias = 1;
	alpha = 0.15;

	init_network(num_neurons);
}

void Network::apply_identity(Vector<double>& v)
{
	int v_size = v.size;
	for (int i = 0; i < v_size - 1; i++)
		v[i] = identity(v[i]);
}

void Network::apply_sigmoid(Vector<double>& v)
{
	int v_size = v.size;
	for (int i = 0; i < v_size - 1; i++)
		v[i] = sigmoid(v[i]);
}

void Network::apply_ReLU(Vector<double>& v)
{
	for (int i = 0; i < v.size - 1; i++)
		v[i] = ReLU(v[i]);
}

void Network::apply_LReLU(Vector<double>& v)
{
	int v_size = v.size;
	for (int i = 0; i < v_size - 1; i++)
		v[i] = LReLU(v[i]);
}

void Network::apply_softmax(Vector<double>& v)
{
	double max = v[0];

	int v_size = v.size;
	for (int i = 1; i < v_size - 1; i++)
		if (v[i] > max)
			max = v[i];

	double sum = 0.0;
	for (int i = 0; i < v_size - 1; i++)
	{
		v[i] = exp(v[i] - max);
		sum += v[i];
	}
	for (int i = 0; i < v_size - 1; i++)
		v[i] /= sum;
}

void Network::feedForward()
{
	for (int i = 0; i < w.size; i++)
	{
		neurons[i + 1] = w[i] * neurons[i];

		if (output_softmax&& i == w.size - 1)
		{
			apply_softmax(neurons[i + 1]);
			return;
		}

		if (activation_type[i] == act::sigmoid)
			apply_sigmoid(neurons[i + 1]);
		else if (activation_type[i] == act::relu)
			apply_ReLU(neurons[i + 1]);
		else
			apply_LReLU(neurons[i + 1]);
	}
}

Vector<double> Network::getOutput()
{
	Vector<double> ret(num_output);
	for (int i = 0; i < num_output; i++)
		ret[i] = neurons[num_layer - 1][i];
	return ret;
}

void Network::setInput(const Vector<double>& v)
{
	for (int i = 0; i < num_input; i++)
		neurons[0][i] = v[i];
}

void Network::backPropagation(const Vector<double>& v)
{
	getGradient_MSE(v);
	update_weight_SGD();
}

void Network::update_weight_SGD()
{
	int w_size = w.size;
	for (int r = w_size - 1; r >= 0; r--)
		for (int i = 0; i < w[r].row; i++)
			for (int j = 0; j < w[r].col; j++)
			{
				const double delta = alpha * layer_grad[r + 1][i] * neurons[r][j];
				w[r].getValue(i, j) -= delta;
			}
}

void Network::update_weight_AdaGrad()
{
	int w_size = w.size;

	for (int r = w_size - 1; r >= 0; r--)
		for (int i = 0; i < w[r].row; i++)
			for (int j = 0; j < w[r].col; j++)
				h[r].getValue(i, j) += (layer_grad[r + 1][i] * neurons[r][j])*(layer_grad[r + 1][i] * neurons[r][j]);

	for (int r = w_size - 1; r >= 0; r--)
		for (int i = 0; i < w[r].row; i++)
			for (int j = 0; j < w[r].col; j++)
			{
				double delta = (layer_grad[r + 1][i] * neurons[r][j]) / (sqrt(h[r].getValue(i, j)) + 1e-7);
				w[r].getValue(i, j) -= delta;
			}
}

void Network::update_weight_momentum()
{
	double eta = 0.5;
	for (int r = w.size - 1; r >= 0; r--)
		for (int i = 0; i < w[r].row; i++)
			for (int j = 0; j < w[r].col; j++)
			{
				double delta = alpha * layer_grad[r + 1][i] * neurons[r][j] + eta * v[r].getValue(i, j);
				w[r].getValue(i, j) -= delta;
				v[r].getValue(i, j) = delta;
			}
}

void Network::getGradient_MSE(const Vector<double>& v)
{
	int last = num_layer - 1;
	for (int i = 0; i < layer_grad[last].size - 1; i++)
	{
		double last_value = neurons[last][i];
		// MSE
		if (activation_type[last] == act::sigmoid)
			layer_grad[last][i] = (last_value - v[i])*grad_sigmoid(last_value);
		else if (activation_type[last] == act::relu)
			layer_grad[last][i] = (last_value - v[i])*grad_ReLU(last_value);
		else
			layer_grad[last][i] = (last_value - v[i])*grad_LReLU(last_value);
	}

	for (int i = w.size - 1; i >= 0; i--)
	{
		layer_grad[i] = gradient_product(w[i], layer_grad[i + 1]);

		for (int j = 0; j < neurons[i].size - 1; j++)
			if (activation_type[i] == act::sigmoid)
				layer_grad[i][j] *= grad_sigmoid(neurons[i][j]);
			else if (activation_type[i] == act::relu)
				layer_grad[i][j] *= grad_ReLU(neurons[i][j]);
			else
				layer_grad[i][j] *= grad_LReLU(neurons[i][j]);
	}
}

Vector<double> Network::gradient_product(const matrix& w, const Vector<double>& layer)
{
	int row_ = w.row;
	int col_ = w.col;
	Vector<double> ret(col_);

	for (int i = 0; i < col_; i++)
	{
		ret[i] = 0.0;
		for (int j = 0, k = i; j < row_; j++, k += col_)
			ret[i] += w.values[k] * layer[j];
	}
	return ret;
}

void Network::printOutput()
{
	for (int i = 0; i < num_output; i++)
		std::cout << neurons[num_layer - 1][i] << " ";
	std::cout << std::endl;
}