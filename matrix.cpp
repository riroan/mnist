#include"matrix.h"
#include<iostream>

std::random_device rd;

matrix::matrix(const int& _row, const int& _col)
	:row(_row), col(_col)
{
	init_matrix(_row, _col);
}

matrix::matrix()
{
	row = 0;
	col = 0;
	values.resize(0);
}

void matrix::print()
{
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			printf(" %.2lf", values[j + i * col]);
		printf("\n");
	}
}

void matrix::init_matrix(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	values.resize(row*col);
}

void matrix::resize(const int& _row, const int& _col)
{
	row = _row;
	col = _col;
	values.resize(_row*_col);
}

void matrix::assign_random(const double& min, const double& max)
{
	v_assign_random(values, min, max);
}

void matrix::assign_random_n(const double& s)
{
	v_assign_random_n(values, s);
}

Vector<double> matrix::getVector(const int& _size)
{
	Vector<double> ret(_size);
	for (int i = 0; i < _size; i++)
		ret[i] = 0.0;
	return ret;
}

void matrix::productTo(const Vector<double>& v, Vector<double>& to)
{
	for (int i = 0; i < row; i++)
	{
		to[i] = 0.0;
		for (int j = 0; j < col; j++)
			to[i] += getValue(i, j)*v[j];
	}
}

Vector<double> matrix::operator*(const Vector<double>& right)
{
	assert(col == right.size);
	Vector<double> ret = getVector(row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret[i] += getValue(i, j)*right.values[j];

	return ret;
}

double& matrix::getValue(const int& i, const int& j)
{
	return values[i*col + j];
}

matrix matrix::operator*(matrix right)
{
	assert(col == right.row);
	matrix ret(row, right.col);
	ret.assign_random(0.0, 0.0);

	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			ret.getValue(i, j) += getValue(i, j)*right.getValue(i, j);

	return ret;
}

double& matrix::operator[](const int& i)
{
	return values[i];
}

matrix matrix::productTranspose(matrix right)
{
	matrix temp(right.col, right.row);
	for (int i = 0; i < right.row; i++)
		for (int j = 0; j < right.col; j++)
			temp.getValue(j, i) = right.getValue(i, j);
	return (*this)*temp;
}

matrix matrix::Transpose()
{
	matrix temp(col, row);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			temp.getValue(j, i) = getValue(i, j);
	return temp;
}

double dot(const Vector<double>& left, const Vector<double>& right)
{
	assert(left.size == right.size);
	double ret = 0.0;

	for (int i = 0; i < left.size; i++)
		ret += left.values[i] * right.values[i];
	return ret;
}

Vector<double> vsum(const Vector<double>& left, const Vector<double>& right)
{
	assert(left.size == right.size);
	Vector<double> ret(left.size);

	for (int i = 0; i < left.size; i++)
		ret[i] = left.values[i] + right.values[i];
	return ret;
}

void v_assign_random(Vector<double>& v, const double& min, const double& max)
{
	std::uniform_real_distribution<double> distribution(min, max);

	for (int i = 0; i < v.size; i++)
		//v[i] = distribution(rd);
		v[i] = (max - min) * ((double)rand() / RAND_MAX) + min;
}

void v_assign_random_n(Vector<double>& v, const double& s)
{
	std::normal_distribution<double> distribution(0.0, 1.0);
	//std::uniform_real_distribution distribution(-1.0, 1.0);

	for (int i = 0; i < v.size; i++)
		v[i] = distribution(rd) / s;
}