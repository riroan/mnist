#pragma once
#include<iostream>

template<class T>
class Vector
{
public:
	int size;
	T * values;

	Vector()
	{
		size = 0;
		values = nullptr;
	}

	Vector(const int& _size)
	{
		size = _size;
		values = new T[size];
	}

	Vector(const int& _size, const T& assign)
	{
		size = _size;
		values = new T[size];
		for (int i = 0; i < size; i++)
			values[i] = assign;
	}

	Vector(const Vector<T>& from)
	{
		size = from.size;
		values = new T[size];
		for (int i = 0; i < size; i++)
			values[i] = from.values[i];
	}

	void resize(const int& _size)
	{
		delete[] values;
		size = _size;
		values = new T[size];
	}

	void resize(const int& _size, const T& assign)
	{
		delete[] values;
		size = _size;
		values = new T[size];

		for (int i = 0; i < size; i++)
			values[i] = assign;
	}

	void print()
	{
		for (int i = 0; i < size; i++)
			printf(" %lf", values[i]);
		printf("\n");
	}


	T& operator[](const int& i) const
	{
		return values[i];
	}

	void operator=(const Vector& v)
	{
		for (int i = 0; i < v.size; i++)
			values[i] = v[i];
	}
};