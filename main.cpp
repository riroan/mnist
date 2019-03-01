#include<iostream>
#include"Network.h"
#pragma warning(disable:4996)
using namespace std;

union BYTE4 {
	int v;
	unsigned char uc[4];
};

int read_int_bigendian(FILE *fp) {
	unsigned char buf4[4] = { 0 };
	BYTE4 b4;
	fread(buf4, 1, 4, fp);
	b4.uc[0] = buf4[3];
	b4.uc[1] = buf4[2];
	b4.uc[2] = buf4[1];
	b4.uc[3] = buf4[0];
	return b4.v;
}

int getAnswer(const Vector<double>& x)
{
	double max = x[0];
	int ix = 0;
	for (int i = 1; i < x.size; i++)
		if (x[i] > max)
		{
			max = x[i];
			ix = i;
		}
	return ix;
}

int main()
{
	FILE *f_image = NULL, *f_label = NULL;
	int i;
	long filelen;
	int magicnum, imagenum, imagewidth, imageheight;
	unsigned char image[28 * 28];
	unsigned char val_label;

	f_label = fopen("mnist/train-labels.idx1-ubyte", "rb");
	f_image = fopen("mnist/train-images.idx3-ubyte", "rb");

	fseek(f_image, 0L, SEEK_END);
	fseek(f_image, 0L, SEEK_SET);

	read_int_bigendian(f_label);
	read_int_bigendian(f_label);

	magicnum = read_int_bigendian(f_image);
	imagenum = read_int_bigendian(f_image);
	imagewidth = read_int_bigendian(f_image);
	imageheight = read_int_bigendian(f_image);

	Network myNetwork(28 * 28, 10, 0);
	myNetwork.output_softmax = true;

	i = 0;
	int num_train = 10000;
	while (i < num_train) {
		fread(image, 28 * 28, 1, f_image);
		fread(&val_label, 1, 1, f_label);

		Vector<double> v(28 * 28);
		Vector<double> y(10, 0.0);

		y[val_label] = 1.0;
		for (int j = 0; j < 28 * 28; j++)
			v[j] = (double)image[j] / 255.0;
		myNetwork.setInput(v);
		myNetwork.feedForward();

		myNetwork.backPropagation(y);

		i++;
		//if (i % 100 == 0)
		//	cout << "check Point" << endl;
	}
	cout << "Training End" << endl;
	cout << "\nPress any key to start test" << endl;
	getchar();

	fclose(f_image);
	fclose(f_label);

	f_label = fopen("mnist/t10k-labels.idx1-ubyte", "rb");
	f_image = fopen("mnist/t10k-images.idx3-ubyte", "rb");

	int correct = 0;

	fseek(f_image, 0L, SEEK_END);
	fseek(f_image, 0L, SEEK_SET);

	read_int_bigendian(f_label);
	read_int_bigendian(f_label);
	magicnum = read_int_bigendian(f_image);
	imagenum = read_int_bigendian(f_image);
	imagewidth = read_int_bigendian(f_image);
	imageheight = read_int_bigendian(f_image);

	int num_data = 1000;

	i = 0;
	while (i < num_data) {
		fread(image, 28 * 28, 1, f_image);
		fread(&val_label, 1, 1, f_label);

		Vector<double> v(28 * 28);
		Vector<double> temp(10);

		for (int j = 0; j < 28 * 28; j++)
			v[j] = (double)image[j] / 255.0;

		myNetwork.setInput(v);
		myNetwork.feedForward();
		
		temp = myNetwork.getOutput();

		int answer = 0;

		answer = getAnswer(temp);

		if (answer == val_label)
			correct++;

		i++;
	}

	printf("Accuracy : %lf%%", (double)correct / (double)num_data * 100.0);

	fclose(f_image);
	fclose(f_label);
}