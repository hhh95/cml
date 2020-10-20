#include "data.hpp"

using namespace std;
using namespace Eigen;

RandomNumberGenerator rng;

MNIST::MNIST(const string& dir_name, int training_split, int validation_split) :
	Data(dir_name)
{
	/* read training data and labels */
	MatrixXd training_images = read_mnist_images(dir_name + "/train-images-idx3-ubyte");
	MatrixXd training_labels = read_mnist_labels(dir_name + "/train-labels-idx1-ubyte");

	/* read test data and labels */
	MatrixXd test_images = read_mnist_images(dir_name + "/t10k-images-idx3-ubyte");
	MatrixXd test_labels = read_mnist_labels(dir_name + "/t10k-labels-idx1-ubyte");

	/* determine inputs and outputs of the data set */
	n_inputs  = training_images.rows();
	n_outputs = training_labels.rows();
	cout << "- " << n_inputs << " inputs, " << n_outputs << " outputs" << endl;

	/* make sure the training and test data have the same layout */
	assert(test_images.rows() == n_inputs);
	assert(test_labels.rows() == n_outputs);

	/* create training data */
	training_data = make_pair(
		training_images.leftCols(training_split),
		training_labels.leftCols(training_split)
	);
	cout << "- " << get_n_training_sets() << " training data sets" << endl;

	/* create validation data */
	validation_data = make_pair(
		training_images.rightCols(validation_split),
		training_labels.rightCols(validation_split)
	);
	cout << "- " << get_n_validation_sets() << " validation data sets" << endl;

	/* create test data */
	test_data = make_pair(
		test_images,
		test_labels
	);
	cout << "- " << get_n_test_sets() << " test data sets" << endl << endl;
}

uint32_t MNIST::reverse_int(uint32_t& n)
{
    uint32_t b0,b1,b2,b3;

    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;

	n = (b0 | b1 | b2 | b3);

    return n;
}

MatrixXd MNIST::read_mnist_images(const string& file_name)
{
	ifstream fin(file_name, ios::binary);
	assert(fin.is_open());

	uint32_t magic_number, n_images, n_rows, n_cols;

	fin.read((char*)&magic_number, sizeof(magic_number));
	assert(reverse_int(magic_number) == 2051);

	fin.read((char*)&n_images, sizeof(n_images)); reverse_int(n_images);
	fin.read((char*)&n_rows, sizeof(n_rows)); reverse_int(n_rows);
	fin.read((char*)&n_cols, sizeof(n_cols)); reverse_int(n_cols);

	int n_pixels = n_rows*n_cols;

	Matrix<uint8_t, Dynamic, Dynamic> images(n_pixels, n_images);

	fin.read((char*)images.data(), n_pixels*n_images*sizeof(uint8_t));

	fin.close();

	return images.cast<double>()/255.0;
}

MatrixXd MNIST::read_mnist_labels(const string& file_name)
{
	ifstream fin(file_name, ios::binary);
	assert(fin.is_open());

	uint32_t magic_number, n_labels;

	fin.read((char*)&magic_number, sizeof(magic_number));
	assert(reverse_int(magic_number) == 2049);

	fin.read((char*)&n_labels, sizeof(n_labels)); reverse_int(n_labels);

	Matrix<uint8_t, Dynamic, 1, 0, Dynamic, 1> _labels(n_labels);

	fin.read((char*)_labels.data(), n_labels*sizeof(uint8_t));

	fin.close();

	int n_outputs = _labels.maxCoeff() - _labels.minCoeff() + 1;

	MatrixXd labels(n_outputs, _labels.size());
	labels.setZero();

	for (int i = 0; i < (int)_labels.size(); ++i)
		labels(_labels(i), i) = 1;

	return labels;
}

void MNIST::show_data(const VectorXd& data) const
{
	int n_pixels = data.size();
	int n_cols = sqrt(n_pixels);

	for (int pixel = 0; pixel < n_pixels; ++pixel) {
		int val = (int)(4*data(pixel) - 0.01);

		switch (val) {
			case 0: cout << "░"; break;
			case 1: cout << "▒"; break;
			case 2: cout << "▓"; break;
			case 3: cout << "█"; break;
		}

		if ((pixel + 1)%n_cols == 0)
			cout << endl;
	}
}
