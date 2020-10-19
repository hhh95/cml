#include "data.hpp"

using namespace std;
using namespace Eigen;

RandomNumberGenerator rng;

MNIST::MNIST(const string& dir_name, const vector<int>& splits) :
	Data(dir_name, splits)
{
	/* training data is split in two parts */
	assert(splits.size() == 2);

	/* read training data and labels */
	MatrixXd training_images = read_mnist_images(dir_name + "/train-images-idx3-ubyte");
	VectorXi training_labels = read_mnist_labels(dir_name + "/train-labels-idx1-ubyte");

	/* read test data and labels */
	MatrixXd test_images = read_mnist_images(dir_name + "/t10k-images-idx3-ubyte");
	VectorXi test_labels = read_mnist_labels(dir_name + "/t10k-labels-idx1-ubyte");

	/* determine inputs and outputs of the data set */
	n_inputs  = training_images.rows();
	n_outputs = training_labels.maxCoeff() - training_labels.minCoeff() + 1;
	cout << "- " << n_inputs << " inputs, " << n_outputs << " outputs" << endl;

	/* make sure the training and test data have the same layout */
	assert(test_images.rows() == n_inputs);
	assert(test_labels.maxCoeff() - test_labels.minCoeff() + 1 == n_outputs);

	/* create training data */
	MatrixXd outputs(n_outputs, splits[0]);
	outputs.setZero();
	for (int i = 0; i < splits[0]; ++i)
		outputs(training_labels(i), i) = 1;
	training_data = make_pair(
		training_images.leftCols(splits[0]),
		outputs
	);
	cout << "- " << get_n_training_sets() << " training data sets" << endl;

	/* create validation data */
	validation_data = make_pair(
		training_images.rightCols(splits[1]),
		training_labels.bottomRows(splits[1])
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

VectorXi MNIST::read_mnist_labels(const string& file_name)
{
	ifstream fin(file_name, ios::binary);
	assert(fin.is_open());

	uint32_t magic_number, n_labels;

	fin.read((char*)&magic_number, sizeof(magic_number));
	assert(reverse_int(magic_number) == 2049);

	fin.read((char*)&n_labels, sizeof(n_labels)); reverse_int(n_labels);

	Matrix<uint8_t, Dynamic, 1, 0, Dynamic, 1> labels(n_labels);

	fin.read((char*)labels.data(), n_labels*sizeof(uint8_t));

	fin.close();

	return labels.cast<int>();
}

void MNIST::show_data(const VectorXd& data, int label) const
{
	int n_pixels = data.size();
	int n_cols = sqrt(n_pixels);

	cout << "Image:" << endl;
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
	cout << "Label: " << label << endl;
}
