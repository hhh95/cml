#include "data.hpp"

using namespace std;
using namespace Eigen;

RandomNumberGenerator rng;

MNIST::MNIST(const string& dir_name, const vector<int>& splits) :
	Data(dir_name, splits)
{
	assert(splits.size() == 3);

	vector<VectorXd> training_images = read_mnist_images(dir_name
			+ "/train-images-idx3-ubyte");
	vector<uint8_t> training_labels = read_mnist_labels(dir_name
			+ "/train-labels-idx1-ubyte");

	vector<VectorXd> test_images = read_mnist_images(dir_name
			+ "/t10k-images-idx3-ubyte");
	vector<uint8_t> test_labels = read_mnist_labels(dir_name
			+ "/t10k-labels-idx1-ubyte");

	/* determine inputs and outputs of the data set */
	n_inputs = training_images[0].size();
	auto minmax = std::minmax_element(begin(training_labels), end(training_labels));
	n_outputs = *minmax.second - *minmax.first + 1;
	cout << "- " << n_inputs << " inputs, " << n_outputs << " outputs" << endl;

	int i_td = 0;

	/* create training data */
	for (int i = 0; i < splits[0]; ++i) {
		VectorXd output(10);

		output.setZero();
		output(training_labels[i_td]) = 1;

		training_data.emplace_back(make_pair(training_images[i_td], output));
		++i_td;
	}
	cout << "- " << training_data.size() << " training data sets" << endl;

	/* create validation data */
	for (int i = 0; i < splits[1]; ++i) {
		validation_data.emplace_back(make_pair(training_images[i_td], training_labels[i_td]));
		++i_td;
	}
	cout << "- " << validation_data.size() << " validation data sets" << endl;

	/* create test data */
	for (int i = 0; i < splits[2]; ++i) {
		test_data.emplace_back(make_pair(training_images[i], training_labels[i]));
	}
	cout << "- " << test_data.size() << " test data sets" << endl << endl;
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

vector<VectorXd> MNIST::read_mnist_images(const string& file_name)
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

	vector<VectorXd> images(n_images);

	for (uint32_t i = 0; i < n_images; ++i) {
		Matrix<uint8_t, Dynamic, 1, 0, Dynamic, 1> image(n_pixels);
		fin.read((char*)image.data(), n_pixels*sizeof(uint8_t));
		images[i] = image.cast<double>()/255.0;
	}

	fin.close();

	return images;
}

vector<uint8_t> MNIST::read_mnist_labels(const string& file_name)
{
	ifstream fin(file_name, ios::binary);
	assert(fin.is_open());

	uint32_t magic_number, n_labels;

	fin.read((char*)&magic_number, sizeof(magic_number));
	assert(reverse_int(magic_number) == 2049);

	fin.read((char*)&n_labels, sizeof(n_labels)); reverse_int(n_labels);

	vector<uint8_t> labels(n_labels);

	fin.read((char*)labels.data(), n_labels*sizeof(uint8_t));

	fin.close();

	return labels;
}

void MNIST::show_data(const TestSet& set) const
{
	int n = sqrt(set.first.size());

	cout << "Image:" << endl;

	for (int pixel = 0; pixel < set.first.size(); ++pixel) {
		int val = (int)(4*set.first(pixel) - 0.01);

		switch (val) {
			case 0: cout << "░"; break;
			case 1: cout << "▒"; break;
			case 2: cout << "▓"; break;
			case 3: cout << "█"; break;
		}

		if ((pixel + 1)%n == 0)
			cout << endl;
	}

	cout << "Label: " << set.second << endl;
}
