#include "network.hpp"

using namespace std;
using namespace Eigen;

Network::Network(Data& data, vector<Layer>& layers) :
	data{data}, layers{layers}
{
	/* start timer */
	wtime_start = chrono::high_resolution_clock::now();

	/* check that the inputs and outputs match the data */
	assert(layers[0].n_inputs == data.get_n_inputs());
	assert(layers[layers.size() - 1].n_outputs == data.get_n_outputs());

	/* check if network is fully connected */
	for (size_t i = 0; i < layers.size() - 1; ++i)
		assert(layers[i].n_outputs == layers[i + 1].n_inputs);

	/* print summary of network */
	cout << "Created fully connected artificial neural network:" << endl;
	for(const Layer& layer : layers) {
		cout << "- " << layer.n_inputs << " inputs, "
			 << layer.n_outputs << " outputs, "
			 << layer.sigma->get_name() << " activation"
			 << endl;
	}
	cout << endl;
}

void Network::train(double alpha, int epochs, int batch_size, shared_ptr<Cost> cost,
		double lambda, bool do_validation_inbetween, bool do_tests_inbetween)
{
	ofstream fout("history.csv");
	assert(fout.is_open());

	int n_training_sets = data.get_n_training_sets();

	cout << "Training neural network on " << n_training_sets << " sets with "
		 << cost->get_name() << " cost:" << endl;

	cout << "Epoch     Training      Validation        Test" << endl;

	fout << "epoch,accuracy training,cost training,"
		 << "accuracy validation,cost validation,"
		 << "accuray test,cost test"
		 << endl;

	for (int epoch = 0; epoch < epochs; ++epoch) {

		/* randomize training data */
		data.shuffle_training_data();

		/* get the training batches */
		const vector<Data::Sets> batches = data.get_training_batches(batch_size);

		int n_correct = 0;
		double C_mean = 0;

		/* perform stochastic gradient descent */
		for(const Data::Sets& batch : batches) {

			MatrixXd a = batch.first;

			/* feed forward */
			for (int l = 0; l < (int)layers.size(); ++l)
				a = layers[l].feed_forward(a);

			/* check if outputs are correct */
			for (int i = 0; i < (int)batch.first.cols(); ++i) {
				int prediction; a.col(i).maxCoeff(&prediction);
				int label; batch.second.col(i).maxCoeff(&label);
				if (prediction == label)
					++n_correct;
			}

			/* add up cost */
			C_mean += cost->eval(a, batch.second);

			/* calculate cost derivative */
			MatrixXd dC_da_out = cost->deriv(a, batch.second);

			/* back propagation */
			for (int l = (int)layers.size() - 1; l >= 0; --l)
				dC_da_out = layers[l].feed_backward(dC_da_out, alpha, lambda,
						data.get_n_training_sets());
		}

		cout << setw((int)log10(epochs) + 1) << epoch + 1
			 << "/" << epochs << fixed << setprecision(2);

		cout << "   " << 100.0*n_correct/data.get_n_training_sets()
			 << "%  " << C_mean/data.get_n_training_sets();

		fout << epoch << ","
			 << n_correct/(double)data.get_n_training_sets() << ","
			 << C_mean/data.get_n_training_sets();

		if (do_validation_inbetween)
			_validate(cost, fout);

		if (do_tests_inbetween)
			_test(cost, fout);

		cout << endl;
		fout << endl;
	}

	fout.close();

	cout << endl;
}

void Network::_validate(std::shared_ptr<Cost> cost, std::ofstream& fout) const
{
	const Data::Sets validation_data = data.get_validation_sets();

	int n_correct = 0;
	double C_mean = 0;

	MatrixXd a = validation_data.first;

	/* feed forward */
	for (int l = 0; l < (int)layers.size(); ++l)
		a = layers[l].feed_forward(a);

	/* check if output is correct */
	for (int i = 0; i < data.get_n_test_sets(); ++i) {
		int prediction; a.col(i).maxCoeff(&prediction);
		int label; validation_data.second.col(i).maxCoeff(&label);
		if (prediction == label)
			++n_correct;
	}

	/* add up cost */
	C_mean += cost->eval(a, validation_data.second);

	/* display the amount of correct classifications */
	cout << "   " << 100.0*n_correct/data.get_n_validation_sets()
		 << "%  " << C_mean/data.get_n_validation_sets();

	fout << "," << n_correct/(double)data.get_n_validation_sets()
		 << "," << C_mean/data.get_n_validation_sets();
}

void Network::_test(std::shared_ptr<Cost> cost, std::ofstream& fout) const
{
	const Data::Sets test_data = data.get_test_sets();

	int n_correct = 0;
	double C_mean = 0;

	MatrixXd a = test_data.first;

	/* feed forward */
	for (int l = 0; l < (int)layers.size(); ++l)
		a = layers[l].feed_forward(a);

	/* check if output is correct */
	for (int i = 0; i < data.get_n_test_sets(); ++i) {
		int prediction; a.col(i).maxCoeff(&prediction);
		int label; test_data.second.col(i).maxCoeff(&label);
		if (prediction == label)
			++n_correct;
	}

	/* add up cost */
	C_mean += cost->eval(a, test_data.second);

	/* display the amount of correct classifications */
	cout << "   " << 100.0*n_correct/data.get_n_test_sets()
		 << "%  " << C_mean/data.get_n_test_sets();

	fout << "," << n_correct/(double)data.get_n_test_sets()
		 << "," << C_mean/data.get_n_test_sets();
}

void Network::test(int n_incorrect, const std::map<int, std::string>& map) const
{
	cout << "Testing neural network on " << data.get_n_test_sets()
		 << " sets:" << endl;

	const Data::Sets test_data = data.get_test_sets();

	int n_correct = 0;

	vector<MatrixXd> incorrect_data;
	vector<int> incorrect_prediction;
	vector<int> incorrect_label;

	MatrixXd a = test_data.first;

	/* feed forward */
	for (int l = 0; l < (int)layers.size(); ++l)
		a = layers[l].feed_forward(a);

	/* check if output is correct */
	for (int i = 0; i < data.get_n_test_sets(); ++i) {
		int prediction; a.col(i).maxCoeff(&prediction);
		int label; test_data.second.col(i).maxCoeff(&label);
		if (prediction == label) {
			++n_correct;
		} else {
			incorrect_data.emplace_back(test_data.first.col(i));
			incorrect_prediction.emplace_back(prediction);
			incorrect_label.emplace_back(label);
		}
	}

	/* display the amount of correct classifications */
	cout << "Accuracy: " << setprecision(2)
		 << 100.0*n_correct/data.get_n_test_sets() << "%" << endl;

	cout << "\nIncorrectly classified data:" << endl;

	/* create random indices, so that differnet images are shown every run */
	vector<int> idx = rng.random_indices(incorrect_data.size());

	for (int i = 0; i < n_incorrect; ++i) {
		cout << "Image No. " << idx[i] << endl;

		data.show_data(incorrect_data[idx[i]]);

		if (map.size() > 0) {
			cout << "Label: " << map.at(incorrect_label[idx[i]]) << endl;
			cout << "Predicition: " << map.at(incorrect_prediction[idx[i]])
				 << endl << endl;
		} else {
			cout << "Label: " << incorrect_label[idx[i]] << endl;
			cout << "Predicition: " << incorrect_prediction[idx[i]]
				 << endl << endl;
		}
	}
}
