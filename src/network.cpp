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
			 << layer.n_outputs << " outputs" << endl;
	}
	cout << endl;
}

void Network::train(double alpha, int epochs, int batch_size, unique_ptr<Cost> cost,
		bool do_tests_inbetween)
{
	int n_training_sets = data.get_n_training_sets();

	cout << "Training neural network on " << n_training_sets << " sets:" << endl;

	for (int epoch = 0; epoch < epochs; ++epoch) {

		/* randomize training data */
		data.shuffle_training_data();

		/* get the training batches */
		const vector<Data::TrainSets> batches = data.get_training_batches(batch_size);

		/* initialize mean cost */
		double C_mean = 0;

		/* perform stochastic gradient descent */
		for(const Data::TrainSets& batch : batches) {

			MatrixXd a = batch.first;

			/* feed forward */
			for (int l = 0; l < (int)layers.size(); ++l)
				a = layers[l].feed_forward(a);

			/* add up cost */
			C_mean += cost->eval(a, batch.second);

			/* calculate cost derivative */
			MatrixXd dC_da_out = cost->deriv(a, batch.second);

			/* back propagation */
			for (int l = (int)layers.size() - 1; l >= 0; --l)
				dC_da_out = layers[l].feed_backward(dC_da_out, alpha);
		}

		if (do_tests_inbetween) {
			cout << "epoch " << setw((int)log10(epochs) + 1) << epoch + 1
				 << "/" << epochs << ": ";

			_test();

			cout << ", mean cost = " << C_mean/data.get_n_training_sets() << endl;

		} else {
			cout << "epoch " << setw((int)log10(epochs) + 1) << epoch + 1
				 << "/" << epochs << fixed << setprecision(2) << setw(5)
				 << ": mean cost = " << C_mean/data.get_n_training_sets() << endl;
		}
	}

	cout << endl;
}

Data::TestSets Network::_test() const
{
	const Data::TestSets test_data = data.get_test_sets();

	int n_correct = 0;

	vector<int> incorrect;

	MatrixXd a = test_data.first;

	/* feed forward */
	for (int l = 0; l < (int)layers.size(); ++l)
		a = layers[l].feed_forward(a);

	/* check if output is correct */
	for (int i = 0; i < data.get_n_test_sets(); ++i) {

		int prediction; a.col(i).maxCoeff(&prediction);

		if (prediction == test_data.second(i)) {
			++n_correct;
		} else {
			incorrect.emplace_back(i);
		}
	}

	/* display the amount of correct classifications */
	cout << setw((int)log10(data.get_n_test_sets()) + 1) << n_correct
		 << "/" << data.get_n_test_sets() << " ("
		 << fixed << setprecision(2) << setw(5) << 100.0*n_correct/data.get_n_test_sets()
		 << "%) correct";

	/* return the incorrect test sets */
	MatrixXd incorrect_data(data.get_n_inputs(), (int)incorrect.size());
	VectorXi incorrect_label((int)incorrect.size());

	for (int i = 0; i < (int)incorrect.size(); ++i) {
		incorrect_data.col(i) = test_data.first.col(incorrect[i]);
		incorrect_label(i) = test_data.second(incorrect[i]);
	}

	return make_pair(incorrect_data, incorrect_label);
}

void Network::test(int n_incorrect) const
{
	cout << "Testing neural network on " << data.get_n_test_sets() << " sets:" << endl;

	Data::TestSets incorrect = _test();

	cout << "\n\nIncorrectly classified data:" << endl;

	/* create random indices, so that differnet images are shown every run */
	vector<int> idx = rng.random_indices(incorrect.first.cols());

	for (int i = 0; i < n_incorrect; ++i) {
		VectorXd a = incorrect.first.col(idx[i]);

		/* feed forward */
		for (int l = 0; l < (int)layers.size(); ++l)
			a = layers[l].feed_forward(a);

		/* get prediciton */
		int prediction; a.maxCoeff(&prediction);

		/* show incorrect data */
		data.show_data(a, incorrect.second(idx[i]));
		cout << "Predicition: " << prediction << endl << endl;
	}
}
