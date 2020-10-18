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
		const vector<Data::Batch> batches = data.get_training_batches(batch_size);

		/* initialize mean cost */
		double C_mean = 0;

		/* perform stochastic gradient descent */
		for(const Data::Batch& batch : batches) {

			/* train on mini batch */
			for (const Data::TrainSet& set : batch) {
				VectorXd a = set.first;

				/* feed forward */
				for (int l = 0; l < (int)layers.size(); ++l)
					a = layers[l].feed_forward(a);

				/* add up cost */
				C_mean += cost->eval(a, set.second);

				/* calculate cost derivative */
				VectorXd dC_da_out = cost->deriv(a, set.second);

				/* back propagation */
				for (int l = (int)layers.size() - 1; l >= 0; --l)
					dC_da_out = layers[l].feed_backward(dC_da_out);
			}

			/* update layer weights */
			for (int l = 0; l < (int)layers.size(); ++l)
				layers[l].update_weights(alpha, batch.size());
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

vector<Data::TestSet> Network::_test() const
{
	const vector<Data::TestSet> test_data = data.get_test_data();

	int n_correct = 0;

	vector<Data::TestSet> incorrect;

	for(const Data::TestSet& set : test_data) {
		VectorXd a = set.first;

		/* feed forward */
		for (int l = 0; l < (int)layers.size(); ++l)
			a = layers[l].feed_forward(a);

		/* check if output is correct */
		int prediction; a.maxCoeff(&prediction);
		if (prediction == set.second) {
			++n_correct;
		} else {
			incorrect.emplace_back(set);
		}
	}

	/* display the amount of correct classifications */
	cout << n_correct << "/" << test_data.size() << " ("
		 << fixed << setprecision(2) << setw(5) << 100.0*n_correct/test_data.size()
		 << "%) correct";

	return incorrect;
}

void Network::test(int n_incorrect) const
{
	cout << "Testing neural network on " << data.get_n_test_sets() << " sets:" << endl;

	vector<Data::TestSet> incorrect = _test();

	/* shuffle the incorrect data */
	rng.shuffle<Data::TestSet>(incorrect);

	cout << "\n\nIncorrectly classified data:" << endl;

	for (int i = 0; i < n_incorrect; ++i) {
		VectorXd a = incorrect[i].first;

		/* feed forward */
		for (int l = 0; l < (int)layers.size(); ++l)
			a = layers[l].feed_forward(a);

		/* get prediciton */
		int prediction; a.maxCoeff(&prediction);

		/* show incorrect data */
		data.show_data(incorrect[i]);
		cout << "Predicition: " << prediction << endl << endl;
	}
}
