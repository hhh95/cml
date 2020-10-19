#ifndef DATA_HPP
#define DATA_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include "random.hpp"

class Data {
	public:
		using VectorXi = Eigen::VectorXi;
		using VectorXd = Eigen::VectorXd;
		using MatrixXd = Eigen::MatrixXd;

		using TrainSets = std::pair<MatrixXd, MatrixXd>;
		using TestSets  = std::pair<MatrixXd, VectorXi>;

		Data(const std::string& dir_name, const std::vector<int>& splits) :
			splits{splits}
		{
			std::cout << "Reading data from '" << dir_name << "':" << std::endl;
		}

		virtual void show_data(const VectorXd& data, int label) const = 0;

		void shuffle_training_data() {
			std::vector<int> idx = rng.random_indices(get_n_training_sets());
			TrainSets training_data_copy = training_data;
			for (int i = 0; i < get_n_training_sets(); ++i) {
				training_data.first.col(i) = training_data_copy.first.col(idx[i]);
				training_data.second.col(i) = training_data_copy.second.col(idx[i]);
			}
		}

		const std::vector<TrainSets> get_training_batches(int batch_size) const {
			std::vector<TrainSets> batches;

			int i = 0;

			while (i + batch_size < get_n_training_sets()) {
				TrainSets batch = std::make_pair(
					training_data.first.middleCols(i, batch_size),
					training_data.second.middleCols(i, batch_size)
				);

				batches.emplace_back(batch);

				i += batch_size;
			}

			if (i < get_n_training_sets()) {
				batch_size = get_n_training_sets() - i - 1;

				TrainSets batch = std::make_pair(
					training_data.first.middleCols(i, batch_size),
					training_data.second.middleCols(i, batch_size)
				);

				batches.emplace_back(batch);
			}

			return batches;
		}

		const TrainSets& get_training_sets() const {
			return training_data;
		}

		const TestSets& get_validation_sets() const {
			return validation_data;
		}

		const TestSets& get_test_sets() const {
			return test_data;
		}

		int get_n_inputs() const { return n_inputs; }

		int get_n_outputs() const { return n_outputs; }

		int get_n_training_sets() const { return training_data.first.cols(); }

		int get_n_validation_sets() const { return validation_data.first.cols(); }

		int get_n_test_sets() const { return test_data.first.cols(); }

	protected:
		const std::vector<int>& splits;

		TrainSets training_data;
		TestSets  validation_data;
		TestSets  test_data;

		int n_inputs;
		int n_outputs;
};

class MNIST : public Data {
	public:

		MNIST(const std::string& dir_name, const std::vector<int>& splits);

		void show_data(const VectorXd& data, int label) const override;

	private:

		uint32_t reverse_int(uint32_t& n);

		MatrixXd read_mnist_images(const std::string& file_name);

		VectorXi read_mnist_labels(const std::string& file_name);
};

#endif
