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
		using VectorXd = Eigen::VectorXd;
		using MatrixXd = Eigen::MatrixXd;

		using Sets = std::pair<MatrixXd, MatrixXd>;

		Data(const std::string& dir_name) {
			std::cout << "Reading data from '" << dir_name << "':" << std::endl;
		}

		virtual void show_data(const VectorXd& data) const = 0;

		void shuffle_training_data() {
			std::vector<int> idx = rng.random_indices(get_n_training_sets());
			Sets training_data_copy = training_data;
			for (int i = 0; i < get_n_training_sets(); ++i) {
				training_data.first.col(i) = training_data_copy.first.col(idx[i]);
				training_data.second.col(i) = training_data_copy.second.col(idx[i]);
			}
		}

		const std::vector<Sets> get_training_batches(int batch_size) const {
			std::vector<Sets> batches;

			int i = 0;

			while (i + batch_size <= get_n_training_sets()) {
				Sets batch = std::make_pair(
					training_data.first.middleCols(i, batch_size),
					training_data.second.middleCols(i, batch_size)
				);

				batches.emplace_back(batch);

				i += batch_size;
			}

			/* left over sets will not be included in the batches */

			return batches;
		}

		const Sets& get_training_sets() const {
			return training_data;
		}

		const Sets& get_validation_sets() const {
			return validation_data;
		}

		const Sets& get_test_sets() const {
			return test_data;
		}

		int get_n_inputs() const { return n_inputs; }

		int get_n_outputs() const { return n_outputs; }

		int get_n_training_sets() const { return training_data.first.cols(); }

		int get_n_validation_sets() const { return validation_data.first.cols(); }

		int get_n_test_sets() const { return test_data.first.cols(); }

	protected:
		Sets training_data;
		Sets validation_data;
		Sets test_data;

		int n_inputs;
		int n_outputs;
};

class MNIST : public Data {
	public:
		MNIST(const std::string& dir_name, int training_split, int validation_split);

		void show_data(const VectorXd& data) const override;

	private:
		uint32_t reverse_int(uint32_t& n);

		MatrixXd read_mnist_images(const std::string& file_name);

		MatrixXd read_mnist_labels(const std::string& file_name);
};

class CSV : public Data {
	public:
		CSV(const std::string& dir_name, int training_split, int validation_split);

		void show_data(const VectorXd& data) const override;

	private:
		MatrixXd read_csv(const std::string& file_name);
};

#endif
