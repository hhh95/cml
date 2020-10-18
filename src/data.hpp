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
		using TrainSet = std::pair<VectorXd, VectorXd>;
		using TestSet = std::pair<VectorXd, int>;
		using Batch = std::vector<TrainSet>;

		Data(const std::string& dir_name, const std::vector<int>& splits) :
			splits{splits}
		{
			std::cout << "Reading data from '" << dir_name << "':" << std::endl;
		}

		virtual void show_data(const TestSet& set) const = 0;

		virtual void show_data(const TrainSet& set) const = 0;

		void shuffle_training_data() { rng.shuffle<TrainSet>(training_data); }

		const std::vector<Batch> get_training_batches(int batch_size) const {
			std::vector<Batch> batches;

			int i_start = 0;
			int i_end = i_start + batch_size;

			while (i_end < get_n_training_sets()) {
				Batch batch = {&training_data[i_start], &training_data[i_end]};

				batches.emplace_back(batch);

				i_start = i_end;
				i_end = i_start + batch_size;
			}

			if (i_start < get_n_training_sets()) {
				i_end = get_n_training_sets();

				Batch batch = {&training_data[i_start], &training_data[i_end]};

				batches.emplace_back(batch);
			}

			return batches;
		}

		const std::vector<TrainSet>& get_training_data() const {
			return training_data;
		}

		const std::vector<TestSet>& get_validation_data() const {
			return validation_data;
		}

		const std::vector<TestSet>& get_test_data() const {
			return test_data;
		}

		int get_n_inputs() const { return n_inputs; }

		int get_n_outputs() const { return n_outputs; }

		int get_n_training_sets() const { return training_data.size(); }

		int get_n_validation_sets() const { return validation_data.size(); }

		int get_n_test_sets() const { return test_data.size(); }

		const std::vector<int>& splits;

	protected:
		std::vector<TrainSet> training_data;
		std::vector<TestSet> validation_data;
		std::vector<TestSet> test_data;

		int n_inputs;
		int n_outputs;
};

class MNIST : public Data {
	public:

		MNIST(const std::string& dir_name, const std::vector<int>& splits);

		void show_data(const TestSet& set) const override;

		void show_data(const TrainSet& set) const override {
			int label; set.second.maxCoeff(&label);
			show_data(std::make_pair(set.first, label));
		}

	private:

		uint32_t reverse_int(uint32_t& n);

		std::vector<VectorXd> read_mnist_images(const std::string& file_name);

		std::vector<uint8_t> read_mnist_labels(const std::string& file_name);
};

#endif
