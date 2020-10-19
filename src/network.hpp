#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <iomanip>

#include "random.hpp"
#include "data.hpp"

class Sigma {
	public:
		using VectorXd = Eigen::VectorXd;

		virtual VectorXd eval(VectorXd x) const = 0;

		virtual VectorXd deriv(VectorXd x) const = 0;
};

class Sigmoid : public Sigma {
	public:
		VectorXd eval(VectorXd x) const override {
			return 1.0/(1.0 + exp(-x.array()));
		}

		VectorXd deriv(VectorXd x) const override {
			return exp(x.array())/pow(exp(x.array()) + 1.0, 2);
		}
};

class TanH : public Sigma {
	public:
		VectorXd eval(VectorXd x) const override {
			return tanh(x.array());
		}

		VectorXd deriv(VectorXd x) const override {
			return 1 - pow(tanh(x.array()), 2);
		}
};

class SoftPlus : public Sigma {
	public:
		VectorXd eval(VectorXd x) const override {
			return log(1 + exp(x.array()));
		}

		VectorXd deriv(VectorXd x) const override {
			return 1.0/(1 + exp(-x.array()));
		}
};


class Layer {
	public:
		using VectorXd = Eigen::VectorXd;
		using MatrixXd = Eigen::MatrixXd;

		Layer(int n_inputs, int n_outputs, std::unique_ptr<Sigma> sigma) :
			n_inputs{n_inputs}, n_outputs{n_outputs}
		{
			this->sigma = std::move(sigma);

			W = rng(n_outputs, n_inputs);
			b = rng(n_outputs);

			dC_dW = MatrixXd::Zero(n_outputs, n_inputs);
			dC_db = VectorXd::Zero(n_outputs);
		}

		VectorXd feed_forward(const VectorXd& a_in) {
			this->a_in = a_in;
			z = W*a_in + b;
			return sigma->eval(z);
		}

		VectorXd feed_backward(const VectorXd& dC_da_out) {
			VectorXd delta = dC_da_out.asDiagonal()*sigma->deriv(z);

			dC_dW += delta*a_in.transpose();
			dC_db += delta;

			return W.transpose()*delta;
		}

		void update_weights(double alpha, int batch_size) {
			W -= alpha/batch_size*dC_dW;
			b -= alpha/batch_size*dC_db;

			dC_dW.setZero();
			dC_db.setZero();
		}

		const int n_inputs;
		const int n_outputs;

	private:
		MatrixXd W;
		VectorXd b, a_in, z;

		MatrixXd dC_dW;
		VectorXd dC_db;

		std::unique_ptr<Sigma> sigma;

};


class Cost {
	public:
		using VectorXd = Eigen::VectorXd;

		virtual double eval(VectorXd a, VectorXd y) const = 0;

		virtual VectorXd deriv(VectorXd a, VectorXd y) const = 0;
};

class MSE : public Cost {
	public:
		double eval(VectorXd a, VectorXd y) const override {
			return 0.5*(a - y).squaredNorm();
		}

		VectorXd deriv(VectorXd a, VectorXd y) const override {
			return (a - y);
		}
};


class Network {
	public:
		using VectorXd = Eigen::VectorXd;

		Network(Data& data, std::vector<Layer>& layers);

		~Network() {
			/* stop timer */
			auto wtime_now = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> wtime_delta = wtime_now - wtime_start;
			std::cout << "Total time: " << wtime_delta.count() << " s" << std::endl;
		}

		void train(double alpha, int epochs, int batch_size, std::unique_ptr<Cost> cost,
				bool do_tests_inbetween = false);

		void test(int n_incorrect) const;

	private:
		Data& data;
		std::vector<Layer>& layers;

		std::chrono::time_point<std::chrono::high_resolution_clock> wtime_start;

		std::vector<Data::TestSet> _test() const;
};


#endif
