#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <map>
#include <string>
#include <memory>
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
		using MatrixXd = Eigen::MatrixXd;

		virtual MatrixXd eval(MatrixXd x) const = 0;

		virtual MatrixXd deriv(MatrixXd x) const = 0;

		virtual std::string get_name() const = 0;
};

class Sigmoid : public Sigma {
	public:
		MatrixXd eval(MatrixXd x) const override {
			return 1.0/(1.0 + exp(-x.array()));
		}

		MatrixXd deriv(MatrixXd x) const override {
			return exp(x.array())/pow(exp(x.array()) + 1.0, 2);
		}

		std::string get_name() const override { return "Sigmoid"; };
};

class TanH : public Sigma {
	public:
		MatrixXd eval(MatrixXd x) const override {
			return tanh(x.array());
		}

		MatrixXd deriv(MatrixXd x) const override {
			return 1 - pow(tanh(x.array()), 2);
		}

		std::string get_name() const override { return "TanH"; };
};

class SoftPlus : public Sigma {
	public:
		MatrixXd eval(MatrixXd x) const override {
			return log(1 + exp(x.array()));
		}

		MatrixXd deriv(MatrixXd x) const override {
			return 1.0/(1 + exp(-x.array()));
		}

		std::string get_name() const override { return "SoftPlus"; };
};

class ReLU : public Sigma {
	public:
		MatrixXd eval(MatrixXd x) const override {
			return x.unaryExpr([&](double x){ return (x > 0.0 ? x : 0.0); });
		}

		MatrixXd deriv(MatrixXd x) const override {
			return x.unaryExpr([&](double x){ return (x > 0.0 ? 1.0 : 0.0); });
		}

		std::string get_name() const override { return "ReLU"; };
};


class Layer {
	public:
		using VectorXd = Eigen::VectorXd;
		using MatrixXd = Eigen::MatrixXd;

		Layer(int n_inputs, int n_outputs, std::unique_ptr<Sigma> sigma) :
			n_inputs{n_inputs}, n_outputs{n_outputs}
		{
			this->sigma = std::move(sigma);

			W = rng(n_outputs, n_inputs)/sqrt(n_inputs);
			b = rng(n_outputs);
		}

		MatrixXd feed_forward(const MatrixXd& a_in) {
			this->a_in = a_in;
			z = W*a_in + b.asDiagonal()*MatrixXd::Ones(b.rows(), a_in.cols());
			return sigma->eval(z);
		}

		MatrixXd feed_backward(const MatrixXd& dC_da_out, double alpha, double lambda,
				double n) {
			MatrixXd delta = dC_da_out.cwiseProduct(sigma->deriv(z));
			MatrixXd _dC_da_out = W.transpose()*delta;

			MatrixXd dC_dW = delta*a_in.transpose();
			VectorXd dC_db = delta.rowwise().sum();

			W -= alpha/a_in.cols()*dC_dW + alpha*lambda/n*W;
			b -= alpha/a_in.cols()*dC_db;

			return _dC_da_out;
		}

		const int n_inputs;
		const int n_outputs;

		std::unique_ptr<Sigma> sigma;

	private:
		MatrixXd W, a_in, z;
		VectorXd b;
};


class Cost {
	public:
		using MatrixXd = Eigen::MatrixXd;

		virtual double eval(MatrixXd a, MatrixXd y) const = 0;

		virtual MatrixXd deriv(MatrixXd a, MatrixXd y) const = 0;

		virtual std::string get_name() const = 0;
};

class MSE : public Cost {
	public:
		double eval(MatrixXd a, MatrixXd y) const override {
			return 0.5*(a - y).colwise().squaredNorm().sum();
		}

		MatrixXd deriv(MatrixXd a, MatrixXd y) const override {
			return (a - y);
		}

		std::string get_name() const override { return "Mean Squared Error"; };
};

class CrossEntropy : public Cost {
	public:
		double eval(MatrixXd a, MatrixXd y) const override {
			MatrixXd tmp = -(y.array()*log(a.array()) + (1 - y.array())*log(1 - a.array()));
			tmp = tmp.unaryExpr([&](double x){ return (std::isfinite(x) ? x : 0.0); });
			return tmp.sum();
		}

		MatrixXd deriv(MatrixXd a, MatrixXd y) const override {
			MatrixXd tmp = -(a.array() - y.array())/(a.array() * (a.array() - 1));
			return tmp.unaryExpr([&](double x){ return (std::isfinite(x) ? x : 0.0); });
		}

		std::string get_name() const override { return "Cross Entropy"; };
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

		void train(double alpha, int epochs, int batch_size, std::shared_ptr<Cost> cost,
				double lambda,
				bool do_tests_inbetween = false, bool do_validation_inbetween = false);

		void test(int n_incorrect, const std::map<int, std::string>& map = {}) const;

	private:
		Data& data;
		std::vector<Layer>& layers;

		std::chrono::time_point<std::chrono::high_resolution_clock> wtime_start;

		void _validate(std::shared_ptr<Cost> cost, std::ofstream& fout) const;

		void _test(std::shared_ptr<Cost> cost, std::ofstream& fout) const;
};


#endif
