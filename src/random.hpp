#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <algorithm>
#include <Eigen/Dense>

#include <iostream>

class RandomNumberGenerator {
	public:
		using VectorXd = Eigen::VectorXd;
		using MatrixXd = Eigen::MatrixXd;

		RandomNumberGenerator() : gen{std::random_device()()}, dist{0.0, 1.0} {}

		double operator()() {return dist(gen);}

		VectorXd operator()(int ni) {
			return VectorXd::NullaryExpr(ni, [&](){return dist(gen);});
		}

		MatrixXd operator()(int ni, int nj) {
			return MatrixXd::NullaryExpr(ni, nj, [&](){return dist(gen);});
		}

		std::vector<int> random_indices(int n) {
			std::vector<int> index(n);
			std::iota(std::begin(index), std::end(index), 0);
			std::shuffle(std::begin(index), std::end(index), gen);
			return index;
		}

	private:
		std::mt19937 gen;
		std::normal_distribution<double> dist;
};

extern RandomNumberGenerator rng;

#endif
