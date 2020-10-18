#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <Eigen/Dense>

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
			std::vector<int> index;
			index.reserve(n);

			for (int i = 0; i < n; ++i)
				index[i] = i;

			std::shuffle(std::begin(index), std::end(index), gen);

			return index;
		}

		template<typename T>
		void shuffle(std::vector<T>& v) {
			std::shuffle(std::begin(v), std::end(v), gen);
		}

	private:
		std::mt19937 gen;
		std::normal_distribution<double> dist;
};

extern RandomNumberGenerator rng;

#endif
