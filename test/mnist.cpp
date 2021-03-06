#include "data.hpp"
#include "network.hpp"

#include <cstdlib>

using namespace std;

int main()
{
	MNIST data("data/mnist", 50000, 10000);

	vector<Layer> layers;
	layers.emplace_back(Layer(784, 30, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(30, 10, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(0.5, 30, 10, make_unique<CrossEntropy>(), 0.1, true, false);

	net.test(1);
}
