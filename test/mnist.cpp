#include "data.hpp"
#include "network.hpp"

using namespace std;

int main()
{
	MNIST data("data/mnist", {50000, 10000});

	vector<Layer> layers;
	layers.emplace_back(Layer(784, 100, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(100, 10, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(0.5, 1, 10, make_unique<CrossEntropy>(), true, true);

	net.test(1);
}
