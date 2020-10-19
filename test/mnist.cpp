#include "data.hpp"
#include "network.hpp"

using namespace std;

int main()
{
	MNIST data("data/mnist", {50000, 10000, 10000});

	vector<Layer> layers;
	layers.emplace_back(Layer(784, 30, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(30, 10, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(3.0, 30, 10, make_unique<MSE>(), true);
}
