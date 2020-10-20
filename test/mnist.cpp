#include "data.hpp"
#include "network.hpp"

#include <cstdlib>

using namespace std;

int main()
{
	MNIST data("data/mnist-fashion", 50000, 10000);

	vector<Layer> layers;
	layers.emplace_back(Layer(784, 30, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(30, 10, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(0.5, 30, 10, make_unique<CrossEntropy>(), 0.1, true, false);

	/* mnist-fashion map */
	map<int, string> map;
	map[0] = "T-shirt/top";
	map[1] = "Trouser";
	map[2] = "Pullover";
	map[3] = "Dress";
	map[4] = "Coat";
	map[5] = "Sandal";
	map[6] = "Shirt";
	map[7] = "Sneaker";
	map[8] = "Bag";
	map[9] = "T-shirt/top";

	net.test(1, map);
}

