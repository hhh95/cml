#include "data.hpp"
#include "network.hpp"

using namespace std;

int main()
{
	MNIST data("data/mnist-fashion", 5);

	vector<Layer> layers;
	layers.emplace_back(Layer(784, 100, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(100, 10, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(0.5, 20, 15, make_unique<CrossEntropy>(), true, true);

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
