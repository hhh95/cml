#include "data.hpp"
#include "network.hpp"

#include <fenv.h>

using namespace std;

int main()
{
	feenableexcept(FE_INVALID | FE_OVERFLOW);

	CSV data("data/xor", 900, 100);

	vector<Layer> layers;
	layers.emplace_back(Layer(2, 4, make_unique<Sigmoid>()));
	layers.emplace_back(Layer(4, 2, make_unique<Sigmoid>()));

	Network net(data, layers);

	net.train(1.0, 5, 2, make_unique<CrossEntropy>(), 1.0, true, true);
}

