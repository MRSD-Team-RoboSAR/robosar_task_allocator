#include "functions.h"

// rdm class, for gentaring random flot numbers
rdm::rdm() { i = time(0); }

float rdm::randomize()
{
    i = i + 1;
    srand(i);
    return float(rand()) / float(RAND_MAX);
}

// Norm function
float Norm(std::pair<float, float> x1, std::pair<float, float> x2)
{
    return pow((pow((x2.first - x1.first), 2) + pow((x2.second - x1.second), 2)), 0.5);
}

// sign function
float sign(float n)
{
    if (n < 0.0)
    {
        return -1.0;
    }
    else
    {
        return 1.0;
    }
}
