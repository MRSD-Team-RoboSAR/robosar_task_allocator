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
float Norm(std::vector<float> x1, std::vector<float> x2)
{
    return pow((pow((x2[0] - x1[0]), 2) + pow((x2[1] - x1[1]), 2)), 0.5);
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
