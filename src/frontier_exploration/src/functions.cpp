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
float Norm(float x1, float y1, float x2, float y2)
{
    return pow((pow((x2 - x1), 2) + pow((y2 - y1), 2)), 0.5);
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
