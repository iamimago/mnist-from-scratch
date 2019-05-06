#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <string>

using namespace std;
struct neuron;

float dot_product(vector<float> vec0, vector<float> vec1);
float dot_product(vector<neuron> vec0, vector<float> vec1);
float sigmoid(float n);