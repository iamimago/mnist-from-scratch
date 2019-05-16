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
struct layer;
struct digit_label;
struct digit_img;
class network;

float dot_product(vector<float> vec0, vector<float> vec1);
float dot_product(vector<neuron> vec0, vector<float> vec1);
vector<float> vector_addition(vector<float> vec0, vector<float> vec1);
float sigmoid(float n);
float sigmoid_prime(float n);
void print_network(network n);
void print_layers(vector<layer> l_l);
void print_digit(digit_img img, digit_label lbl);