/* 
    TODO: 
        Consider suicide
        Cry a little
        Go through the backpropagation function for the 21th time
        See if the labels actually are matching the images

 */
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <time.h>
#include <string>
#include "main.h"
using namespace std;

const int INPUT_SIZE = 28 * 28;
const int OUTPUT_SIZE = 10;
const int AM_HIDDEN_LAYERS = 1;
const int SIZE_HIDDEN_LAYER = 30;
const int BATCH_SIZE = 10;
const float SPLIT_FACTOR = 0.1;
const bool VERBOSE = false; 
float LEARNING_RATE = 3.0;

struct digit_img
{
    unsigned char pixel[784];
};

struct digit_label
{
    vector<float> num;
};

struct neuron
{
    float value;
    float bias;
    vector<float> con_weights;
};

struct layer
{
    vector<neuron> n_list;
};

class network
{
public:
    //A list of all the layers in the network. Input is layer_list[0], output is layer_list[layer_list.size()-1] (depends on the amount of hidden layers)
    vector<layer> layer_list;

    network(bool random = true)
    {
        layer l_tmp;

        //Fix input layer to size of 28*28 pixel grid img
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            l_tmp.n_list.push_back({0, 0});
        }
        layer_list.push_back(l_tmp);
        l_tmp.n_list.clear();

        //For every hidden layer in AM_HIDDEN_LAYERS, fix the hidden layer.
        for (int i0 = 0; i0 < AM_HIDDEN_LAYERS; i0++)
        {
            for (int i1 = 0; i1 < SIZE_HIDDEN_LAYER; i1++)
            {
                float r0 = random ? ((float)rand() / (RAND_MAX)*4) - 2 : 0;
                float r1 = random ? ((float)rand() / (RAND_MAX)*4) - 2 : 0;
                neuron n_tmp = {r0, r1};
                l_tmp.n_list.push_back(n_tmp);
            }
            layer_list.push_back(l_tmp);
        }
        l_tmp.n_list.clear();

        //Fix output layer fixed size 0-9, random bias 
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            float r0 = random ? ((float)rand() / (RAND_MAX)*4) - 2 : 0;
            l_tmp.n_list.push_back({0, r0});
        }

        layer_list.push_back(l_tmp);

        //When layer setup is done, loop through all layers except first one, set up weights (yes ineffective i'm tired leave me alone)
        for (int i0 = 1; i0 < layer_list.size(); i0++)
        {
            int index_prevlayer = i0 - 1;
            int size_prevlayer = layer_list[index_prevlayer].n_list.size();

            //For each node in layer
            for (int i1 = 0; i1 < layer_list[i0].n_list.size(); i1++)
            {
                //Add random weight for each node in previous layer
                for (int i2 = 0; i2 < size_prevlayer; i2++)
                {
                    float r = random ? ((float)rand() / (RAND_MAX)*2) - 1.0 : 0;
                    layer_list[i0].n_list[i1].con_weights.push_back(r);
                }
            }
        }
    }

    //Used for stochastic error calculation before gradient descent
    network network_addition(network net0, network net1)
    {
        network ret_net;
        for (int l_l = 0; l_l < net0.layer_list.size(); l_l++)
        {
            for (int n_l = 0; n_l < net0.layer_list[l_l].n_list.size(); n_l++)
            {
                neuron net0_n = net0.layer_list[l_l].n_list[n_l];
                neuron net1_n = net1.layer_list[l_l].n_list[n_l];

                ret_net.layer_list[l_l].n_list[n_l].value = net0_n.value + net1_n.value;
                ret_net.layer_list[l_l].n_list[n_l].bias = net0_n.bias + net1_n.bias;
                ret_net.layer_list[l_l].n_list[n_l].con_weights = vector_addition(net0_n.con_weights, net1_n.con_weights);
            }
        }
        return ret_net;
    }

    network feed_forward(digit_img img)
    {
        network pre_sig(false);
        float z = 0;
        //Load pixels into the network's first layer
        for (int p = 0; p < sizeof(img.pixel); p++)
        {
            layer_list[0].n_list[p].value = sigmoid((int) img.pixel[p]);
        }

        for(int l_l = 1; l_l < layer_list.size(); l_l++)
        {
            for(int n_l = 0; n_l < layer_list[l_l].n_list.size(); n_l++)
            {
                vector<neuron> prev_l = layer_list[l_l-1].n_list;
                vector<float> w = layer_list[l_l].n_list[n_l].con_weights;
                float b = layer_list[l_l].n_list[n_l].bias;
                z = dot_product(prev_l, w) + b;

                pre_sig.layer_list[l_l].n_list[n_l].value = z;
                layer_list[l_l].n_list[n_l].value = sigmoid(z);
            }
        }
        return pre_sig;
    }

    network backprop(digit_label label, network z_net)
    {
        network delta_nabla(false);
        int o_layer = layer_list.size() - 1;

        //Calculate error in the output layer
        for(int n_l = 0; n_l < layer_list[o_layer].n_list.size(); n_l++)
        {

        }

        return delta_nabla;
    }

    //In place gradient descent, updating local layer_list's weights and biases
    void gradient_descent(network err_net, int batch_run)
    {
    }

    void train(vector<digit_img> img_list, vector<digit_label> label_list, int epochs, float learning_rate)
    {
        network z_net;
        float loss = 0;
        float hit = 0;
        float runs = 0;

        for (int e = 0; e < epochs; e++)
        {
            cout << " - Epoch: " << e << " - \n" << flush;
            if(VERBOSE) cout << "batch: " << flush;

            network delta_nabla(false);
            for (int b = 0; b < (int)(img_list.size() / BATCH_SIZE); b++)
            {
                if(VERBOSE) cout << b << " " << flush;
                int batch_run = 0;
                for (int p = 0; p < BATCH_SIZE; p++)
                {
                    int rand_indx = rand() % img_list.size();
                    z_net = feed_forward(img_list[rand_indx]);
                    //TODO: When done: try different weight initializations to find out what works optimally.

                    delta_nabla = network_addition(backprop(label_list[rand_indx], z_net), delta_nabla);
               }
            }
            cout << " Accuracy: " << (float)hit / runs << endl;
        }
    }
};

float mean_squared_error(vector<neuron> output, vector<float> label)
{
    if (output.size() != label.size())
    {
        cout << "Size of the output layer and the label is not the same." << endl;
        throw stderr;
    }

    float loss = 0;
    int n = output.size();
    for (int i = 0; i < n; i++)
    {
        loss += pow(label[i] - output[i].value, 2);
    }

    return loss;
}

vector<float> argmax(vector<neuron> output)
{
    vector<float> ret(10);
    float tmp = 0;
    int pred = 0;
    for (int i = 0; i < output.size(); i++)
    {
        if (output[i].value > tmp)
        {
            tmp = output[i].value;
            pred = i;
        }
    }
    ret[pred] = 1;
    return ret;
}

int target_hit(vector<neuron> n_list, vector<float> l_list)
{
    vector<float> max = argmax(n_list);
    for (int i = 0; i < l_list.size(); i++)
    {
        if (l_list[i] == 1.0)
        {
            if (max[i] == 1)
            {
                return 1;
            }
        }
    }
    return 0;
}

void print_network(network n)
{
    int layers = n.layer_list.size();
    cout << "Printing network:"<<endl;
    for (int i = 1; i < layers; i++)
    {
        cout << i << ":\n";
        layer l_tmp = n.layer_list[i];
        for (int i1 = 0; i1 < l_tmp.n_list.size(); i1++)
        {
            string delim = i1 % 2 ? "\n" : "\t";
            cout << "n: " << i1 << " v: " << l_tmp.n_list[i1].value << "\tb: " << l_tmp.n_list[i1].bias << " am_w: " << l_tmp.n_list[i1].con_weights.size() << delim;
        }
        cout << "\n\n";
    }

    cout << "weights for first node in layer " << layers -1 << endl;
    for (int w = 0; w < n.layer_list[layers -1].n_list[0].con_weights.size(); w++)
    {
        cout << "w" << w << ": " << n.layer_list[layers -1].n_list[0].con_weights[w] << "\t" << flush;
    }
    cout << endl;
}

void print_layers(vector<layer> l_list)
{
    cout << "\nlayer print:\n";
    int e_l = l_list.size() -1;
    for(int l_l = 1; l_l < l_list.size(); l_l++)
    {
        cout << "l: " << l_l << endl;
        for(int n_l = 0; n_l < l_list[l_l].n_list.size(); n_l++)
        {
            cout << fixed << "n: "<<n_l<<" v: "<< l_list[l_l].n_list[n_l].value<<" b: "<<l_list[l_l].n_list[n_l].bias<<"\t"<<flush; 
            if(n_l % 2 == 0) cout << endl;
        }
        if (l_l == l_list.size() - 1)
        {
            cout << "\n\nweights for first node in layer " << e_l << endl;
            for (int w = 0; w < l_list[e_l].n_list[0].con_weights.size(); w++)
            {
                cout << "w" << w << ": " << l_list[e_l].n_list[0].con_weights[w] << "\t" << flush;
            }
            cout << endl;
       }
        cout << "\n\n";
    }
}

float dot_product(vector<neuron> vec0, vector<float> vec1)
{
    float sum = 0;
    if (vec0.size() != vec1.size())
    {
        cout << "Dot product error: vector sizes doesn't match." << endl;
        throw stderr;
    }

    for (int i = 0; i < vec0.size(); i++)
    {
        sum += vec0[i].value * vec1[i];
    }
    return sum;
}

vector<float> vector_addition(vector<float> vec0, vector<float> vec1)
{
    if (vec0.size() != vec1.size())
    {
        cout << "Error in vector addition: vectors of different sizes";
        throw stderr;
    }

    vector<float> ret;
    for (int i = 0; i < vec0.size(); i++)
    {
        ret.push_back(vec0[i] + vec1[i]);
    }

    return ret;
}

float sigmoid(float n)
{
    return 1 / (1 + (exp(-n)));
}

float sigmoid_prime(float n)
{
    return sigmoid(n) * (1 - sigmoid(n));
}

int reverse_int(int i)
{
    unsigned char c0, c1, c2, c3;

    c0 = i & 255;
    c1 = (i >> 8) & 255;
    c2 = (i >> 16) & 255;
    c3 = (i >> 24) & 255;

    return ((int)c0 << 24) + ((int)c1 << 16) + ((int)c2 << 8) + c3;
}

vector<float> one_hot_encode(float n)
{
    vector<float> ret = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ret[n] = 1;
    return ret;
}

vector<vector<digit_img>> get_img_data(char *fname, float split_am)
{
    vector<vector<digit_img>> ret;
    vector<digit_img> training, valid;

    //Read the file data
    ifstream file(fname);
    while (file.is_open())
    {
        int magic_num = 0;
        int am_img = 0;
        int am_rows = 0;
        int am_cols = 0;

        file.read((char *)&magic_num, sizeof(magic_num));
        magic_num = reverse_int(magic_num);

        file.read((char *)&am_img, sizeof(am_img));
        am_img = reverse_int(am_img);

        file.read((char *)&am_rows, sizeof(am_rows));
        am_rows = reverse_int(am_rows);

        file.read((char *)&am_cols, sizeof(am_cols));
        am_cols = reverse_int(am_cols);

        for (int i0 = 0; i0 < am_img; i0++)
        {
            digit_img tmp;
            for (int i1 = 0; i1 < am_cols; i1++)
            {
                for (int i2 = 0; i2 < am_rows; i2++)
                {
                    file.read((char *)&tmp.pixel[i2 + i1 * 28], sizeof(char));
                }
            }
            if (i0 < (int)(split_am * am_img))
            {

                valid.push_back(tmp);
            }
            else
            {
                training.push_back(tmp);
            }
        }
        file.close();
    }

    ret.push_back(training);
    ret.push_back(valid);

    return ret;
}

vector<vector<digit_label>> get_label_data(char *fname, float split_am)
{
    vector<vector<digit_label>> ret;
    vector<digit_label> training, valid;

    ifstream file(fname);

    while (file.is_open())
    {
        int magic_num = 0;
        int am_img = 0;

        file.read((char *)&magic_num, sizeof(magic_num));
        magic_num = reverse_int(magic_num);

        file.read((char *)&am_img, sizeof(am_img));
        am_img = reverse_int(am_img);

        for (int i = 0; i < am_img; i++)
        {
            char tmp = -1;
            file.read((char *)&tmp, sizeof(char));
            digit_label label = {one_hot_encode((float)tmp)};
            if (i < (int)(split_am * am_img))
            {
                valid.push_back(label);
            }
            else
            {
                training.push_back(label);
            }
        }
        file.close();
    }

    ret.push_back(training);
    ret.push_back(valid);

    return ret;
}

void print_digit(digit_img img, digit_label lbl)
{
    cout<< "Printing digit:"<<endl;
    for(int p = 1; p < sizeof(img.pixel)+1; p++)
    {
        char sign = (int) img.pixel[p-1] == 0 ? ' ':'x'; 
        cout << sign << " " << flush;
        if(p % 28 == 0 && p != 0) cout << endl;
    }
    int label_num= -1;
    for(int i = 0; i < lbl.num.size(); i++)
    {
        if(lbl.num[i] == 1)
        {
            label_num = i;
        }
    }
    cout << "\nLabel: " << label_num << endl;
}

void run()
{
    cout << "Loading images and labels..." << endl;
    auto test_imgs = get_img_data((char *)"mnist/test-images/t10k-images.idx3-ubyte", SPLIT_FACTOR);
    auto test_labels = get_label_data((char *)"mnist/test-labels/t10k-labels.idx1-ubyte", SPLIT_FACTOR);
    auto all_imgs = get_img_data((char *)"mnist/training-images/train-images.idx3-ubyte", SPLIT_FACTOR);
    auto all_labels = get_label_data((char *)"mnist/training-labels/train-labels.idx1-ubyte", SPLIT_FACTOR);

    auto valid_imgs = all_imgs[0];
    auto valid_labels = all_labels[0];
    auto training_imgs = all_imgs[1];
    auto training_labels = all_labels[1];

    cout << "Beginning training..." << endl;

    //Network initialized by global variables
    network agent;

    agent.train(training_imgs, training_labels, 5, LEARNING_RATE);
}

int main()
{
    srand(time(NULL));
    cout.precision(4);

    run();

    return 0;
}