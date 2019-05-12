/* 
    TODO: 
        Train and valid data set split - DONE
        Cost function implementation - DONE... but not implemented
        Back propagataion
        DONE

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
const int SIZE_HIDDEN_LAYER = 20;
const int BATCH_SIZE = 200;
const float SPLIT_FACTOR = 0.1;
float LEARNING_RATE = 3;

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
            float r0 = ((float)rand() / (RAND_MAX));
            float r1 = ((float)rand() / (RAND_MAX));
            l_tmp.n_list.push_back({0, 0});
        }
        layer_list.push_back(l_tmp);
        l_tmp.n_list.clear();

        //For every hidden layer in AM_HIDDEN_LAYERS, fix the hidden layer.
        for (int i0 = 0; i0 < AM_HIDDEN_LAYERS; i0++)
        {
            for (int i1 = 0; i1 < SIZE_HIDDEN_LAYER; i1++)
            {
                float r0 = ((float)rand() / (RAND_MAX));
                float r1 = ((float)rand() / (RAND_MAX));
                neuron n_tmp = {r0, r1};
                l_tmp.n_list.push_back(n_tmp);
            }
            layer_list.push_back(l_tmp);
        }
        l_tmp.n_list.clear();

        //Fix output layer fixed size 0-9
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            l_tmp.n_list.push_back({0, 0});
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
                    float r = random ? ((float)rand() / (RAND_MAX)) : 0;
                    layer_list[i0].n_list[i1].con_weights.push_back(r);
                }
            }
        }
    }

    void print_network(network n, int spec_layer = -1)
    {
        int layers = n.layer_list.size();
        cout << "Layer am: " << layers << endl;

        cout << "Neurons in layer: \n\n";
        if (spec_layer >= 0)
        {
            cout << spec_layer << ": \n";
            layer l_tmp = n.layer_list[spec_layer];
            for (int i1 = 0; i1 < l_tmp.n_list.size(); i1++)
            {
                string delim = i1 % 2 ? "\n" : "\t";
                cout << "n: " << i1 << " v: " << (int)l_tmp.n_list[i1].value << " b: " << l_tmp.n_list[i1].bias << " am_w: " << l_tmp.n_list[i1].con_weights.size() << delim;
            }
            cout << "\n\n";
        }
        else
        {
            for (int i = 1; i < layers; i++)
            {
                cout << i << ": \n";
                layer l_tmp = n.layer_list[i];
                for (int i1 = 0; i1 < l_tmp.n_list.size(); i1++)
                {
                    string delim = i1 % 2 ? "\n" : "\t";
                    cout << "n: " << i1 << " v: " << l_tmp.n_list[i1].value << " b: " << l_tmp.n_list[i1].bias << " am_w: " << l_tmp.n_list[i1].con_weights.size() << delim;
                }
                cout << "\n\n";
            }
        }
    }

    network feed_forward(digit_img img)
    {
        if (sizeof(img.pixel) != layer_list[0].n_list.size())
        {
            cout << "Input size of images doens't match network input size." << endl;
            throw stderr;
        }
        network weighted_input;

        //Feed img-pixel-data into input layer
        for (int i = 0; i < sizeof(img.pixel); i++)
        {
            int pixel_val = img.pixel[i];
            layer_list[0].n_list[i].value = img.pixel[i];
            weighted_input.layer_list[0].n_list[i].value = img.pixel[i];
        }

        //For every layer (except input)
        for (int i0 = 1; i0 < layer_list.size(); i0++)
        {
            //For every node in layer
            for (int i1 = 0; i1 < layer_list[i0].n_list.size(); i1++)
            {
                vector<neuron> prev_layer_nodes = layer_list[i0 - 1].n_list;
                vector<float> curr_node_weights = layer_list[i0].n_list[i1].con_weights;
                float curr_node_bias = layer_list[i0].n_list[i1].bias;

                float z = dot_product(prev_layer_nodes, curr_node_weights) + curr_node_bias;
                float a = sigmoid(z);
                layer_list[i0].n_list[i1].value = a;
                weighted_input.layer_list[i0].n_list[i1].value = z;
            }
        }

        //Return output layer
        return weighted_input; 
    }

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

    //Used for stochastic error calculation before gradient descent
    network network_addition(network net0, network net1)
    {
        network err_net;
        for (int l_l = 0; l_l < net0.layer_list.size(); l_l++)
        {
            for (int n_l = 0; n_l < net0.layer_list[l_l].n_list.size(); n_l++)
            {
                neuron *net0_n = &net0.layer_list[l_l].n_list[n_l];
                neuron *net1_n = &net1.layer_list[l_l].n_list[n_l];

                err_net.layer_list[l_l].n_list[n_l].value = net0_n->value + net1_n->value;
                err_net.layer_list[l_l].n_list[n_l].bias = net0_n->bias + net1_n->bias;
                err_net.layer_list[l_l].n_list[n_l].con_weights = vector_addition(net0_n->con_weights, net1_n->con_weights);
            }
        }
        return err_net;
    }

    network backprop(digit_label label, network weighted_input_net)
    {
        //Layer_list in the network is the current set of nodes/weighs/biases.
        network err_net;
        int output_layer = layer_list.size() -1;
        int output_layer_size = layer_list[output_layer].n_list.size();

        //Calculate error in layer L
        for(int n_l = 0; n_l < output_layer_size; n_l++){
            float a = layer_list[output_layer].n_list[n_l].value;
            float y_x = label.num[n_l];
            float z = weighted_input_net.layer_list[output_layer].n_list[n_l].value;

            err_net.layer_list[output_layer].n_list[n_l].value = (a-y_x) * sigmoid_prime(z);
            err_net.layer_list[output_layer].n_list[n_l].bias = (a-y_x) * sigmoid_prime(z);
        }

        //Calculate weights connecting the output layer
        for(int n_l = 0; n_l < output_layer_size; n_l++)
        {
            for (int w_l = 0; w_l < err_net.layer_list[output_layer].n_list[n_l].con_weights.size(); w_l++)
            {
                err_net.layer_list[output_layer].n_list[n_l].con_weights[w_l] = layer_list[output_layer -1].n_list[w_l].value * err_net.layer_list[output_layer].n_list[n_l].value;
            }
        }

        //Calculate the rest of the weights and biases
        for(int l = output_layer - 1; l > 0; l--)
        {
           for(int n_l = 0; n_l < layer_list[l].n_list.size(); n_l++)
           {
               float z = weighted_input_net.layer_list[l].n_list[n_l].value;
               float delta = 0.0;
               for(int w_l = 0; w_l < layer_list[l+1].n_list.size(); w_l++){
                   //weight is stored in the node _pointing backwards_ to each of the previous node in the list. node n in layer 2 has a weight list pointing at every node in layer 1. 
                   //therefore layer_list[l+1] (layer ahead).node_list[w_l] (for each node in this layer).get weight pointing at node in current list [n_l]
                   //should work. a bit messy, perhaps.
                   float w = layer_list[l+1].n_list[w_l].con_weights[n_l];
                   float delta_next_l = err_net.layer_list[l+1].n_list[w_l].value;
                   delta += w*delta_next_l*sigmoid_prime(z);
               }
               err_net.layer_list[l].n_list[n_l].value = delta;
           } 
        }

        return err_net;
    }

    //In place gradient descent, updating local layer_list's weights and biases
    void gradient_descent(network err_net)
    {
        int output_layer = layer_list.size() - 1;
        //Starting at output layer (for no particular reason)
        for(int l = output_layer; l > 0; l--)
        {
            for(int n = 0; n < layer_list[l].n_list.size(); n++)
            {
                for(int w = 0; w < layer_list[l].n_list[n].con_weights.size(); w++)
                {
                    float a = err_net.layer_list[l-1].n_list[w].value;
                    float delta = err_net.layer_list[l].n_list[n].value;
                    layer_list[l].n_list[n].con_weights[w] = layer_list[l].n_list[n].con_weights[w] - (LEARNING_RATE/BATCH_SIZE) * (a * delta); 
                }
                layer_list[l].n_list[n].bias = layer_list[l].n_list[n].bias - (LEARNING_RATE/BATCH_SIZE) * err_net.layer_list[l].n_list[n].value;
            }
        }
    }

    vector<float> argmax(vector<neuron> output)
    {
        vector<float> ret(10);
        float tmp = 0;
        int pred = 0;
        for(int i = 0; i < output.size(); i++){
            if(output[i].value > tmp)
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
        for(int i = 0; i < l_list.size(); i++)
        {
            if(l_list[i] == 1.0)
            {
                if(max[i] == 1)
                {
                    return 1;
                }
            }
        }
        return 0;
    }

    void train(vector<digit_img> img_list, vector<digit_label> label_list, int epochs, float learning_rate)
    {
        network weighted_input_net;
        float loss = 0;
        float hit = 0;
        float runs = 0;

        for (int i0 = 0; i0 < epochs; i0++)
        {
            cout << " - Epoch: " << i0 << " -\nbatch: " << flush;

            network stochastic_error;
            for (int i1 = 0; i1 < (int)(img_list.size() / BATCH_SIZE); i1++)
            {
                cout << i1 << " " << flush;
                for (int i2 = 0; i2 < BATCH_SIZE; i2++)
                {
                    int rand_indx = rand() % img_list.size();

                    weighted_input_net = feed_forward(img_list[rand_indx]); //<- Updates internal network layer-list node values, which is saved when entering back propagation. Weights, biases and input sets activations
                    hit += target_hit(layer_list[layer_list.size() - 1].n_list, label_list[rand_indx].num);
                    stochastic_error = network_addition(backprop(label_list[rand_indx], weighted_input_net), stochastic_error);
                    runs++;
                }
            }
            gradient_descent(stochastic_error);

            cout << " Accuracy: " <<  (float) hit/runs << endl;
        }
    }

    void print_output(vector<neuron> output_layer)
    {

        cout << "Output layer:" << endl;
        for (int i = 0; i < output_layer.size(); i++)
        {
            neuron n = output_layer[i];
            cout << i << ": " << n.value << endl;
        }
    }
};

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
}

float dot_product(vector<float> vec0, vector<float> vec1)
{
    float sum = 0;
    if (vec0.size() != vec1.size())
    {
        cout << "Dot product error: vector sizes doesn't match." << endl;
        throw stderr;
    }

    for (int i = 0; i < vec0.size(); i++)
    {
        sum += vec0[i] * vec1[i];
    }
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

network test()
{
    network ret = new network();
    return ret;
}

int main()
{
    srand(time(NULL));

    run();

    return 0;
}