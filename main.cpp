/* 
    TODO: 
        Train and valid data set split - DONE
        Cost function implementation - 
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
float LEARNING_RATE = 0.01;

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
    vector<layer> l_list;

    void init_network()
    {
        layer l_tmp;

        //Fix input layer to size of 28*28 pixel grid img
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            float r0 = ((float)rand() / (RAND_MAX));
            float r1 = ((float)rand() / (RAND_MAX));
            l_tmp.n_list.push_back({0, 0});
        }
        l_list.push_back(l_tmp);
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
            l_list.push_back(l_tmp);
        }
        l_tmp.n_list.clear();

        //Fix output layer fixed size 0-9
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            l_tmp.n_list.push_back({0, 0});
        }

        l_list.push_back(l_tmp);

        //Loop through all layers except first one, set up weights (yes ineffective i'm tired leave me alone)
        for (int i0 = 1; i0 < l_list.size(); i0++)
        {
            int index_prevlayer = i0 - 1;
            int size_prevlayer = l_list[index_prevlayer].n_list.size();

            //For each node in layer
            for (int i1 = 0; i1 < l_list[i0].n_list.size(); i1++)
            {
                //Add random weight for each node in previous layer
                for (int i2 = 0; i2 < size_prevlayer; i2++)
                {
                    float r = ((float)rand() / (RAND_MAX));
                    l_list[i0].n_list[i1].con_weights.push_back(r);
                }
            }
        }
    }

    void print_network(network n, int spec_layer = -1)
    {
        int layers = n.l_list.size();
        cout << "Layer am: " << layers << endl;

        cout << "Neurons in layer: \n\n";
        if (spec_layer >= 0)
        {
            cout << spec_layer << ": \n";
            layer l_tmp = n.l_list[spec_layer];
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
                layer l_tmp = n.l_list[i];
                for (int i1 = 0; i1 < l_tmp.n_list.size(); i1++)
                {
                    string delim = i1 % 2 ? "\n" : "\t";
                    cout << "n: " << i1 << " v: " << l_tmp.n_list[i1].value << " b: " << l_tmp.n_list[i1].bias << " am_w: " << l_tmp.n_list[i1].con_weights.size() << delim;
                }
                cout << "\n\n";
            }
        }
    }

    vector<neuron> feed_forward(digit_img img)
    {
        if (sizeof(img.pixel) != l_list[0].n_list.size())
        {
            cout << "Input size of images doens't match network input size." << endl;
            throw stderr;
        } 
        //Feed img-pixel-data into input layer
        for (int i = 0; i < sizeof(img.pixel); i++)
        {
            int pixel_val = img.pixel[i];
            l_list[0].n_list[i].value = img.pixel[i];
        }

        //For every layer (except input)
        for (int i0 = 1; i0 < l_list.size(); i0++)
        {
            //For every node in layer
            for (int i1 = 0; i1 < l_list[i0].n_list.size(); i1++)
            {
                vector<neuron> prev_layer_nodes = l_list[i0 - 1].n_list;
                vector<float> curr_node_weights = l_list[i0].n_list[i1].con_weights;
                float curr_node_bias = l_list[i0].n_list[i1].bias;

                float new_node_val = dot_product(prev_layer_nodes, curr_node_weights) + curr_node_bias;
                new_node_val = sigmoid(new_node_val);
                l_list[i0].n_list[i1].value = new_node_val;
            }
        }
        
        //Return output layer
        return l_list[l_list.size() - 1].n_list;
    }

    float mean_squared_error(vector<float> output, vector<float> label){
        if(output.size() != label.size())
        {
            cout << "Size of the output layer and the label is not the same." << endl;
            throw stderr;
        }

        float loss = 0;
        int n = output.size();
        for(int i = 0; i < n; i++){loss += pow(label[i] - output[i], 2);}
        loss = loss / n;

        return loss;
    }

    void train(vector<digit_img> img_list, vector<digit_label> label_list, int epochs, float learning_rate)
    {
        vector<neuron> output;
        float tot_loss = 0;
        for(int i0 = 0; i0 < epochs; i0++)
        {
            cout << " - Epoch: " << i0 << " -\nbatch: " << flush;
            for(int i1 = 0; i1 < (int) (img_list.size()/BATCH_SIZE); i1++){
                cout << i1 << " " << flush;
                for(int i2 = 0; i2 < BATCH_SIZE; i2++){
                    int rand_indx = rand() % img_list.size();
                    output = feed_forward(img_list[rand_indx]);
                    tot_loss = mean_squared_error(output, label_list[rand_indx]);
                    //grad_descent(output, label_list[rand_indx]);
                }
            }
            cout << "epoch complete, loss: " << tot_loss<< endl;
        }
    }

    void print_output(vector<neuron> output_layer){

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

float sigmoid(float n)
{
    return 1 / (1 + (exp(-n)));
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
            digit_label label = {one_hot_encode((float) tmp)};
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

    network agent;
    //Network initialized by global variables
    agent.init_network();
    agent.train(training_imgs, training_labels, 5, LEARNING_RATE);
}

int main()
{
    srand(time(NULL));

    run();
    return 0;
}