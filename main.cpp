/* 
    TODO: 
        Train and valid data set split
        Cost function implementation
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

struct digit_img
{
    unsigned char pixel[784];
};

struct digit_label
{
    vector<int> num;
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
        for(int i0 = 1; i0 < l_list.size(); i0++)
        {
            int index_prevlayer = i0 -1;
            int size_prevlayer = l_list[index_prevlayer].n_list.size();

            //For each node in layer
            for(int i1 = 0; i1 < l_list[i0].n_list.size(); i1++)
            {
                //Add random weight for each node in previous layer
                for(int i2 = 0; i2 < size_prevlayer; i2++)
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

    vector<neuron> feed_forward(vector<digit_img> img_list, vector<digit_label> label_list, bool stochastic = true)
    {
        if (sizeof(img_list[0].pixel) != l_list[0].n_list.size())
        {
            cout << "Input size of images doens't match network input size." << endl;
            throw stderr;
        }
        if (img_list.size() != label_list.size())
        {
            cout << "Amount of labels doesn't match amount of images in set" << endl;
            throw stderr;
        }

        // If no stochastic: loop through entire set for one episode.
        if (stochastic)
        {
            //For each picture in randomly chosen batch
            for (int i0 = 0; i0 < BATCH_SIZE; i0++)
            {
                int img_index = rand() % img_list.size();

                //Feed img-pixel-data into input layer
                for (int i1 = 0; i1 < sizeof(img_list[img_index].pixel); i1++)
                {
                    int pixel_val = img_list[img_index].pixel[i1];
                    l_list[0].n_list[i0].value = pixel_val;
                }

                //Feed forward operation on the rest of the layers
                for (int i1 = 1; i1 < l_list.size(); i1++)
                {
                    for (int i2 = 0; i2 < l_list[i1].n_list.size(); i2++)
                    {
                        //Curr_node = dot_prod(prev_layer_all_nodes, weights_to_prev_nodes) + curr_node_bias.
                        float n_value = dot_product(l_list[i1 - 1].n_list, l_list[i1].n_list[i2].con_weights) + l_list[i1].n_list[i2].bias;
                        n_value = sigmoid(n_value);
                        l_list[i1].n_list[i2].value = n_value;
                    }
                }
            }
        }
        else
        {
            cout << "Loading images input into network... " << endl;
            int cr0 = 0;
            for (auto img : img_list)
            {
                //Load the pixel values into the network
                cout << "Feed forward img: (" << cr0 << "/" << img_list.size() << ")" << endl;
                for (int i0 = 0; i0 < sizeof(img.pixel); i0++)
                {
                    l_list[0].n_list[i0].value = (int)img.pixel[i0];
                }

                //For each other layer, perform feed-forward calculations
                for (int i0 = 1; i0 < l_list.size(); i0++)
                {
                    cout << "Layer: " << i0 << endl;
                    cout << "\tNode: ";
                    for (int i1 = 0; i1 < l_list[i0].n_list.size(); i1++)
                    {
                        cout << i1 << " | " << flush;
                        //Matrix (vector) multiplication of all preious nodes times weights to that connected node plus current node bias.
                        l_list[i0].n_list[i1].value = dot_product(l_list[i0 - 1].n_list, l_list[i0].n_list[i1].con_weights) + l_list[i0].n_list[i1].bias;
                    }
                    cout << endl;
                }
                cr0++;
            }
        }

        //Return output layer
        return l_list[l_list.size()-1].n_list;
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

vector<int> one_hot_encode(int n)
{
    vector<int> ret = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ret[n] = 1;
    return ret;
}

vector<digit_img> get_img_data(char *fname)
{
    vector<digit_img> ret;

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
            ret.push_back(tmp);
        }
        file.close();
    }
    return ret;
}

vector<digit_label> get_label_data(char *fname)
{
    vector<digit_label> ret;

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
            int tmp = -1;
            file.read((char *)&tmp, sizeof(char));
            digit_label label = {one_hot_encode(tmp)};
            ret.push_back(label);
        }
        file.close();
    }

    return ret;
}

void run()
{
    cout << "Loading images and labels..." << endl;
    auto test_imgs = get_img_data((char *)"mnist/test-images/t10k-images.idx3-ubyte");
    auto test_label = get_label_data((char *)"mnist/test-labels/t10k-labels.idx1-ubyte");
    auto train_imgs = get_img_data((char *)"mnist/training-images/train-images.idx3-ubyte");
    auto train_label = get_label_data((char *)"mnist/training-labels/train-labels.idx1-ubyte");

    cout << "Beginning training..." << endl;

    network agent;
    agent.init_network();

    vector<neuron> output = agent.feed_forward(train_imgs, train_label);

    cout << "Output layer:" << endl;
    for(int i = 0; i < output.size(); i++)
    {
        neuron n = output[i];
        cout << i << ": " << n.value << endl; 
    }

}

int main()
{
    srand(time(NULL));

    run();
    return 0;
}