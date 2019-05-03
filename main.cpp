#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>

using namespace std;

struct digit_img{
    char pixel[784];
};

struct digit_label{
    int num;
};

int reverse_int(int i){
    unsigned char c0, c1, c2, c3;

    c0 = i & 255;
    c1 = (i >> 8) & 255;
    c2 = (i >> 16) & 255;
    c3 = (i >> 24) & 255;

    return ((int) c0 << 24) + ((int) c1 << 16) + ((int) c2 << 8) + c3;
}

vector<digit_img> get_img_data(char* fname){
    vector<digit_img> ret;

    ifstream file (fname);
    while(file.is_open()){
        int magic_num = 0;
        int am_img = 0;
        int am_rows = 0;
        int am_cols = 0;

        file.read((char*)&magic_num, sizeof(magic_num));
        magic_num = reverse_int(magic_num);

        file.read((char*)&am_img, sizeof(am_img));
        am_img = reverse_int(am_img);

        file.read((char*)&am_rows, sizeof(am_rows));
        am_rows = reverse_int(am_rows);

        file.read((char*)&am_cols, sizeof(am_cols));
        am_cols = reverse_int(am_cols);
        
        for(int i0 = 0; i0 < am_img; i0++){
            digit_img tmp;
            for(int i1 = 0; i1 < am_cols; i1++){
                for(int i2 = 0; i2 < am_rows; i2++){
                    file.read((char*)&tmp.pixel[i2 + i1*28], sizeof(char));
                }
            }
            ret.push_back(tmp);

        }
        file.close();
    }
    return ret;
}

vector<digit_label> get_label_data(char* fname){
    vector<digit_label> ret;

    ifstream file(fname);

    while(file.is_open()){
        int magic_num = 0;
        int am_img = 0;

        file.read((char*)&magic_num, sizeof(magic_num));
        magic_num = reverse_int(magic_num);

        file.read((char*)&am_img, sizeof(am_img));
        am_img = reverse_int(am_img);

        for(int i = 0; i < am_img; i++){
            digit_label label;
            file.read((char*)&label.num, sizeof(char));
            ret.push_back(label);
        }
        file.close();
    }

    return ret;
}

int main(){
    cout << "Loading images and labels..." << endl;
    auto test_imgs     = get_img_data((char*) "mnist/test-images/t10k-images.idx3-ubyte");
    auto test_labels   = get_label_data((char*) "mnist/test-labels/t10k-labels.idx1-ubyte");
    auto train_imgs    = get_img_data((char*) "mnist/training-images/train-images.idx3-ubyte");
    auto train_label     = get_label_data((char*) "mnist/training-labels/train-labels.idx1-ubyte");

    cout << "Beginning training..." << endl;
    return 0;
}