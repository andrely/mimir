//
// Created by André Lynum on 31/03/18.
//

#ifndef CPP_DATA_H
#define CPP_DATA_H

#include <string>
#include <vector>


template <class X_T, class Y_T>
class Dataset {
public:
    X_T x;
    Y_T y;

    Dataset(X_T x, Y_T y) : x(x), y(y) {}
};

class Binarizer {
private:
    std::vector<std::string> classes;

public:
    void fit(const std::vector<std::string> &data);
    Eigen::MatrixXd transform(const std::vector<std::string> &data);
    Eigen::MatrixXd fitTransform(std::vector<std::string> data) {
        fit(data);
        return transform(data);
    }

};

Dataset<Eigen::MatrixXd, std::vector<std::string>> irisData(const std::string &path);

#endif //CPP_DATA_H
