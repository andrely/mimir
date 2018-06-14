#include <iostream>
#include <Eigen/Dense>

#include "data.h"

int main() {
    Dataset<Eigen::MatrixXd, std::vector<std::string>> d = irisData("/Users/stinky/Documents/Work/mimir/data/iris.data.csv.txt");

    std::cout << d.x << std::endl;
    return 0;
}
