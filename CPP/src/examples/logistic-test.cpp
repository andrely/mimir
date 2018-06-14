#include <iostream>
#include <Eigen/Dense>
#include <random>

#include "../data.h"


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double pred(double x) {
    if (x < 0.5) {
        return 0.0;
    }
    return 1.0;
}

double accuracy(const Eigen::VectorXd &pred, const Eigen::VectorXd &truth) {
    return (pred.array() == truth.array()).count() / static_cast<float>(truth.size());
}

int main() {
    Dataset<Eigen::MatrixXd, std::vector<std::string>> d = irisData("/Users/stinky/Documents/Work/mimir/data/iris.data.csv.txt");
    
    Binarizer b = Binarizer();
    b.fit(d.y);

    const Eigen::MatrixXd &x = d.x;
    const Eigen::MatrixXd &y = b.transform(d.y).col(0);

    std::random_device dev;
    std::default_random_engine gen(dev());
    // gen.seed(1);
    std::normal_distribution<> dist(0.0, 0.001);

    Eigen::VectorXd w = Eigen::VectorXd::Zero(4);

    for (int i = 0; i < 4; i++) {
        w(i) = dist(gen);
    }

    double w0 = dist(gen);

    std::cout << w0 << std::endl << w << std::endl;

    for (int i = 0; i < 100; i++) {
        const Eigen::VectorXd p = ((x*w).array() + w0).unaryExpr(&sigmoid);

        if (i % 10 == 0) {
            const auto cost = (-y.array() * Eigen::log(p.array()) - (1.0 - y.array()) * Eigen::log(1.0 - p.array())).sum();
            std::cout << "cost: " << cost << std::endl;
        }

        const auto g0 = (p - y).sum();
        const auto g = x.transpose() * (p - y);
        const auto diag = (p.array() * (1.0 - p.array())).matrix();
        const auto h0 = diag.sum();
        const auto h = x.transpose() * diag.asDiagonal() * x;

        w0 = w0 - 0.1 * g0 / h0;
        w = w - (0.1 * (h.inverse() * g).array()).matrix();
    }

    std::cout << "w0:" << std::endl << w0 << std::endl;
    std::cout << "w:" << std::endl << w << std::endl;

    auto p = ((x*w).array() + w0).unaryExpr(&sigmoid).unaryExpr(&pred).matrix();
    std::cout << "accuracy: " << accuracy(p, y) << std::endl;
}
