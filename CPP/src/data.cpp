//
// Created by André Lynum on 31/03/18.
//

#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "data.h"

class CSVRow {
public:
    std::string const &operator[](std::size_t index) const {
        return m_data[index];
    }

    std::size_t size() const {
        return m_data.size();
    }

    void readNextRow(std::istream &str) {
        std::string line;
        std::getline(str, line);

        std::stringstream lineStream(line);
        std::string cell;

        m_data.clear();
        while (std::getline(lineStream, cell, ',')) {
            m_data.push_back(cell);
        }
        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty()) {
            // If there was a trailing comma then add an empty element.
            m_data.emplace_back("");
        }
    }

private:
    std::vector<std::string> m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}

class CSVIterator {
public:
    typedef std::input_iterator_tag iterator_category;
    typedef CSVRow value_type;
    typedef std::size_t difference_type;
    typedef CSVRow *pointer;
    typedef CSVRow &reference;

    explicit CSVIterator(std::istream &str) : m_str(str.good() ? &str : nullptr) { ++(*this); }

    CSVIterator() : m_str(nullptr) {}

    // Pre Increment
    CSVIterator &operator++() {
        if (m_str) { if (!((*m_str) >> m_row)) { m_str = nullptr; }}
        return *this;
    }

    // Post increment
    const CSVIterator operator++(int) {
        CSVIterator tmp(*this);
        ++(*this);
        return tmp;
    }

    CSVRow const &operator*() const { return m_row; }

    CSVRow const *operator->() const { return &m_row; }

    bool operator==(CSVIterator const &rhs) {
        return ((this == &rhs) || ((this->m_str == nullptr) && (rhs.m_str == nullptr)));
    }

    bool operator!=(CSVIterator const &rhs) { return !((*this) == rhs); }

private:
    std::istream *m_str;
    CSVRow m_row;
};

Dataset<Eigen::MatrixXd, std::vector<std::string>> irisData(const std::string &path) {
    std::vector<double> x_values;
    std::vector<std::string> y_values;

    std::ifstream file(path);

    for(CSVIterator loop(file); loop != CSVIterator(); ++loop)
    {
        const CSVRow& row = *loop;

        for (int i = 0; i < 4; i++) {
            x_values.push_back(strtod(row[i].c_str(), nullptr));
        }

        y_values.push_back(row[4]);
    }

    Eigen::MatrixXd x = Eigen::Map<const Eigen::MatrixXd>(x_values.data(), 4, x_values.size() / 4).transpose();

    return Dataset<Eigen::MatrixXd, std::vector<std::string>>(x, y_values);
}

void Binarizer::fit(const std::vector<std::string> &data) {
    classes = std::vector<std::string>(data);

    std::sort(classes.begin(), classes.end());
    classes.erase(std::unique(classes.begin(), classes.end()), classes.end());
}

Eigen::MatrixXd Binarizer::transform(const std::vector<std::string> &data) {
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(data.size(), classes.size());

    for (auto it = data.begin(); it != data.end(); it++) {
        std::string e = *it;

        auto idx = std::distance(classes.begin(), std::find(classes.begin(), classes.end(), e));
        auto row = it - data.begin();

        result(row, idx) = 1.0;
    }

    return result;
}
