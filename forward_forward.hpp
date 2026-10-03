//
// Created by cartercpp on 10/3/26.
//

#pragma once

#include <initializer_list>
#include <vector>
#include <memory_resource>
#include "math_vector.hpp"
#include "matrix.hpp"

class forward_forward
{
public:

    // CONSTRUCTORS

    explicit forward_forward(
        const std::initializer_list<matrix<double>> weightMatrices,
        const std::initializer_list<math_vector<double>> biasVectors,
        const std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate,
        double goodnessThreshold,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(weightMatrices.size() == biasVectors.size())
        pre(weightMatrices.size() + 1 == neuronsPerLayer.size())
        pre(neuronsPerLayer.size() >= 3);

    explicit forward_forward(
        const std::initializer_list<std::size_t> neuronsPerLayer,
        double learningRate,
        double goodnessThreshold,
        std::pmr::memory_resource* resource = std::pmr::get_default_resource()
    )
        pre(neuronsPerLayer.size() >= 3)
        pre(*(neuronsPerLayer.begin() + neuronsPerLayer.size() - 1) == 1);

    // METHODS

    double predict(const math_vector<double>& input) const
        pre(input.size() == m_neuronsPerLayer[0]);

    void fit(const math_vector<double>& input, bool positiveExample)
        pre(input.size() == m_neuronsPerLayer[0]);

private:

    [[nodiscard]] static math_vector<double> Relu(math_vector<double>);
    [[nodiscard]] static math_vector<double> ReluDerivative(math_vector<double>);
    [[nodiscard]] static double Sigmoid(double);
    [[nodiscard]] static double SigmoidDerivative(double);

    std::pmr::vector<math_vector<double>> forward(const math_vector<double>& input) const
        pre(input.size() == m_neuronsPerLayer[0]);

    std::pmr::vector<matrix<double>> m_weightMatrices;
    std::pmr::vector<math_vector<double>> m_biasVectors;
    std::pmr::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate,
           m_goodnessThreshold;
    mutable std::pmr::memory_resource* m_resource;
};