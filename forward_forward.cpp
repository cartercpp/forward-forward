//
// Created by cartercpp on 10/3/26.
//

#include "forward_forward.hpp"
#include <initializer_list>
#include <vector>
#include <utility>
#include <memory_resource>
#include <random>
#include <cmath>
#include "math_vector.hpp"
#include "matrix.hpp"

forward_forward::forward_forward(
    const std::initializer_list<matrix<double>> weightMatrices,
    const std::initializer_list<math_vector<double>> biasVectors,
    const std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate,
    double goodnessThreshold,
    std::pmr::memory_resource* resource
)
    : m_weightMatrices{resource},
    m_biasVectors{resource},
    m_neuronsPerLayer{resource},
    m_learningRate{learningRate},
    m_goodnessThreshold{goodnessThreshold},
    m_resource{resource}
{
    m_weightMatrices.reserve(weightMatrices.size());
    for (const matrix<double>& weightMatrix : weightMatrices)
    {
        matrix<double> mat(weightMatrix.rows(), weightMatrix.columns(), 0, m_resource);

        for (std::size_t row = 0; row < weightMatrix.rows(); ++row)
            for (std::size_t column = 0; column < weightMatrix.columns(); ++column)
                mat[row][column] = weightMatrix[row][column];

        m_weightMatrices.emplace_back(std::move(mat));
    }

    m_biasVectors.reserve(biasVectors.size());
    for (const math_vector<double>& biasVector : biasVectors)
    {
        math_vector<double> vec(biasVector.size(), 0, m_resource);

        for (std::size_t i = 0; i < biasVector.size(); ++i)
            vec[i] = biasVector[i];

        m_biasVectors.emplace_back(std::move(vec));
    }

    m_neuronsPerLayer.reserve(neuronsPerLayer.size());
    for (const std::size_t neurons : neuronsPerLayer)
        m_neuronsPerLayer.push_back(neurons);
}

forward_forward::forward_forward(
    const std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate,
    double goodnessThreshold,
    std::pmr::memory_resource* resource
)
    : m_weightMatrices{resource},
    m_biasVectors{resource},
    m_neuronsPerLayer{resource},
    m_learningRate{learningRate},
    m_goodnessThreshold{goodnessThreshold},
    m_resource{resource}
{
    m_neuronsPerLayer.reserve(neuronsPerLayer.size());
    for (const std::size_t neurons : neuronsPerLayer)
        m_neuronsPerLayer.push_back(neurons);

    m_weightMatrices.reserve(m_neuronsPerLayer.size() - 1);
    m_biasVectors.reserve(m_neuronsPerLayer.size() - 1);

    std::mt19937 rng{std::random_device{}()};
    for (std::size_t layer = 1; layer < m_neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = m_neuronsPerLayer[layer],
                          prevLayerSize = m_neuronsPerLayer[layer - 1];

        matrix<double> weightMatrix(layerSize, prevLayerSize, 0, m_resource);

        std::normal_distribution<double> dist(0, 1 / static_cast<double>(2 * prevLayerSize));
        for (std::size_t row = 0; row < layerSize; ++row)
            for (std::size_t column = 0; column < prevLayerSize; ++column)
                weightMatrix[row][column] = dist(rng);

        m_weightMatrices.emplace_back(std::move(weightMatrix));
        m_biasVectors.emplace_back(layerSize, 0, m_resource);
    }
}

std::pmr::vector<math_vector<double>> forward_forward::forward(const math_vector<double>& input) const
{
    std::pmr::vector<math_vector<double>> activations{m_resource};
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t i = 0; i + 1 < m_weightMatrices.size(); ++i)
        activations.emplace_back(Relu(m_weightMatrices[i] * activations[i] + m_biasVectors[i]));

    return activations;
}

double forward_forward::predict(const math_vector<double>& input) const
{
    return Sigmoid((m_weightMatrices.back() * forward(input).back())[0] + m_biasVectors.back()[0]);
}

void forward_forward::fit(const math_vector<double>& input, bool positiveExample)
{
    const std::pmr::vector<math_vector<double>> activations{forward(input)};
    const double prediction = Sigmoid((m_weightMatrices.back() * activations.back())[0] + m_biasVectors.back()[0]);
    const math_vector<double> predictionGradient(1, prediction - positiveExample);

    for (std::size_t i = 0; i + 1 < m_weightMatrices.size(); ++i)
    {
        const double goodness = (activations[i + 1] * activations[i + 1]) / activations[i + 1].size();
        const math_vector<double> biasGradient{(2.0 / activations[i + 1].size()) * activations[i + 1]};
        const matrix<double> weightGradient{outer_product(biasGradient, activations[i])};

        if (positiveExample && (goodness < m_goodnessThreshold))
        {
            m_weightMatrices[i] += m_learningRate * weightGradient;
            m_biasVectors[i] += m_learningRate * biasGradient;
        }
        else if (!positiveExample && (goodness > m_goodnessThreshold))
        {
            m_weightMatrices[i] -= m_learningRate * weightGradient;
            m_biasVectors[i] -= m_learningRate * biasGradient;
        }
    }

    m_weightMatrices.back() -= m_learningRate * outer_product(predictionGradient, activations.back());
    m_biasVectors.back() -= m_learningRate * predictionGradient;
}

math_vector<double> forward_forward::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) * vec[i];

    return vec;
}

math_vector<double> forward_forward::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

double forward_forward::Sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

double forward_forward::SigmoidDerivative(double x)
{
    return x * (1 - x);
}