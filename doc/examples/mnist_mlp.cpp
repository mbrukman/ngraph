/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdio>
#include <functional>
#include <iostream>
#include <list>
#include <math.h>
#include <memory>
#include <random>
#include <set>
#include <string>

#include <ngraph/graph_util.hpp>
#include <ngraph/ngraph.hpp>

#include "mnist.hpp"

using namespace ngraph;

template <typename T>
T read_scalar(const std::shared_ptr<runtime::TensorView>& t, size_t index = 0)
{
    T result;
    t->read(&result, index * sizeof(T), sizeof(T));
    return result;
}

template <typename T>
void write_scalar(const std::shared_ptr<runtime::TensorView>& t, T value, size_t index = 0)
{
    t->write(&value, index * sizeof(T), sizeof(T));
}

class TensorDumper
{
protected:
    TensorDumper(const std::string& name, const std::shared_ptr<runtime::TensorView>& tensor)
        : m_name(name)
        , m_tensor(tensor)
    {
    }

public:
    const std::string& get_name() const { return m_name; }
    std::shared_ptr<runtime::TensorView> get_tensor() const { return m_tensor; }
protected:
    std::string m_name;
    std::shared_ptr<runtime::TensorView> m_tensor;
};

class MinMax : public TensorDumper
{
public:
    MinMax(const std::string& name, const std::shared_ptr<runtime::TensorView>& tensor)
        : TensorDumper(name, tensor)
    {
        size_t n = m_tensor->get_element_count();
        for (size_t i = 0; i < n; ++i)
        {
            float s = read_scalar<float>(m_tensor, i);
            m_max = std::max(m_max, s);
            m_min = std::min(m_min, s);
        }
    }

    float get_min() const { return m_min; }
    float get_max() const { return m_max; }
protected:
    float m_min{std::numeric_limits<float>::max()};
    float m_max{std::numeric_limits<float>::min()};
};

std::ostream& operator<<(std::ostream& s, const MinMax& mm)
{
    return s << "MinMax[" << mm.get_name() << ":" << mm.get_min() << ", " << mm.get_max() << "]";
}

std::ostream& operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        s << shape.at(i);
        if (i + 1 < shape.size())
        {
            s << ", ";
        }
    }
    s << "}";
    return s;
}

class DumpTensor : public TensorDumper
{
public:
    DumpTensor(const std::string& name, const std::shared_ptr<runtime::TensorView>& tensor)
        : TensorDumper(name, tensor)
    {
    }
};

std::ostream& operator<<(std::ostream& s, const DumpTensor& dumper)
{
    std::shared_ptr<runtime::TensorView> t{dumper.get_tensor()};
    const Shape& shape = t->get_shape();
    s << "Tensor<" << dumper.get_name() << ": ";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        s << shape.at(i);
        if (i + 1 < shape.size())
        {
            s << ", ";
        }
    }
    size_t pos = 0;
    s << ">{";
    size_t rank = shape.size();
    if (rank == 0)
    {
        s << read_scalar<float>(t, pos++);
    }
    else if (rank <= 2)
    {
        s << "[";
        for (size_t i = 0; i < shape.at(0); ++i)
        {
            if (rank == 1)
            {
                s << read_scalar<float>(t, pos++);
            }
            else if (rank == 2)
            {
                s << "[";
                for (size_t j = 0; j < shape.at(1); ++j)
                {
                    s << read_scalar<float>(t, pos++);

                    if (j + 1 < shape.at(1))
                    {
                        s << ", ";
                    }
                }
                s << "]";
            }
            if (i + 1 < shape.at(0))
            {
                s << ", ";
            }
        }
        s << "]";
    }
    s << "}";
    return s;
}

std::shared_ptr<runtime::TensorView> make_output_tensor(std::shared_ptr<runtime::Backend> backend,
                                                        std::shared_ptr<Node> node,
                                                        size_t output)
{
    return backend->make_primary_tensor_view(node->get_output_element_type(output),
                                             node->get_output_shape(output));
}

void randomize(std::function<float()> rand, std::shared_ptr<runtime::TensorView> t)
{
    size_t element_count = t->get_element_count();
    std::vector<float> temp;
    for (size_t i = 0; i < element_count; ++i)
    {
        temp.push_back(rand());
    }
    t->write(&temp[0], 0, element_count * sizeof(float));
}

int main(int argc, const char* argv[])
{
    size_t epochs = 20;
    size_t batch_size = 64;
    size_t output_size = 10;

    size_t l0_size = 500;
    size_t l1_size = output_size;
    float log_min = static_cast<float>(exp(-50.0));
    MNistDataLoader test_loader{batch_size, MNistImageLoader::TEST, MNistLabelLoader::TEST};
    test_loader.open();
    size_t input_size = test_loader.get_columns() * test_loader.get_rows();

    // The data input
    auto X = std::make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    auto Y = std::make_shared<op::Parameter>(element::f32, Shape{batch_size});
    auto learning_rate = std::make_shared<op::Parameter>(element::f32, Shape{});
    auto N = std::make_shared<op::Parameter>(element::f32, Shape{});

    // Layer 0
    auto W0 = std::make_shared<op::Parameter>(element::f32, Shape{input_size, l0_size});
    auto b0 = std::make_shared<op::Parameter>(element::f32, Shape{l0_size});
    auto l0_dot = std::make_shared<op::Dot>(X, W0, 1);
    auto b0_broadcast = std::make_shared<op::Broadcast>(b0, Shape{batch_size, l0_size}, AxisSet{0});
    auto l0_sum = std::make_shared<op::Add>(l0_dot, b0_broadcast);
    auto l0 = std::make_shared<op::Relu>(l0_sum);

    // Layer 1
    auto W1 = std::make_shared<op::Parameter>(element::f32, Shape{l0_size, l1_size});
    auto b1 = std::make_shared<op::Parameter>(element::f32, Shape{l1_size});
    auto l1_dot = std::make_shared<op::Dot>(l0, W1, 1);
    auto b1_broadcast = std::make_shared<op::Broadcast>(b1, Shape{batch_size, l1_size}, AxisSet{0});
    auto l1_sum = std::make_shared<op::Add>(l1_dot, b1_broadcast);
    auto l1 = std::make_shared<op::Relu>(l1_sum);

    // Softmax
    auto sm = std::make_shared<op::Softmax>(l1, AxisSet{1});

    // Cost computation
    auto labels = std::make_shared<op::OneHot>(Y, Shape{batch_size, output_size}, 1);
    auto sm_clip_value =
        std::make_shared<op::Constant>(element::f32, Shape{}, std::vector<float>{log_min});
    auto sm_clip_broadcast = std::make_shared<op::Broadcast>(
        sm_clip_value, Shape{batch_size, output_size}, AxisSet{0, 1});
    auto sm_clip = std::make_shared<op::Maximum>(sm, sm_clip_broadcast);
    auto sm_log = std::make_shared<op::Log>(sm_clip);
    auto prod = std::make_shared<op::Multiply>(sm_log, labels);
    auto loss = std::make_shared<op::Divide>(std::make_shared<op::Sum>(prod, AxisSet{0, 1}), N);

    // Backprop
    // Each of W0, b0, W1, and b1
    auto delta =
        std::make_shared<op::Multiply>(std::make_shared<op::Negative>(learning_rate), loss);

    auto W0_delta = loss->backprop_node(W0, delta);
    auto b0_delta = loss->backprop_node(b0, delta);
    auto W1_delta = loss->backprop_node(W1, delta);
    auto b1_delta = loss->backprop_node(b1, delta);

    // Updates
    auto W0_next = std::make_shared<op::Add>(W0, W0_delta);
    auto b0_next = std::make_shared<op::Add>(b0, b0_delta);
    auto W1_next = std::make_shared<op::Add>(W1, W1_delta);
    auto b1_next = std::make_shared<op::Add>(b1, b1_delta);

    // Plain inference
    // X, W0, b0, W1, b1 -> sm
    auto inference_function =
        std::make_shared<Function>(NodeVector{sm}, op::ParameterVector{X, W0, b0, W1, b1});

    // Inference test function
    // X, Y, W0, b0, W1, b1 -> sm, loss
    auto inference_test_function = std::make_shared<Function>(
        NodeVector{sm, loss}, op::ParameterVector{X, Y, N, W0, b0, W1, b1});

    // Get the backend
    auto manager = runtime::Manager::get("CPU");
    auto backend = manager->allocate_backend();

    // Allocate and randomly initialize variables
    auto t_W0 = make_output_tensor(backend, W0, 0);
    auto t_b0 = make_output_tensor(backend, b0, 0);
    auto t_W1 = make_output_tensor(backend, W1, 0);
    auto t_b1 = make_output_tensor(backend, b1, 0);

    std::function<float()> rand(std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                                          std::default_random_engine(0)));
    randomize(rand, t_W0);
    randomize(rand, t_b0);
    randomize(rand, t_W1);
    randomize(rand, t_b1);

    // Allocate inputs
    auto t_X = make_output_tensor(backend, X, 0);
    auto t_Y = make_output_tensor(backend, Y, 0);

    auto t_learning_rate = make_output_tensor(backend, learning_rate, 0);
    auto t_N = make_output_tensor(backend, N, 0);
    write_scalar(t_N, static_cast<float>(batch_size), 0);

    // Allocate updated variables
    auto t_W0_next = make_output_tensor(backend, W0_next, 0);
    auto t_b0_next = make_output_tensor(backend, b0_next, 0);
    auto t_W1_next = make_output_tensor(backend, W1_next, 0);
    auto t_b1_next = make_output_tensor(backend, b1_next, 0);

    auto t_loss = make_output_tensor(backend, loss, 0);
    write_scalar(t_learning_rate, .01f);

    // Train
    // X, Y, learning_rate, W0, b0, W1, b1 -> loss, W0_next, b0_next, W1_next, b1_next
    auto train_function = std::make_shared<Function>(
        NodeVector{loss, W0_next, b0_next, W1_next, b1_next},
        op::ParameterVector{X, Y, N, learning_rate, W0, b0, W1, b1});
    auto train_ext = manager->compile(train_function);
    auto train_cf = backend->make_call_frame(train_ext);

    while (test_loader.get_epoch() < epochs)
    {
        test_loader.load();
        t_X->write(
            test_loader.get_image_floats(), 0, test_loader.get_image_batch_size() * sizeof(float));
        t_Y->write(
            test_loader.get_label_floats(), 0, test_loader.get_label_batch_size() * sizeof(float));
        train_cf->call({t_X, t_Y, t_N, t_learning_rate, t_W0, t_b0, t_W1, t_b1},
                       {t_loss, t_W0_next, t_b0_next, t_W1_next, t_b1_next});

        float this_loss = read_scalar<float>(t_loss);
        t_W0.swap(t_W0_next);
        t_b0.swap(t_b0_next);
        t_W1.swap(t_W1_next);
        t_b1.swap(t_b1_next);

        std::cout << "Pos: " << test_loader.get_pos() << " " << this_loss << std::endl;
        //break;
    }
    return 0;
}
