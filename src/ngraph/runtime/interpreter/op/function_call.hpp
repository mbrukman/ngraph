/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the \"License\");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an \"AS IS\" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

#include "ngraph/op/function_call.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/backend.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class FunctionCallExec;
        }
    }
}

using call_t =
    std::function<bool(std::shared_ptr<ngraph::Function> function,
                       const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outputs,
                       const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs)>;

class ngraph::runtime::interpreter::FunctionCallExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        return std::static_pointer_cast<ExecNode>(std::make_shared<FunctionCallExec>(node));
    }

    FunctionCallExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::FunctionCall>(node)}
    {
        (void)m_node; // Silence compiler warning
    }

    virtual ~FunctionCallExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args,
                 ngraph::runtime::Backend* backend)
    {
        std::shared_ptr<Function> function = m_node->get_functions()[0];

        std::vector<std::shared_ptr<runtime::TensorView>> outputs;
        for (auto tv : out)
        {
            outputs.push_back(std::static_pointer_cast<runtime::TensorView>(tv));
        }

        std::vector<std::shared_ptr<runtime::TensorView>> inputs;
        for (auto tv : args)
        {
            inputs.push_back(std::static_pointer_cast<runtime::TensorView>(tv));
        }

        backend->call(function, outputs, inputs);
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::FunctionCall_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::FunctionCall> m_node;
};
