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

#include <iostream>
#include <memory>
#include <vector>

#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class ReverseSequenceExec;
        }
    }
}

class ngraph::runtime::interpreter::ReverseSequenceExec : public ExecNode
{
public:
    static std::shared_ptr<ExecNode> create(const std::shared_ptr<ngraph::Node>& node)
    {
        std::cout << "create ReverseSequence" << std::endl;
        return std::static_pointer_cast<ExecNode>(std::make_shared<ReverseSequenceExec>(node));
    }

    ReverseSequenceExec(const std::shared_ptr<ngraph::Node> node)
        : ExecNode{node}
        , m_node{std::dynamic_pointer_cast<const ngraph::op::ReverseSequence>(node)}
    {
        (void)m_node; // Silence compiler warning

        std::cout << "ReverseSequence ctor" << std::endl;
    }

    virtual ~ReverseSequenceExec() {}
    template <typename T>
    void execute(const std::vector<std::shared_ptr<HostTensorView>>& out,
                 const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        std::cout << "execute ReverseSequence" << std::endl;
    }

    OP_TYPEID get_typeid() const override { return OP_TYPEID::ReverseSequence_TYPEID; }
private:
    std::shared_ptr<const ngraph::op::ReverseSequence> m_node;
};
