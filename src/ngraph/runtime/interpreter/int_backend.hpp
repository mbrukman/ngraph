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

#pragma once

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/host_tensor_view.hpp"
#include "ngraph/runtime/interpreter/exec_node.hpp"
#include "ngraph/runtime/interpreter/op/abs.hpp"
#include "ngraph/runtime/interpreter/op/acos.hpp"
#include "ngraph/runtime/interpreter/op/add.hpp"
#include "ngraph/runtime/interpreter/op/allreduce.hpp"
#include "ngraph/runtime/interpreter/op/and.hpp"
#include "ngraph/runtime/interpreter/op/asin.hpp"
#include "ngraph/runtime/interpreter/op/atan.hpp"
#include "ngraph/runtime/interpreter/op/avg_pool.hpp"
#include "ngraph/runtime/interpreter/op/batch_norm.hpp"
#include "ngraph/runtime/interpreter/op/broadcast.hpp"
#include "ngraph/runtime/interpreter/op/ceiling.hpp"
#include "ngraph/runtime/interpreter/op/concat.hpp"
#include "ngraph/runtime/interpreter/op/constant.hpp"
#include "ngraph/runtime/interpreter/op/convert.hpp"
#include "ngraph/runtime/interpreter/op/convolution.hpp"
#include "ngraph/runtime/interpreter/op/cos.hpp"
#include "ngraph/runtime/interpreter/op/cosh.hpp"
#include "ngraph/runtime/interpreter/op/divide.hpp"
#include "ngraph/runtime/interpreter/op/dot.hpp"
#include "ngraph/runtime/interpreter/op/equal.hpp"
#include "ngraph/runtime/interpreter/op/exp.hpp"
#include "ngraph/runtime/interpreter/op/floor.hpp"
#include "ngraph/runtime/interpreter/op/function_call.hpp"
#include "ngraph/runtime/interpreter/op/get_output_element.hpp"
#include "ngraph/runtime/interpreter/op/greater.hpp"
#include "ngraph/runtime/interpreter/op/greater_eq.hpp"
#include "ngraph/runtime/interpreter/op/less.hpp"
#include "ngraph/runtime/interpreter/op/less_eq.hpp"
#include "ngraph/runtime/interpreter/op/log.hpp"
#include "ngraph/runtime/interpreter/op/max.hpp"
#include "ngraph/runtime/interpreter/op/max_pool.hpp"
#include "ngraph/runtime/interpreter/op/maximum.hpp"
#include "ngraph/runtime/interpreter/op/min.hpp"
#include "ngraph/runtime/interpreter/op/minimum.hpp"
#include "ngraph/runtime/interpreter/op/multiply.hpp"
#include "ngraph/runtime/interpreter/op/negative.hpp"
#include "ngraph/runtime/interpreter/op/not.hpp"
#include "ngraph/runtime/interpreter/op/not_equal.hpp"
#include "ngraph/runtime/interpreter/op/one_hot.hpp"
#include "ngraph/runtime/interpreter/op/or.hpp"
#include "ngraph/runtime/interpreter/op/pad.hpp"
#include "ngraph/runtime/interpreter/op/parameter.hpp"
#include "ngraph/runtime/interpreter/op/power.hpp"
#include "ngraph/runtime/interpreter/op/product.hpp"
#include "ngraph/runtime/interpreter/op/reduce.hpp"
#include "ngraph/runtime/interpreter/op/reduce_window.hpp"
#include "ngraph/runtime/interpreter/op/relu.hpp"
#include "ngraph/runtime/interpreter/op/relu_backprop.hpp"
#include "ngraph/runtime/interpreter/op/replace_slice.hpp"
#include "ngraph/runtime/interpreter/op/reshape.hpp"
#include "ngraph/runtime/interpreter/op/result.hpp"
#include "ngraph/runtime/interpreter/op/reverse.hpp"
#include "ngraph/runtime/interpreter/op/reverse_sequence.hpp"
#include "ngraph/runtime/interpreter/op/select.hpp"
#include "ngraph/runtime/interpreter/op/select_and_scatter.hpp"
#include "ngraph/runtime/interpreter/op/sigmoid.hpp"
#include "ngraph/runtime/interpreter/op/sigmoid_backprop.hpp"
#include "ngraph/runtime/interpreter/op/sign.hpp"
#include "ngraph/runtime/interpreter/op/sin.hpp"
#include "ngraph/runtime/interpreter/op/sinh.hpp"
#include "ngraph/runtime/interpreter/op/slice.hpp"
#include "ngraph/runtime/interpreter/op/softmax.hpp"
#include "ngraph/runtime/interpreter/op/sqrt.hpp"
#include "ngraph/runtime/interpreter/op/stop_gradient.hpp"
#include "ngraph/runtime/interpreter/op/subtract.hpp"
#include "ngraph/runtime/interpreter/op/sum.hpp"
#include "ngraph/runtime/interpreter/op/tan.hpp"
#include "ngraph/runtime/interpreter/op/tanh.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/add.hpp"
#include "ngraph/runtime/reference/and.hpp"
#include "ngraph/runtime/reference/asin.hpp"
#include "ngraph/runtime/reference/atan.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/constant.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/copy.hpp"
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/runtime/reference/cosh.hpp"
#include "ngraph/runtime/reference/divide.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/equal.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/greater.hpp"
#include "ngraph/runtime/reference/greater_eq.hpp"
#include "ngraph/runtime/reference/less.hpp"
#include "ngraph/runtime/reference/less_eq.hpp"
#include "ngraph/runtime/reference/log.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/maximum.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/minimum.hpp"
#include "ngraph/runtime/reference/multiply.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/not_equal.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/or.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/power.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/reduce.hpp"
#include "ngraph/runtime/reference/reduce_window.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/select_and_scatter.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sin.hpp"
#include "ngraph/runtime/reference/sinh.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/subtract.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/tan.hpp"
#include "ngraph/runtime/reference/tanh.hpp"
#include "ngraph/runtime/tensor_view.hpp"

#ifdef NGRAPH_DISTRIBUTED
#include "ngraph/runtime/reference/allreduce.hpp"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
        }
    }
}
class ngraph::runtime::interpreter::INTBackend : public Backend
{
public:
    std::shared_ptr<TensorView>
        create_tensor(const element::Type& type, const Shape& shape, void* memory_pointer) override;

    std::shared_ptr<TensorView> create_tensor(const element::Type& type,
                                              const Shape& shape) override;

    bool compile(std::shared_ptr<Function> function) override;

    bool call(std::shared_ptr<Function> function,
              const std::vector<std::shared_ptr<TensorView>>& outputs,
              const std::vector<std::shared_ptr<TensorView>>& intputs) override;

    void set_nan_check(std::shared_ptr<Function> func, bool);

    void enable_performance_data(std::shared_ptr<Function> func, bool enable) override;
    std::vector<PerformanceCounter>
        get_performance_data(std::shared_ptr<Function> func) const override;

private:
    class FunctionInstance
    {
    public:
        bool m_is_compiled = false;
        bool m_nan_check_enabled = false;
        bool m_performance_counters_enabled = false;
        std::unordered_map<const Node*, stopwatch> m_timer_map;
        std::vector<std::shared_ptr<ExecNode>> m_ops;
    };
    std::map<std::shared_ptr<Function>, FunctionInstance> m_function_map;

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensorView>>&,
                                  const Node* op = nullptr);

    void generate_calls(const element::Type& type,
                        ExecNode& op,
                        const std::vector<std::shared_ptr<HostTensorView>>& outputs,
                        const std::vector<std::shared_ptr<HostTensorView>>& inputs);

    template <typename T>
    void op_engine(ExecNode& node,
                   const std::vector<std::shared_ptr<HostTensorView>>& out,
                   const std::vector<std::shared_ptr<HostTensorView>>& args)
    {
        switch (node.get_typeid())
        {
        case OP_TYPEID::Abs_TYPEID:
        {
            dynamic_cast<AbsExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Acos_TYPEID:
        {
            dynamic_cast<AcosExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Add_TYPEID:
        {
            dynamic_cast<AddExec*>(&node)->execute<T>(out, args);
            break;
        }
#ifdef NGRAPH_DISTRIBUTED
        case OP_TYPEID::AllReduce_TYPEID:
        {
            dynamic_cast<AllReduceExec*>(&node)->execute<T>(out, args);
            break;
        }
#endif
        case OP_TYPEID::And_TYPEID:
        {
            dynamic_cast<AndExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Asin_TYPEID:
        {
            dynamic_cast<AsinExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Atan_TYPEID:
        {
            dynamic_cast<AtanExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::AvgPool_TYPEID:
        {
            dynamic_cast<AvgPoolExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::BatchNorm_TYPEID:
        {
            dynamic_cast<BatchNormExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Broadcast_TYPEID:
        {
            dynamic_cast<BroadcastExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Ceiling_TYPEID:
        {
            dynamic_cast<CeilingExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Concat_TYPEID:
        {
            dynamic_cast<ConcatExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Constant_TYPEID:
        {
            dynamic_cast<ConstantExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Convert_TYPEID:
        {
            dynamic_cast<ConvertExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Convolution_TYPEID:
        {
            dynamic_cast<ConvolutionExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Cos_TYPEID:
        {
            dynamic_cast<CosExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Cosh_TYPEID:
        {
            dynamic_cast<CoshExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Divide_TYPEID:
        {
            dynamic_cast<DivideExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Dot_TYPEID:
        {
            dynamic_cast<DotExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Equal_TYPEID:
        {
            dynamic_cast<EqualExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Exp_TYPEID:
        {
            dynamic_cast<ExpExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Floor_TYPEID:
        {
            dynamic_cast<FloorExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::FunctionCall_TYPEID:
        {
            dynamic_cast<FunctionCallExec*>(&node)->execute<T>(out, args, this);
            break;
        }
        case OP_TYPEID::GetOutputElement_TYPEID:
        {
            dynamic_cast<GetOutputElementExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Greater_TYPEID:
        {
            dynamic_cast<GreaterExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::GreaterEq_TYPEID:
        {
            dynamic_cast<GreaterEqExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Less_TYPEID:
        {
            dynamic_cast<LessExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::LessEq_TYPEID:
        {
            dynamic_cast<LessEqExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Log_TYPEID:
        {
            dynamic_cast<LogExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Max_TYPEID:
        {
            dynamic_cast<MaxExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Maximum_TYPEID:
        {
            dynamic_cast<MaximumExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::MaxPool_TYPEID:
        {
            dynamic_cast<MaxPoolExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Min_TYPEID:
        {
            dynamic_cast<MinExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Minimum_TYPEID:
        {
            dynamic_cast<MinimumExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Multiply_TYPEID:
        {
            dynamic_cast<MultiplyExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Negative_TYPEID:
        {
            dynamic_cast<NegativeExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Not_TYPEID:
        {
            dynamic_cast<NotExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::NotEqual_TYPEID:
        {
            dynamic_cast<NotEqualExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::OneHot_TYPEID:
        {
            dynamic_cast<OneHotExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Or_TYPEID:
        {
            dynamic_cast<OrExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Pad_TYPEID:
        {
            dynamic_cast<PadExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Parameter_TYPEID:
        {
            dynamic_cast<ParameterExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Power_TYPEID:
        {
            dynamic_cast<PowerExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Product_TYPEID:
        {
            dynamic_cast<ProductExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Reduce_TYPEID:
        {
            dynamic_cast<ReduceExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::ReduceWindow_TYPEID:
        {
            dynamic_cast<ReduceWindowExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Relu_TYPEID:
        {
            dynamic_cast<ReluExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::ReplaceSlice_TYPEID:
        {
            dynamic_cast<ReplaceSliceExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Reshape_TYPEID:
        {
            dynamic_cast<ReshapeExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Result_TYPEID:
        {
            dynamic_cast<ResultExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Reverse_TYPEID:
        {
            dynamic_cast<ReverseExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::ReverseSequence_TYPEID:
        {
            dynamic_cast<ReverseSequenceExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Select_TYPEID:
        {
            dynamic_cast<SelectExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::SelectAndScatter_TYPEID:
        {
            dynamic_cast<SelectAndScatterExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sigmoid_TYPEID:
        {
            dynamic_cast<SigmoidExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sign_TYPEID:
        {
            dynamic_cast<SignExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sin_TYPEID:
        {
            dynamic_cast<SinExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sinh_TYPEID:
        {
            dynamic_cast<SinhExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Slice_TYPEID:
        {
            dynamic_cast<SliceExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Softmax_TYPEID:
        {
            dynamic_cast<SoftmaxExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sqrt_TYPEID:
        {
            dynamic_cast<SqrtExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::StopGradient_TYPEID:
        {
            dynamic_cast<StopGradientExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Subtract_TYPEID:
        {
            dynamic_cast<SubtractExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Sum_TYPEID:
        {
            dynamic_cast<SumExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Tan_TYPEID:
        {
            dynamic_cast<TanExec*>(&node)->execute<T>(out, args);
            break;
        }
        case OP_TYPEID::Tanh_TYPEID:
        {
            dynamic_cast<TanhExec*>(&node)->execute<T>(out, args);
            break;
        }
        }

        //         node.execute<T>(out, args);
        //         std::string node_op = node.description();
        //         if (node_op == "Abs")
        //         {
        //         }
        //         else if (node_op == "Acos")
        //         {
        //         }
        //         else if (node_op == "Add")
        //         {
        //         }
        // #ifdef NGRAPH_DISTRIBUTED
        //         else if (node_op == "AllReduce")
        //         {
        //         }
        // #endif
        //         else if (node_op == "And")
        //         {
        //         }
        //         else if (node_op == "Asin")
        //         {
        //         }
        //         else if (node_op == "Atan")
        //         {
        //         }
        //         else if (node_op == "AvgPool")
        //         {
        //         }
        //         else if (node_op == "GetOutputElement")
        //         {
        //         }
        //         else if (node_op == "BatchNorm")
        //         {
        //         }
        //         else if (node_op == "AvgPoolBackprop")
        //         {
        //             op::AvgPoolBackprop* apb = dynamic_cast<op::AvgPoolBackprop*>(&node);
        //             reference::avg_pool_backprop<T>(args[0]->get_data_ptr<T>(),
        //                                             out[0]->get_data_ptr<T>(),
        //                                             args[0]->get_shape(),
        //                                             out[0]->get_shape(),
        //                                             apb->get_window_shape(),
        //                                             apb->get_window_movement_strides(),
        //                                             apb->get_padding_below(),
        //                                             apb->get_padding_above(),
        //                                             apb->get_include_padding_in_avg_computation());
        //         }
        //         else if (node_op == "Broadcast")
        //         {
        //         }
        //         else if (node_op == "Ceiling")
        //         {
        //         }
        //         else if (node_op == "Concat")
        //         {
        //         }
        //         else if (node_op == "Constant")
        //         {
        //         }
        //         else if (node_op == "Convert")
        //         {
        //         }
        //         else if (node_op == "Convolution")
        //         {
        //         }
        //         else if (node_op == "ConvolutionBackpropFilters")
        //         {
        //             auto c = static_cast<const op::ConvolutionBackpropFilters*>(&node);
        //             reference::convolution<T>(args[0]->get_data_ptr<T>(),
        //                                       args[1]->get_data_ptr<T>(),
        //                                       out[0]->get_data_ptr<T>(),
        //                                       args[0]->get_shape(),
        //                                       args[1]->get_shape(),
        //                                       out[0]->get_shape(),
        //                                       c->get_window_movement_strides_backward(),
        //                                       c->get_window_dilation_strides_backward(),
        //                                       c->get_padding_below_backward(),
        //                                       c->get_padding_above_backward(),
        //                                       c->get_data_dilation_strides_backward(),
        //                                       1,
        //                                       0,
        //                                       0,
        //                                       1,
        //                                       1,
        //                                       0,
        //                                       false);
        //         }
        //         else if (node_op == "ConvolutionBackpropData")
        //         {
        //             // Note that args[1] and args[0] are switched here from the usual order.
        //             auto c = static_cast<const op::ConvolutionBackpropData*>(&node);
        //             reference::convolution<T>(args[1]->get_data_ptr<T>(),
        //                                       args[0]->get_data_ptr<T>(),
        //                                       out[0]->get_data_ptr<T>(),
        //                                       args[1]->get_shape(),
        //                                       args[0]->get_shape(),
        //                                       out[0]->get_shape(),
        //                                       c->get_window_movement_strides_backward(),
        //                                       c->get_window_dilation_strides_backward(),
        //                                       c->get_padding_below_backward(),
        //                                       c->get_padding_above_backward(),
        //                                       c->get_data_dilation_strides_backward(),
        //                                       0,
        //                                       1,
        //                                       0,
        //                                       1,
        //                                       0,
        //                                       1,
        //                                       true);
        //         }
        //         else if (node_op == "Cos")
        //         {
        //         }
        //         else if (node_op == "Cosh")
        //         {
        //         }
        //         else if (node_op == "Divide")
        //         {
        //         }
        //         else if (node_op == "Dot")
        //         {
        //         }
        //         else if (node_op == "Equal")
        //         {
        //         }
        //         else if (node_op == "Exp")
        //         {
        //         }
        //         else if (node_op == "Floor")
        //         {
        //         }
        //         else if (node_op == "FunctionCall")
        //         {
        //         }
        //         else if (node_op == "Greater")
        //         {
        //         }
        //         else if (node_op == "GreaterEq")
        //         {
        //         }
        //         else if (node_op == "Less")
        //         {
        //         }
        //         else if (node_op == "LessEq")
        //         {
        //         }
        //         else if (node_op == "Log")
        //         {
        //         }
        //         else if (node_op == "Max")
        //         {
        //         }
        //         else if (node_op == "Maximum")
        //         {
        //         }
        //         else if (node_op == "MaxPool")
        //         {
        //         }
        //         else if (node_op == "MaxPoolBackprop")
        //         {
        //         }
        //         else if (node_op == "Min")
        //         {
        //             const op::Min* min = static_cast<const op::Min*>(&node);
        //             reference::min<T>(args[0]->get_data_ptr<T>(),
        //                               out[0]->get_data_ptr<T>(),
        //                               args[0]->get_shape(),
        //                               out[0]->get_shape(),
        //                               min->get_reduction_axes());
        //         }
        //         else if (node_op == "Minimum")
        //         {
        //             reference::minimum<T>(args[0]->get_data_ptr<T>(),
        //                                   args[1]->get_data_ptr<T>(),
        //                                   out[0]->get_data_ptr<T>(),
        //                                   out[0]->get_element_count());
        //         }
        //         else if (node_op == "Multiply")
        //         {
        //             reference::multiply<T>(args[0]->get_data_ptr<T>(),
        //                                    args[1]->get_data_ptr<T>(),
        //                                    out[0]->get_data_ptr<T>(),
        //                                    out[0]->get_element_count());
        //         }
        //         else if (node_op == "Negative")
        //         {
        //             reference::negate<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Not")
        //         {
        //             reference::logical_not(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "NotEqual")
        //         {
        //             reference::not_equal<T>(args[0]->get_data_ptr<T>(),
        //                                     args[1]->get_data_ptr<T>(),
        //                                     out[0]->get_data_ptr<char>(),
        //                                     out[0]->get_element_count());
        //         }
        //         else if (node_op == "OneHot")
        //         {
        //             auto oh = static_cast<const op::OneHot*>(&node);
        //             reference::one_hot<T>(args[0]->get_data_ptr<T>(),
        //                                   out[0]->get_data_ptr<T>(),
        //                                   args[0]->get_shape(),
        //                                   out[0]->get_shape(),
        //                                   oh->get_one_hot_axis());
        //         }
        //         else if (node_op == "Or")
        //         {
        //             reference::logical_or(args[0]->get_data_ptr<T>(),
        //                                   args[1]->get_data_ptr<T>(),
        //                                   out[0]->get_data_ptr<T>(),
        //                                   out[0]->get_element_count());
        //         }
        //         else if (node_op == "Parameter")
        //         {
        //         }
        //         else if (node_op == "Pad")
        //         {
        //             op::Pad* pad = dynamic_cast<op::Pad*>(&node);

        //             reference::pad(args[0]->get_data_ptr<T>(),
        //                            args[1]->get_data_ptr<T>(),
        //                            out[0]->get_data_ptr<T>(),
        //                            node.get_inputs().at(0).get_shape(),
        //                            node.get_output_shape(0),
        //                            pad->get_padding_below(),
        //                            pad->get_padding_above(),
        //                            pad->get_padding_interior());
        //         }
        //         else if (node_op == "Power")
        //         {
        //             reference::power<T>(args[0]->get_data_ptr<T>(),
        //                                 args[1]->get_data_ptr<T>(),
        //                                 out[0]->get_data_ptr<T>(),
        //                                 out[0]->get_element_count());
        //         }
        //         else if (node_op == "Product")
        //         {
        //             const op::Product* product = static_cast<const op::Product*>(&node);
        //             reference::product<T>(args[0]->get_data_ptr<T>(),
        //                                   out[0]->get_data_ptr<T>(),
        //                                   args[0]->get_shape(),
        //                                   out[0]->get_shape(),
        //                                   product->get_reduction_axes());
        //         }
        //         else if (node_op == "Reduce")
        //         {
        //             op::Reduce* reduce = dynamic_cast<op::Reduce*>(&node);
        //             std::shared_ptr<Function> reduction_function = reduce->get_functions()[0];

        //             std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
        //                 auto tx = std::make_shared<HostTensorView>(
        //                     node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_temp_x");
        //                 auto ty = std::make_shared<HostTensorView>(
        //                     node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_temp_y");
        //                 auto tr = std::make_shared<HostTensorView>(
        //                     node.get_output_element_type(0), Shape{}, "reduce_temp_r");
        //                 *(tx->get_data_ptr<T>()) = x;
        //                 *(ty->get_data_ptr<T>()) = y;
        //                 call(reduction_function, {tr}, {tx, ty});
        //                 return *(tr->get_data_ptr<T>());
        //             };

        //             reference::reduce(args[0]->get_data_ptr<T>(),
        //                               args[1]->get_data_ptr<T>(),
        //                               out[0]->get_data_ptr<T>(),
        //                               node.get_inputs().at(0).get_shape(),
        //                               node.get_output_shape(0),
        //                               reduce->get_reduction_axes(),
        //                               f);
        //         }
        //         else if (node_op == "ReduceWindow")
        //         {
        //             op::ReduceWindow* reduce_window = dynamic_cast<op::ReduceWindow*>(&node);
        //             std::shared_ptr<Function> reduction_function = reduce_window->get_functions()[0];

        //             std::function<T(T, T)> f = [this, &node, reduction_function](T x, T y) -> T {
        //                 auto tx = std::make_shared<HostTensorView>(
        //                     node.get_inputs().at(0).get_element_type(), Shape{}, "reduce_window_temp_x");
        //                 auto ty = std::make_shared<HostTensorView>(
        //                     node.get_inputs().at(1).get_element_type(), Shape{}, "reduce_window_temp_y");
        //                 auto tr = std::make_shared<HostTensorView>(
        //                     node.get_output_element_type(0), Shape{}, "reduce_window_temp_r");
        //                 *(tx->get_data_ptr<T>()) = x;
        //                 *(ty->get_data_ptr<T>()) = y;
        //                 call(reduction_function, {tr}, {tx, ty});
        //                 return *(tr->get_data_ptr<T>());
        //             };

        //             reference::reduce_window(args[0]->get_data_ptr<T>(),
        //                                      args[1]->get_data_ptr<T>(),
        //                                      out[0]->get_data_ptr<T>(),
        //                                      node.get_inputs().at(0).get_shape(),
        //                                      node.get_output_shape(0),
        //                                      f,
        //                                      reduce_window->get_window_shape(),
        //                                      reduce_window->get_window_movement_strides());
        //         }
        //         else if (node_op == "Relu")
        //         {
        //             reference::relu<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "ReluBackprop")
        //         {
        //             reference::relu_backprop<T>(args[0]->get_data_ptr<T>(),
        //                                         args[1]->get_data_ptr<T>(),
        //                                         out[0]->get_data_ptr<T>(),
        //                                         out[0]->get_element_count());
        //         }
        //         else if (node_op == "ReplaceSlice")
        //         {
        //             const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
        //             reference::replace_slice<T>(args[0]->get_data_ptr<T>(),
        //                                         args[1]->get_data_ptr<T>(),
        //                                         out[0]->get_data_ptr<T>(),
        //                                         args[1]->get_shape(),
        //                                         slice->get_lower_bounds(),
        //                                         slice->get_upper_bounds(),
        //                                         slice->get_strides(),
        //                                         out[0]->get_shape());
        //         }
        //         else if (node_op == "Reshape")
        //         {
        //             op::Reshape* reshape = dynamic_cast<op::Reshape*>(&node);
        //             reference::reshape(args[0]->get_data_ptr<T>(),
        //                                out[0]->get_data_ptr<T>(),
        //                                args[0]->get_shape(),
        //                                reshape->get_input_order(),
        //                                out[0]->get_shape());
        //         }
        //         else if (node_op == "Result")
        //         {
        //             op::Result* res = dynamic_cast<op::Result*>(&node);
        //             reference::result(args[0]->get_data_ptr<T>(),
        //                               out[0]->get_data_ptr<T>(),
        //                               shape_size(res->get_shape()));
        //         }
        //         else if (node_op == "Reverse")
        //         {
        //             op::Reverse* reverse = dynamic_cast<op::Reverse*>(&node);
        //             reference::reverse(args[0]->get_data_ptr<T>(),
        //                                out[0]->get_data_ptr<T>(),
        //                                args[0]->get_shape(),
        //                                out[0]->get_shape(),
        //                                reverse->get_reversed_axes());
        //         }
        //         else if (node_op == "ReverseSequence")
        //         {
        //             op::ReverseSequence* reverse = dynamic_cast<op::ReverseSequence*>(&node);

        //             if (args[1]->get_element_type() == element::i32)
        //             {
        //                 reference::reverse_sequence<T, int>(args[0]->get_data_ptr<T>(),
        //                                                     out[0]->get_data_ptr<T>(),
        //                                                     args[0]->get_shape(),
        //                                                     reverse->get_batch_axis(),
        //                                                     reverse->get_sequence_axis(),
        //                                                     args[1]->get_data_ptr<int>());
        //             }
        //             else
        //             {
        //                 throw ngraph_error("only int32 indices are supported");
        //             }
        //         }
        //         else if (node_op == "Select")
        //         {
        //             reference::select<T>(args[0]->get_data_ptr<char>(),
        //                                  args[1]->get_data_ptr<T>(),
        //                                  args[2]->get_data_ptr<T>(),
        //                                  out[0]->get_data_ptr<T>(),
        //                                  out[0]->get_element_count());
        //         }
        //         else if (node_op == "SelectAndScatter")
        //         {
        //             ngraph::op::SelectAndScatter* select_and_scatter =
        //                 dynamic_cast<ngraph::op::SelectAndScatter*>(&node);

        //             std::shared_ptr<ngraph::Function> selection_function =
        //                 select_and_scatter->get_functions()[0];
        //             std::function<bool(T, T)> f_selection = [this, &node, selection_function](T x,
        //                                                                                       T y) -> bool {
        //                 auto tx = std::make_shared<runtime::HostTensorView>(
        //                     node.get_inputs().at(0).get_element_type(), Shape{}, "selection_temp_x");
        //                 auto ty = std::make_shared<runtime::HostTensorView>(
        //                     node.get_inputs().at(1).get_element_type(), Shape{}, "selection_temp_y");
        //                 auto tr = std::make_shared<runtime::HostTensorView>(
        //                     element::boolean, Shape{}, "selection_temp_r");
        //                 *(tx->get_data_ptr<T>()) = x;
        //                 *(ty->get_data_ptr<T>()) = y;
        //                 call(selection_function, {tr}, {tx, ty});
        //                 return *(tr->get_data_ptr<char>());
        //             };

        //             std::shared_ptr<ngraph::Function> scatter_function =
        //                 select_and_scatter->get_functions()[1];
        //             std::function<T(T, T)> f_scatter = [this, &node, scatter_function](T x, T y) -> T {
        //                 auto tx = std::make_shared<runtime::HostTensorView>(
        //                     node.get_inputs().at(0).get_element_type(), Shape{}, "scatter_temp_x");
        //                 auto ty = std::make_shared<runtime::HostTensorView>(
        //                     node.get_inputs().at(1).get_element_type(), Shape{}, "scatter_temp_y");
        //                 auto tr = std::make_shared<runtime::HostTensorView>(
        //                     node.get_output_element_type(0), Shape{}, "scatter_temp_r");
        //                 *(tx->get_data_ptr<T>()) = x;
        //                 *(ty->get_data_ptr<T>()) = y;
        //                 call(scatter_function, {tr}, {tx, ty});
        //                 return *(tr->get_data_ptr<T>());
        //             };

        //             reference::select_and_scatter<T>(args[0]->get_data_ptr<T>(),
        //                                              args[1]->get_data_ptr<T>(),
        //                                              args[2]->get_data_ptr<T>(),
        //                                              out[0]->get_data_ptr<T>(),
        //                                              args[0]->get_shape(),
        //                                              args[1]->get_shape(),
        //                                              out[0]->get_shape(),
        //                                              f_selection,
        //                                              f_scatter,
        //                                              select_and_scatter->get_window_shape(),
        //                                              select_and_scatter->get_window_movement_strides());
        //         }
        //         else if (node_op == "Sigmoid")
        //         {
        //             reference::sigmoid<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "SigmoidBackprop")
        //         {
        //             reference::sigmoid_backprop<T>(args[0]->get_data_ptr<T>(),
        //                                            args[1]->get_data_ptr<T>(),
        //                                            out[0]->get_data_ptr<T>(),
        //                                            out[0]->get_element_count());
        //         }
        //         else if (node_op == "Sign")
        //         {
        //             reference::sign<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Sin")
        //         {
        //             reference::sin<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Sinh")
        //         {
        //             reference::sinh<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Slice")
        //         {
        //             const op::Slice* slice = static_cast<const op::Slice*>(&node);
        //             reference::slice<T>(args[0]->get_data_ptr<T>(),
        //                                 out[0]->get_data_ptr<T>(),
        //                                 args[0]->get_shape(),
        //                                 slice->get_lower_bounds(),
        //                                 slice->get_upper_bounds(),
        //                                 slice->get_strides(),
        //                                 out[0]->get_shape());
        //         }
        //         else if (node_op == "Softmax")
        //         {
        //             const op::Softmax* softmax = static_cast<const op::Softmax*>(&node);
        //             reference::softmax<T>(args[0]->get_data_ptr<T>(),
        //                                   out[0]->get_data_ptr<T>(),
        //                                   out[0]->get_shape(),
        //                                   softmax->get_axes());
        //         }
        //         else if (node_op == "Sqrt")
        //         {
        //             reference::sqrt<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Subtract")
        //         {
        //             reference::subtract<T>(args[0]->get_data_ptr<T>(),
        //                                    args[1]->get_data_ptr<T>(),
        //                                    out[0]->get_data_ptr<T>(),
        //                                    out[0]->get_element_count());
        //         }
        //         else if (node_op == "Sum")
        //         {
        //             const op::Sum* sum = static_cast<const op::Sum*>(&node);
        //             reference::sum<T>(args[0]->get_data_ptr<T>(),
        //                               out[0]->get_data_ptr<T>(),
        //                               args[0]->get_shape(),
        //                               out[0]->get_shape(),
        //                               sum->get_reduction_axes());
        //         }
        //         else if (node_op == "Tan")
        //         {
        //             reference::tan<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else if (node_op == "Tanh")
        //         {
        //             reference::tanh<T>(
        //                 args[0]->get_data_ptr<T>(), out[0]->get_data_ptr<T>(), out[0]->get_element_count());
        //         }
        //         else
        //         {
        //             std::stringstream ss;
        //             ss << "unsupported op " << node_op;
        //             throw ngraph_error(ss.str());
        //         }
    }
};
