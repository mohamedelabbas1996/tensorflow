/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/analysis/alias_info.h"

#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// Returns in-place input/output pairs for the given fusion instruction,
// according to the aliasing rules for the corresponding fusion computation.
//
// `instruction` must be a fusion instruction.
std::vector<std::pair<HloOperandIndex, ShapeIndex>>
AliasInfo::GetFusionInstructionInPlaceInputOutputPairs(
    const HloFusionInstruction* fusion) const {
  std::vector<std::pair<HloOperandIndex, ShapeIndex>>
      in_place_input_output_pairs;

  // Each of these leaves represents one array output of the fusion that might
  // be aliased with one of the fusion computation's array inputs (both could be
  // nested arbitrarily deep inside tuples).
  ShapeUtil::ForEachLeafShape(fusion->shape(), [&](const Shape& sub_shape,
                                                   const ShapeIndex& index) {
    // Start from the root instruction of the fusion computation and follow
    // tuple indirection backwards to find the "output source", i.e. the
    // instruction that is the original source of the array output in
    // question. If there is no such indirection the "output source" will
    // just be the fusion root instruction itself.
    const HloInstruction* output_source_instruction =
        fusion->fused_expression_root();
    ShapeIndex output_source_index = index;
    std::tie(output_source_instruction, output_source_index) =
        FollowTupleIndirection(output_source_instruction, output_source_index);

    // The aliasing rules of the "output source" instruction determine the
    // aliasing rules for the entire fusion. If we can connect (following
    // tuple indirection) the input of an "in-place" pair to one of the
    // fusion's inputs, and the output of this "in-place" pair to the fusion
    // output in question, then this fusion input and output must alias.
    auto in_place_pairs = GetInPlaceInputOutputPairs(output_source_instruction);
    ShapeIndex in_place_input_index;
    const HloInstruction* in_place_input_source = nullptr;

    for (const auto& output_source_in_place_pair : in_place_pairs) {
      const HloOperandIndex& input = output_source_in_place_pair.first;
      const ShapeIndex& output_index = output_source_in_place_pair.second;
      if (output_index == output_source_index) {
        // It is not possible for the same output to alias multiple inputs.
        CHECK(in_place_input_source == nullptr);
        in_place_input_source =
            output_source_instruction->operand(input.operand_number);
        in_place_input_index = input.operand_index;
        // Follow tuple indirection backwards from the instruction input to
        // try to find a fusion parameter. If found, that parameter aliases
        // the current output. If not, the current output aliases no input.
        std::tie(in_place_input_source, in_place_input_index) =
            FollowTupleIndirection(in_place_input_source, in_place_input_index);
        if (in_place_input_source->opcode() == HloOpcode::kFusion) {
          // Nested fusions can have aliasing that allows us to peephole
          // through to their producer.
          auto nested_in_place_input_output_pairs =
              GetInPlaceInputOutputPairs(in_place_input_source);
          for (const auto& pair : nested_in_place_input_output_pairs) {
            if (pair.second == in_place_input_index) {
              // If the nested fusion has aliasing that matches the index of
              // this input for its output, then peephole to its input.
              in_place_input_source =
                  in_place_input_source->operand(pair.first.operand_number);
              in_place_input_index = pair.first.operand_index;
              std::tie(in_place_input_source, in_place_input_index) =
                  FollowTupleIndirection(in_place_input_source,
                                         in_place_input_index);
            }
          }
        }
      }
    }
    // Skip bitcast
    if (in_place_input_source != nullptr &&
        in_place_input_source->opcode() == HloOpcode::kBitcast) {
      in_place_input_source = in_place_input_source->operand(0);
    }
    if (in_place_input_source != nullptr &&
        in_place_input_source->opcode() == HloOpcode::kParameter) {
      in_place_input_output_pairs.emplace_back(
          HloOperandIndex{in_place_input_source->parameter_number(),
                          in_place_input_index},
          index);
    }
  });
  return in_place_input_output_pairs;
}

bool AliasInfo::MustAlias(const HloInstruction* operand,
                          const ShapeIndex& operand_index,
                          const HloInstruction* user,
                          const ShapeIndex& user_index) const {
  for (const auto& [hlo_operand_index, user_shape_index] :
       GetInPlaceInputOutputPairs(user)) {
    if (user->operand(hlo_operand_index.operand_number) == operand &&
        hlo_operand_index.operand_index == operand_index &&
        user_shape_index == user_index) {
      return true;
    }
  }
  return false;
}

std::vector<std::pair<HloOperandIndex, ShapeIndex>>
AliasInfo::GetInPlaceInputOutputPairs(const HloInstruction* user) const {
  if (std::optional<std::vector<std::pair<HloOperandIndex, ShapeIndex>>> hint =
          GetNonDefaultInPlaceInputOutputPairs(user)) {
    return *hint;
  }

  // TODO tixxx: nvshmem default one-shot allreduce algo requires
  // separate buffers for IO, remove this once nvshmem is upgraded to 3.3
  if (user->opcode() == HloOpcode::kAllReduceStart) {
    if (absl::StrContainsIgnoreCase(user->raw_backend_config_string(),
                                    "nvshmem")) {
      return {};
    }
  }
  if (IsDefaultInPlaceOperation(user)) {
    int64_t num_in_place_operands = user->operand_count();
    const HloScatterInstruction* scatter = DynCast<HloScatterInstruction>(user);
    if (scatter) {
      num_in_place_operands = scatter->scatter_operand_count();
    } else if (user->opcode() == HloOpcode::kDynamicUpdateSlice) {
      num_in_place_operands = 1;
    }
    // Default handling: one operand shares buffer with single output.
    if (num_in_place_operands == 1) {
      return {{HloOperandIndex{0, {}}, {}}};
    }
    // Default handling: operand i shares buffer with output i.
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(num_in_place_operands);
    for (int i = 0; i < num_in_place_operands; i++) {
      in_place_pairs.push_back({HloOperandIndex{i, {}}, {i}});
    }
    return in_place_pairs;
  }

  // Ops that require special handling.
  if (user->opcode() == HloOpcode::kCollectivePermute &&
      user->operands().size() == 4) {
    if (user->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {}}});
      for (int i = 0; i < user->operand(1)->shape().tuple_shapes().size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {i}});
      }
      return in_place_pairs;
    }
    return {{HloOperandIndex{1, {}}, {}}};
  }
  if (user->opcode() == HloOpcode::kCollectivePermuteStart &&
      user->operands().size() == 4) {
    if (user->operand(1)->shape().IsTuple()) {
      std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs(
          {{HloOperandIndex{1, {}}, {1}}});
      for (int i = 0; i < user->operand(1)->shape().tuple_shapes().size();
           i++) {
        in_place_pairs.push_back({HloOperandIndex{1, {i}}, {1, i}});
      }
      return in_place_pairs;
    }
    return {{HloOperandIndex{1, {}}, {1}}};
  }
  if (user->opcode() == HloOpcode::kCustomCall) {
    // Custom Calls previously assumed that aliased operands were
    // forwarded, but now supports modification semantics.
    const auto& aliasing_pairs =
        Cast<HloCustomCallInstruction>(user)->output_to_operand_aliasing();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(aliasing_pairs.size());
    for (const auto& pair : aliasing_pairs) {
      ShapeIndex output_shape_index = pair.first;
      int64_t operand_index = pair.second.first;
      ShapeIndex operand_shape_index = pair.second.second;
      in_place_pairs.push_back(
          {HloOperandIndex{operand_index, {operand_shape_index}},
           output_shape_index});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kFusion) {
    const HloFusionInstruction* fusion = Cast<HloFusionInstruction>(user);
    const auto& aliasing_pairs = fusion->output_to_operand_aliasing();
    // WARNING: The users of fusion's output_to_operand_aliasing should be aware
    // that the annotated output-operand-aliasing pairs should not conflict with
    // those discovered by GetFusionInstructionInPlaceInputOutputPairs.
    // TODO (b/259460539): Make sure the annotated and discovered pairs do not
    // conflict (possibly through implementing a new pass)
    auto in_place_pairs = GetFusionInstructionInPlaceInputOutputPairs(fusion);
    if (!aliasing_pairs.empty()) {
      for (const auto& pair : aliasing_pairs) {
        ShapeIndex output_shape_index = pair.first;
        int64_t operand_index = pair.second.first;
        ShapeIndex operand_shape_index = pair.second.second;
        in_place_pairs.push_back(
            {HloOperandIndex{operand_index, {operand_shape_index}},
             output_shape_index});
      }
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kAsyncStart) {
    // Custom Calls previously assumed that aliased operands were
    // forwarded, but now supports modification semantics.
    const auto& aliasing_pairs =
        Cast<HloAsyncStartInstruction>(user)->output_to_operand_aliasing();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    in_place_pairs.reserve(aliasing_pairs.size());
    for (const auto& pair : aliasing_pairs) {
      ShapeIndex output_shape_index = pair.first;
      int64_t operand_index = pair.second.first;
      ShapeIndex operand_shape_index = pair.second.second;
      in_place_pairs.push_back(
          {HloOperandIndex{operand_index, {operand_shape_index}},
           output_shape_index});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kAsyncUpdate ||
      user->opcode() == HloOpcode::kAsyncDone) {
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    // AsyncUpdate/AsyncDone always alias their chain operand if shapes match.
    // Index {0} is the context/chain.
    if (user->shape().IsTuple() && user->shape().tuple_shapes_size() > 0 &&
        user->operand(0)->shape().IsTuple() &&
        user->operand(0)->shape().tuple_shapes_size() > 0) {
      int64_t num_prev_params =
          user->operand(0)->shape().tuple_shapes(0).tuple_shapes_size();
      int64_t num_curr_params =
          user->shape().tuple_shapes(0).tuple_shapes_size();

      // 1. Alias prefix from operand(0) {0, i} -> output {0, i}
      for (int64_t i = 0; i < std::min(num_prev_params, num_curr_params); ++i) {
        ShapeUtil::ForEachLeafShape(
            user->shape().tuple_shapes(0).tuple_shapes(i),
            [&](const Shape& subshape, const ShapeIndex& index) {
              ShapeIndex full_index = {0, i};
              full_index.insert(full_index.end(), index.begin(), index.end());
              in_place_pairs.push_back(
                  {HloOperandIndex{0, full_index}, full_index});
            });
      }

      // 2. Alias newly bound operands i -> output {0, i}
      for (int64_t i = num_prev_params; i < num_curr_params; ++i) {
        int64_t operand_idx = i - num_prev_params + 1;
        if (operand_idx < user->operand_count()) {
          ShapeUtil::ForEachLeafShape(
              user->operand(operand_idx)->shape(),
              [&](const Shape& subshape, const ShapeIndex& index) {
                ShapeIndex full_output_index = {0, i};
                full_output_index.insert(full_output_index.end(), index.begin(),
                                         index.end());
                in_place_pairs.push_back(
                    {HloOperandIndex{operand_idx, index}, full_output_index});
              });
        }
      }
    }
    if (user->opcode() == HloOpcode::kAsyncDone) {
      // AsyncDone output aliases the result produced in the async computation.
      // In the context tuple (from operand 0), the result is index {1}.
      if (!user->shape().IsTuple()) {
        in_place_pairs.push_back({HloOperandIndex{0, {1}}, {}});
      } else if (user->shape().tuple_shapes_size() > 0) {
        // If AsyncDone output is a tuple (e.g. (Result, Context)), Result is
        // {0}.
        ShapeUtil::ForEachLeafShape(
            user->shape().tuple_shapes(0),
            [&](const Shape& subshape, const ShapeIndex& index) {
              ShapeIndex operand_index = {1};
              operand_index.insert(operand_index.end(), index.begin(),
                                   index.end());
              ShapeIndex output_index = {0};
              output_index.insert(output_index.end(), index.begin(),
                                  index.end());
              in_place_pairs.push_back(
                  {HloOperandIndex{0, operand_index}, output_index});
            });
      }
    } else {
      // AsyncUpdate late operands alias with their respective parameter slots.
      const HloAsyncInstruction* async_start =
          Cast<HloAsyncInstruction>(user)->async_chain_start();
      int64_t num_prev_params = async_start->operand_count();
      if (user->shape().IsTuple() && user->shape().tuple_shapes_size() > 0 &&
          user->shape().tuple_shapes(0).IsTuple()) {
        for (int64_t i = 1; i < user->operand_count(); ++i) {
          int64_t param_idx = num_prev_params + i - 1;
          if (param_idx < user->shape().tuple_shapes(0).tuple_shapes_size()) {
            ShapeUtil::ForEachLeafShape(
                user->operand(i)->shape(),
                [&](const Shape& subshape, const ShapeIndex& index) {
                  ShapeIndex full_output_index = {0, param_idx};
                  full_output_index.insert(full_output_index.end(),
                                           index.begin(), index.end());
                  in_place_pairs.push_back(
                      {HloOperandIndex{i, index}, full_output_index});
                });
          }
        }
      }
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kSetDimensionSize) {
    int64_t dimension = user->dimension();
    std::vector<std::pair<HloOperandIndex, ShapeIndex>> in_place_pairs;
    if (user->shape().is_dynamic_dimension(dimension) ==
        user->shape().is_dynamic_dimension(dimension)) {
      in_place_pairs.push_back({HloOperandIndex{0, {}}, {}});
    }
    return in_place_pairs;
  }
  if (user->opcode() == HloOpcode::kRaggedAllToAll) {
    return {{HloOperandIndex{1, {}}, {}}};
  }
  return {};
}

std::pair<const HloInstruction*, ShapeIndex> FollowTupleIndirection(
    const HloInstruction* instruction, ShapeIndex operand_index) {
  while (instruction->opcode() == HloOpcode::kTuple && !operand_index.empty()) {
    instruction = instruction->operand(operand_index.front());
    operand_index.pop_front();
  }
  while (instruction->opcode() == HloOpcode::kGetTupleElement) {
    operand_index.push_front(instruction->tuple_index());
    instruction = instruction->operand(0);
  }

  return {instruction, operand_index};
}

}  // namespace xla
