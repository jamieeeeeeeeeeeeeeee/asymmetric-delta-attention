# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.ada.configuration_ada import ADAConfig
from fla.models.ada.modeling_ada import ADAForCausalLM, ADAModel

AutoConfig.register(ADAConfig.model_type, ADAConfig, exist_ok=True)
AutoModel.register(ADAConfig, ADAModel, exist_ok=True)
AutoModelForCausalLM.register(ADAConfig, ADAForCausalLM, exist_ok=True)

__all__ = ['ADAConfig', 'ADAForCausalLM', 'ADAModel']
