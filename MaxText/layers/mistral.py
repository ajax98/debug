"""
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module


from typing import Optional

import jax.experimental
from layers import quantizations
from layers import linears
from layers import initializers
import jax
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import Mesh
from flax import linen as nn
import jax.numpy as jnp
from layers import attentions
from layers import embeddings
from layers import normalizations
from layers import models
import common_types
import max_logging
import numpy as np
import time

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization

# -----------------------------------------
# The Decoder Layer for Mistral or Mixtral
# -----------------------------------------

num_layers_passed = 0

def save_array_host(array, type):
    global num_layers_passed
    print("num_layers_passed: ", num_layers_passed)
    if num_layers_passed == 0:
      print("Storing array ", type, " of dtype: ", array.dtype,  " and shape: ", array.shape)
      print("Storing array: ", array)
      convert_array = array.astype(jnp.float32)
      timestamp = time.time()
      output_name = type + "_" + str(timestamp) + ".npy"
      np.save(output_name, convert_array)
      print('saved array')

def save_input(array):
  save_array_host(array, "input")

def save_post_attention_output(array):
  save_array_host(array, "after_attention")

def save_post_moe_output(array):
  save_array_host(array, "after_moe")
    
def save_final_output(array):
  global num_layers_passed
  save_array_host(array, "final")
  num_layers_passed+=1

# @jax.jit
# def save_array(x):
#     jax.experimental.io_callback(save_array_host, None, x)

class MistralDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    jax.debug.print("running mistral decoder layer timestamp: " + str(time.time()))
    jax.experimental.io_callback(save_input, None, inputs)
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    lnx_rms = models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )
    lnx = lnx_rms(inputs)

    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    # Self-attention block
    attention_layer = Attention(
        config=cfg,
        num_query_heads=cfg.num_query_heads,
        num_kv_heads=cfg.num_kv_heads,
        head_dim=cfg.head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        mesh=mesh,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        dropout_rate=cfg.dropout_rate,
        name="self_attention",
        quant=self.quant,
        kv_quant=quantizations.configure_kv_quant(cfg),
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )
    jax.experimental.io_callback(save_post_attention_output, None, attention_lnx)


    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ("activation_batch", "activation_length", "activation_embed"))

    load_balance_loss = None
    if cfg.num_experts > 1:
      mlp_lnx, load_balance_loss = linears.MoeBlock(
          config=cfg,
          num_experts=cfg.num_experts,
          num_experts_per_tok=cfg.num_experts_per_tok,
          mesh=mesh,
          kernel_init=initializers.nd_dense_init(1.0, "fan_in", "truncated_normal"),
          kernel_axes=("embed", "mlp"),
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          quant=self.quant,
      )(hidden_states)
      mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))
      jax.experimental.io_callback(save_post_moe_output, None, mlp_lnx)
    else:
      mlp_lnx = linears.MlpBlock(
          intermediate_dim=cfg.mlp_dim,
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          config=cfg,
          quant=self.quant,
      )(hidden_states, deterministic=deterministic)
      mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))


    layer_output = mlp_lnx + intermediate_inputs
    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)
  
    # jax.debug.print("First element of layer output is: {x}", x=jnp.ravel(layer_output)[0])
    jax.experimental.io_callback(save_final_output, None, layer_output)
  
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )
    # jax.debug.print("ajaygopi@ mistral decoder layer output: {x}", x=layer_output)
    # jax.numpy.save(("mistral_decoder_layer_output_" + str(time.time())), layer_output)

    if cfg.num_experts > 1 and load_balance_loss is not None:
      self.sow("intermediates", "moe_lb_loss", load_balance_loss)

    if cfg.record_internal_nn_metrics:
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
