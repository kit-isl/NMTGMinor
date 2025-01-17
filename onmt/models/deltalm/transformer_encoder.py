import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
# from fairseq.modules import (
#     FairseqDropout,
#     LayerDropModuleList,
#     LayerNorm,
#     PositionalEmbedding,
#     SinusoidalPositionalEmbedding,
# )
from .modules.positional_embeddings import PositionalEmbedding, SinusoidalPositionalEmbedding
from .modules.layer_drop import LayerDropModuleList
from onmt.modules.layer_norm import LayerNorm
from .modules.transformer_layer import TransformerEncoderLayerBase


class TransformerEncoderBase(nn.Module):
    """
    Transformer encoder consisting of *cfg.encoder.layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, cfg, embed_tokens):
        self.cfg = cfg
        super(TransformerEncoderBase, self).__init__()

        # TODO
        # self.dictionary = dictionary

        self.register_buffer("version", torch.Tensor([3]))

        # TODO
        self.dropout_module = nn.Dropout(cfg.dropout)

        # TODO
        self.encoder_layerdrop = cfg.encoder_layerdrop

        # TODO
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = cfg.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.no_token_positional_embeddings:
            self.embed_positions = (
                PositionalEmbedding(
                    cfg.max_source_positions,
                    embed_dim,
                    self.padding_idx,
                    learned=cfg.encoder_learned_pos,
                )
            )
        else:
            self.embed_positions = None

        # TODO
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        self.layers.extend(
            [self.build_encoder_layer(cfg) for i in range(cfg.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerBase(cfg)
        # removed the checkpointing and fdsp part

        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        return x, embed

    def forward(
        self,
        src_tokens,
        src_mask: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):

        return self.forward_scriptable(
            src_tokens, src_mask, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_mask: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """

        """
        # compute padding mask
        if src_mask is None:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
        else:
            encoder_padding_mask = src_mask
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for layer in self.layers:
            x = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not support returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        # return {
        #     "encoder_out": [x],  # T x B x C
        #     "encoder_padding_mask": [encoder_padding_mask],  # B x T
        #     "encoder_embedding": [encoder_embedding],  # B x T x C
        #     "encoder_states": encoder_states,  # List[T x B x C]
        #     "src_tokens": [],
        #     "src_lengths": [src_lengths],
        # }
        return x, encoder_padding_mask, encoder_embedding, encoder_states