import onmt
import torch
import torch.nn as nn


if __name__ == "__main__":

    from onmt.models.multilingual_translator.reversible_transformers import reversible_encoder, \
        ReversibleTransformerEncoderLayer

    import argparse

    parser = argparse.ArgumentParser(description='reversible transformer')
    parser.add_argument('-model_size', type=int, default=32,
                        help='Size of embedding / transformer hidden')
    parser.add_argument('-gpu', default=0, type=int,
                        help="Seed for deterministic runs.")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)

    opt.layers = 4
    opt.variational_dropout = False
    opt.dropout = 0.0
    opt.attn_dropout = 0.0
    opt.residual_dropout = 0.0
    opt.ffn_dropout = 0.0
    opt.n_heads = 4
    opt.inner_size = 4 * opt.model_size
    opt.ffn_glu = False
    opt.ffn_activation = 'gelu'
    opt.head_dim = opt.model_size // opt.n_heads
    opt.learnable_position_encoding = False

    layers = torch.nn.ModuleList()

    for l in range(opt.layers):
        layer = ReversibleTransformerEncoderLayer(opt)
        layers.append(layer)

    # layers = layers.cuda()


    class TestNetwork(torch.nn.Module):

        def __init__(self, layers):
            super().__init__()
            self.function = reversible_encoder
            self.layers = layers

        def forward(self, input, pos):

            return self.function(self.layers, input, pos, None)


    bsz = 4
    len_q = 7
    len_r = 7

    device=torch.device('cuda:0')
    input_states = torch.randn(*(len_q, bsz, opt.model_size), dtype=torch.float64, requires_grad=True, device=device)
    pos = torch.randn(*(len_q, 1, opt.model_size), dtype=torch.float64, requires_grad=False, device=device)

    net = TestNetwork(layers)
    net = net.double().cuda()

    print(net)

    print("gradchecking start.")

    torch.autograd.gradcheck(net, (input_states, pos), eps=1e-6, atol=1e-5, rtol=1e-3)

    print("gradchecking completed.")
