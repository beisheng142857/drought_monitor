import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attn_channel, kernel_size):
        super(Attention, self).__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.input_dim = input_dim

        self.H = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=attn_channel,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )

        self.W = nn.Conv2d(
            in_channels=input_dim,
            out_channels=attn_channel,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )

        self.V = nn.Conv2d(
            in_channels=attn_channel,
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=False,
        )

        self.saved_attn_weights = None

    def forward(self, input_tensor, hidden):
        """
        :param torch.Tensor input_tensor: (B, D, H, W)
        :param tuple of torch.Tensor hidden: ((B, hidden, H, W), (B, hidden, H, W))
        :return: attention energies, (B, D, H, W)
        """
        hid_conv_out = self.H(hidden[0])
        in_conv_out = self.W(input_tensor)
        energy = self.V((hid_conv_out + in_conv_out).tanh())

        # 使用 detach() 从计算图中分离，cpu() 放到内存，numpy() 转成数组
        self.saved_attn_weights = energy.detach().cpu().numpy()
        
        return energy
