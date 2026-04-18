import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvGRU(nn.Module):
    def __init__(self, input_size, window_in, num_layers, encoder_params, device, num_classes=4):
        super().__init__()

        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size
        self.window_in = window_in
        self.num_layers = num_layers
        self.encoder_params = encoder_params
        self.num_classes = num_classes

        self.encoder = self.__define_block(encoder_params)
        last_hidden_dim = encoder_params['hidden_dims'][-1]
        self.classifier_head = nn.Conv2d(
            in_channels=last_hidden_dim,
            out_channels=self.num_classes,
            kernel_size=1,
        )

        self.hidden = None
        self.is_trainable = True

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.encoder[i].init_hidden(batch_size))
        return init_states

    def __define_block(self, block_params):
        input_dim = block_params['input_dim']
        hidden_dims = block_params['hidden_dims']
        kernel_size = block_params['kernel_size']
        bias = block_params['bias']

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvGRUCell(
                    input_size=(self.height, self.width),
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_size[i],
                    bias=bias,
                    device=self.device,
                )
            )
        return nn.ModuleList(cell_list)

    def forward(self, x, hidden=None, **kwargs):
        _, cur_states = self.__forward_block(x, hidden, return_all_layers=True)
        last_layer_h = cur_states[-1]
        drought_map = self.classifier_head(last_layer_h)
        return drought_map

    def __forward_block(self, input_tensor, hidden_state, return_all_layers):
        layer_output_list = []
        layer_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h = self.encoder[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=h,
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            layer_state_list.append(h)

        if not return_all_layers:
            layer_output_list = layer_output_list[-1]
            layer_state_list = layer_state_list[-1]

        return layer_output_list, layer_state_list


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device):
        super().__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2
        self.device = device

        self.conv_gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.conv_candidate = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = torch.split(gates, self.hidden_dim, dim=1)

        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        combined_candidate = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        candidate = torch.tanh(self.conv_candidate(combined_candidate))
        h_next = (1 - update_gate) * h_cur + update_gate * candidate
        return h_next

    def init_hidden(self, batch_size):
        hidden = Variable(
            torch.zeros(batch_size, self.hidden_dim, self.height, self.width)
        ).to(self.device)
        return hidden
