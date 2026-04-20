import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.drought.attention import Attention

class ConvLSTM(nn.Module):
    def __init__(self, input_size, window_in, num_layers, encoder_params, input_attn_params, device):
        nn.Module.__init__(self)

        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size

        self.window_in = window_in
        self.num_layers = num_layers
        self.encoder_params = encoder_params

        if input_attn_params is not None:
            self.input_attn = Attention(
                input_dim=input_attn_params["input_dim"],
                hidden_dim=input_attn_params["hidden_dim"],
                attn_channel=input_attn_params["attn_channel"],
                kernel_size=input_attn_params["kernel_size"]
            )
        else:
            self.input_attn = None

        self.encoder = self.__define_block(encoder_params)
        last_hidden_dim = encoder_params['hidden_dims'][-1]
        self.num_classes = encoder_params.get('num_classes', 4)
        self.classifier_head = nn.Conv2d(
            in_channels=last_hidden_dim,
            out_channels=self.num_classes,
            kernel_size=1
        )

        self.hidden = None
        self.is_trainable = True
    
    def __forward_input_attn(self, x_t, hidden):
        if self.input_attn is None:
            return x_t, None

        alpha_tensor = self.input_attn(x_t, hidden)
        alpha_tensor = torch.sigmoid(alpha_tensor)
        x_t_attn = x_t * (1.0 + alpha_tensor)

        return x_t_attn, alpha_tensor
    
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
        peephole_con = block_params['peephole_con']

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list += [self.__create_cell_unit(cur_input_dim,
                                                  hidden_dims[i],
                                                  kernel_size[i],
                                                  bias,
                                                  peephole_con)]
        block = nn.ModuleList(cell_list)
        return block

    def forward(self, x, hidden, **kwargs):
        _, cur_states = self.__forward_block(x, hidden, 'encoder', return_all_layers=True)
        last_layer_h = cur_states[-1][0]
        drought_map = self.classifier_head(last_layer_h)
        return drought_map

    def __forward_block(self, input_tensor, hidden_state, block_name, return_all_layers):
        block = getattr(self, block_name)
        layer_output_list = []
        layer_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            alphas = []
            
            for t in range(seq_len):
                x_t = cur_layer_input[:, t, :, :, :]

                if block_name == 'encoder' and layer_idx == 0:
                    x_t, alpha = self.__forward_input_attn(x_t, hidden=(h, c))
                    alphas.append(alpha)

                h, c = block[layer_idx](
                    input_tensor=x_t,
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            layer_state_list.append([h, c])

        if not return_all_layers:
            layer_output_list = layer_output_list[-1]
            layer_state_list = layer_state_list[-1]

        return layer_output_list, layer_state_list

    def __create_cell_unit(self, cur_input_dim, hidden_dim, kernel_size, bias, peephole_con):
        cell_unit = ConvLSTMCell(input_size=(self.height, self.width),
                                 input_dim=cur_input_dim,
                                 hidden_dim=hidden_dim,
                                 kernel_size=kernel_size,
                                 bias=bias,
                                 device=self.device,
                                 peephole_con=peephole_con)
        return cell_unit


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim,
                 kernel_size, bias, device, peephole_con=False):
        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = self.kernel_size // 2

        self.peephole_con = peephole_con
        self.device = device

        if peephole_con:
            self.w_peep = None

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        if self.peephole_con:
            w_ci, w_cf, w_co = torch.split(self.w_peep, self.hidden_dim, dim=1)
            cc_i += w_ci * c_cur
            cc_f += w_cf * c_cur

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g

        o = torch.sigmoid(cc_o + w_co * c_next) if self.peephole_con else torch.sigmoid(cc_o)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        if self.peephole_con:
            self.w_peep = Variable(torch.zeros(batch_size,
                                               self.hidden_dim * 3,
                                               self.height, self.width)).to(self.device)

        hidden = (Variable(torch.zeros(batch_size, self.hidden_dim,
                                       self.height, self.width)).to(self.device),
                  Variable(torch.zeros(batch_size,
                                       self.hidden_dim, self.height, self.width)).to(self.device))
        return hidden
