import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTM(nn.Module):
    #移除了 decoder_params 和 window_out，因为我们只需要对最后时刻输出一张分类图
    def __init__(self, input_size, window_in, num_layers, encoder_params, device):
        nn.Module.__init__(self)

        self.device = device
        self.input_size = input_size
        self.height, self.width = self.input_size

        self.window_in = window_in
        self.num_layers = num_layers
        self.encoder_params = encoder_params

        # 定义 encoder
        self.encoder = self.__define_block(encoder_params)

        # ---------------- 【核心修改】增加干旱等级分类头 ----------------
        # 提取最后一层隐藏层的通道数
        last_hidden_dim = encoder_params['hidden_dims'][-1]
        # 获取干旱分类的类别数（默认为4：无旱、轻度、中度、重度）
        self.num_classes = encoder_params.get('num_classes', 4)
        
        # 使用 1x1 卷积将隐藏状态映射到干旱等级类别上
        self.classifier_head = nn.Conv2d(
            in_channels=last_hidden_dim, 
            out_channels=self.num_classes,
            kernel_size=1
        )
        # -------------------------------------------------------------

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
        peephole_con = block_params['peephole_con']

        # 定义模块
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
        """
        :param x: 5-D tensor of shape (b, t, d, m, n) 即 (Batch, Time, Channel, H, W)
        :param hidden:
        :return: (Batch, num_classes, H, W)
        """
        # 前向传播 encoder，获取所有层的隐藏状态
        _, cur_states = self.__forward_block(x, hidden, 'encoder', return_all_layers=True)

        # ---------------- 【核心修改】提取特征并分类 ----------------
        # cur_states 的结构是每一层的 [h_state, c_state]
        # 我们取最后一层 (cur_states[-1]) 的隐藏状态 h_state (索引0)
        # 其形状为 (Batch, hidden_dim, Height, Width)
        last_layer_h = cur_states[-1][0]

        # 通过分类头生成干旱等级图
        # 输出形状为 (Batch, num_classes, Height, Width)
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
            for t in range(seq_len):
                h, c = block[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                        cur_state=[h, c])
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
    # (保持原样，这部分负责基础的 LSTM 单元运算，不需要修改)
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
