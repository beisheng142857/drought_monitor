from configs.config_generator import Param

experiment_params = {
    "global_start_date": '2015-01-01',
    "global_end_date": '2016-12-31',
    "data_step": 6,     # in months
    "data_length": 24,  # in months
    "val_ratio": 0.1,
    "test_ratio": 0.0,
    "normalize_flag": True,
    "model": "weather_model",
    "device": 'cuda',
    "selected_criterion": "MSE"  # choices are MSE, MAE, and MAPE
}

data_params = {
    "rebuild": False,
    "dump_data_folder": "train_data",
    # if rebuild true we extract from nc files inside data_raw (only for high res data)
    "weather_raw_dir": 'data/data_raw',
    "spatial_range": [],  # [[30, 45], [20, 50]],
    "weather_freq": 1,
    "downsample_mode": "selective",  # can be average or selective
    "check_files": False,
    "features": ['d', 'cc', 'z', 'pv', 'r', 'ciwc', 'clwc', 'q', 'crwc', 'cswc', 't', 'u', 'v', 'w'],
    "atm_dim": -1,
    "target_dim": 10,
    "smooth": False,
    "smooth_win_len": 31  # select odd
}

model_params = {
    "moving_avg": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
            "stride": 1,
            "temporal_freq": 1,
            "max_temporal_freq": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "window_in": 10,  # should be same with batch_gen["window_in_len"]
            "window_out": 5,  # should be same with batch_gen["window_out_len"]
            "output_dim": 5,
            "mode": "WMA"
        }
    },
    "convlstm": {
        "batch_gen": {
            # 当前拼成张量时的顺序是: 0:NDVI, 1:VV, 2:VH, 3:VVVH
            "input_dim": [0, 1, 2, 3],  
            "output_dim": 1,         # 输出1张分类图
            "window_in_len": 5,      # 5个月的输入时间步
            "window_out_len": 1,     # (不再用于Decoder，仅作为占位符)
            "batch_size": 16,        # 128x128可以在Colab上使用较大的batch_size
            "shuffle": True,
            "stride": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "input_size": (128, 128),  # ★ 直接匹配你的 100m 分辨率切片图块大小
            "window_in": 5,            
            "num_layers": 2,           # 建议增加为2层以提取更复杂的时空特征
            "encoder_params": {
                "input_dim": 4,        # ★ 4个输入通道 (NDVI, VV, VH, VVVH)
                "hidden_dims": [64, 64], # LSTM隐藏状态通道数，最后一层决定输出前的特征维度
                "kernel_size": [3, 3],
                "bias": True,
                "peephole_con": False,
                "num_classes": 4       # ★ 新增：输出4个干旱等级（无，轻，中，重）
            },
            "input_attn_params": {
                "input_dim": 1,        # 需与数据特征维度一致
                "hidden_dim": 64,      # 建议与 Encoder 第一层 hidden_dim 一致
                "attn_channel": 16,    # 注意力中间卷积通道数
                "kernel_size": 3       # 卷积核大小
            }
        },
    },
    "u_net": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
            "stride": 1,
            "temporal_freq": 1,
            "max_temporal_freq": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "selected_dim": 5,
            "in_channels": 10,  # should be same with batch_gen["window_in_len"]
            "out_channels": 5  # should be same with batch_gen["window_out_len"]
        }

    },
    "weather_model": {
        "batch_gen": {
            # "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],  # indexes of selected features for era5
            # "output_dim": 10,  # indexes of selected features for era5
            "input_dim": "all",  # indexes of selected features for weather bench
            "output_dim": 13,  # index for temperature in weather bench 13
            "window_in_len": 20,
            "window_out_len": 72,
            "batch_size": 8,
            "shuffle": True,
            "stride": 1,
            "temporal_freq": 1,
            "max_temporal_freq": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            # "input_size": (61, 121),
            "input_size": (32, 64),  # weatherbench inputsize
            "window_in": 20,  # must be same with batch_gen["window_in_len"]
            "window_out": 72,  # must be same with batch_gen["window_out_len"]
            "num_layers": 1,
            "selected_dim": [13],  # indexes of batch_gen["output_dim"] on batch_gen["input_dim"] list
            "input_attn_params": {
                "input_dim": 20,
                "hidden_dim": 32,
                "attn_channel": 5,
                "kernel_size": 3
            },
            "encoder_params": {
                "input_dim": 19,
                "hidden_dims": [32],
                "kernel_size": [3],
                "bias": False,
                "peephole_con": False
            },
            "decoder_params": {
                "input_dim": 1,  # must be same with len(core["selected_dim"])
                "hidden_dims": [32],
                "kernel_size": [3],
                "bias": False,
                "peephole_con": False
            },
            "output_conv_params": {
                "mid_channel": 5,
                "out_channel": 1,  # must be same with len(batch_gen["output_dim"])
                "in_kernel": 3,
                "out_kernel": 1
            }
        }
    },
    "lstm": {
        "batch_gen": {
            "input_dim": [0, 2, 3, 4, 7, 10, 11, 12, 13],
            "output_dim": 10,
            "window_in_len": 10,
            "window_out_len": 5,
            "batch_size": 8,
            "shuffle": True,
            "stride": 1,
            "temporal_freq": 1,
            "max_temporal_freq": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 6
        },
        "core": {
            "input_size": (61, 121),
            "window_in": 10,
            "window_out": 5,  # must be same with batch_gen["window_out_len"]
            "num_layers": 2,
            "selected_dim": 5,
            "hidden_dim": 256,
            "dropout": 0.1,
            "bias": True
        }
    },
    "traj_gru": {
        "batch_gen": {
            "input_dim": [0, 1, 2, 3],  # 统一为 NDVI, VV, VH, VVVH
            "output_dim": 1,
            "window_in_len": 5,         # 5个月的输入时间步
            "window_out_len": 1,
            "batch_size": 16,           # 保持与ConvLSTM相同的Batch Size
            "shuffle": True,
            "stride": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "input_size": (128, 128),   # 匹配你的 100m 分辨率切片
            "window_in": 5, 
            "window_out": 1,
            "encoder_params": {
                "input_dim": 4,         # 4个输入通道
                "hidden_dim": 64,       # 建议设定为64对标ConvLSTM
                "kernel_size": 3,
                "bias": True,
                "connection": 1
            },
            "decoder_params": {
                "input_dim": 1,
                "hidden_dim": 64,
                "kernel_size": 3,
                "bias": True,
                "connection": 1
            },
            "num_classes": 4            # 输出4个干旱等级（无，轻，中，重）
        },
    },
    "convgru": {
        "batch_gen": {
            "input_dim": [0, 1, 2, 3],
            "output_dim": 1,
            "window_in_len": 5,
            "window_out_len": 1,
            "batch_size": 16,
            "shuffle": True,
            "stride": 1
        },
        "trainer": {
            "num_epochs": 50,
            "momentum": 0.7,
            "optimizer": "adam",
            "weight_decay": 0.00023,
            "learning_rate": Param([0.01, 0.001, 0.0005, 0.00001]),
            "clip": 5,
            "early_stop_tolerance": 4
        },
        "core": {
            "input_size": (128, 128),
            "window_in": 5,
            "num_layers": 2,
            "encoder_params": {
                "input_dim": 4,
                "hidden_dims": [64, 64],
                "kernel_size": [3, 3],
                "bias": True
            },
            "num_classes": 4
        }
    },
}
