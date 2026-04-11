import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer,
                 learning_rate, weight_decay, momentum, device, selected_dim=-1):
        self.num_epochs = num_epochs
        self.clip = clip
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = device
        
        # ---------------- 【核心修改】更换为多分类损失函数 ----------------
        self.criterion = nn.CrossEntropyLoss()
        # ---------------- 【核心修改】评估指标改为准确率 ----------------
        self.metric_names = ["Accuracy"]

    def train(self, model, batch_generator):
        model = model.to(self.device)
        model.train()

        optimizer = self.__get_optimizer(model)
        train_loss, val_loss = [], []
        tolerance, best_epoch, best_val_loss, best_train_loss = 0, 0, 1e6, 1e6
        best_train_metric, best_val_metric = None, None
        best_dict = model.state_dict()
        for epoch in range(self.num_epochs):
            start_time = time.time()
            running_train_loss, train_metric_scores = self.__step_loop(model=model,
                                                                       generator=batch_generator,
                                                                       mode='train',
                                                                       optimizer=optimizer)
            running_val_loss, val_metric_scores = self.__step_loop(model=model,
                                                                   generator=batch_generator,
                                                                   mode='val',
                                                                   optimizer=None)
            epoch_time = time.time() - start_time

            train_loss.append(running_train_loss)
            val_loss.append(running_val_loss)

            train_metric_str = self.get_metric_string(metric_scores=train_metric_scores)
            val_metric_str = self.get_metric_string(metric_scores=val_metric_scores)
            print(f"\t --> Epoch:{epoch + 1}/{self.num_epochs} took {epoch_time:.3f} secs:\t"
                  f"Train_loss: {running_train_loss:.5f}, {train_metric_str}\t "
                  f"Val_loss: {running_val_loss:.5f}, {val_metric_str}")

            if running_val_loss < best_val_loss:
                best_dict = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                best_val_loss = running_val_loss
                best_val_metric = val_metric_scores
                best_train_loss = running_train_loss
                best_train_metric = train_metric_scores
                tolerance = 0
            else:
                tolerance += 1

            if tolerance > self.tolerance or epoch == self.num_epochs - 1:
                model.load_state_dict(best_dict)
                train_metric_str = self.get_metric_string(metric_scores=best_train_metric)
                val_metric_str = self.get_metric_string(metric_scores=best_val_metric)
                print(f"\tEarly exiting from epoch: {best_epoch}:\t"
                      f"Train_loss {best_train_loss:.5f}, {train_metric_str}\t"
                      f"Validation_loss: {best_val_loss:.5f}, {val_metric_str}")
                break
            torch.cuda.empty_cache()

        return (train_loss, val_loss), best_train_metric, best_val_metric

    def evaluate(self, model, batch_generator):
        # 简化版保留原样
        pass 
    
    def predict(self, model, batch_generator):
        # 简化版保留原样
        pass

    def __step_loop(self, model, generator, mode, optimizer):
        if mode in ['test', 'val']:
            step_fun = self.__val_step
        else:
            step_fun = self.__train_step

        running_loss, running_metric_scores = 0, {key: 0 for key in self.metric_names}
        for idx, (x, y) in enumerate(generator.generate(mode)):
            print('\r\t{}:{}/{}'.format(mode, idx, generator.num_iter(mode)), flush=True, end='')

            if hasattr(model, 'hidden'):
                hidden = model.init_hidden(batch_size=x.shape[0])
            else:
                hidden = None

            # ---------------- 【核心修改】提取特征与标签 ----------------
            x = self.__prep_input(x)
            # 假设 y 的原始输入是 (Batch, Time, Channel, H, W)
            # 针对分类任务，我们需要最后的干旱标签矩阵 (Batch, H, W)，并转为 long 整数型
            # 如果你的 Dataloader 里 y 已经是 (Batch, H, W)，这里可以改为 y_target = y.long().to(self.device)
            y_target = y.long().to(self.device)
            
            loss, metric_scores = step_fun(model=model,
                                           inputs=[x, y_target, hidden],
                                           optimizer=optimizer)
            running_loss += loss
            for key, score in metric_scores.items():
                running_metric_scores[key] += score

        running_loss /= (idx + 1)
        for key, score in running_metric_scores.items():
            running_metric_scores[key] = score / (idx + 1)

        return running_loss, running_metric_scores

    def __train_step(self, model, inputs, optimizer):
        x, y_target, hidden = inputs
        if optimizer is not None:
            optimizer.zero_grad()
            
        pred = model.forward(x=x, hidden=hidden) # pred 形状: (Batch, 4类别, H, W)
        
        # 计算交叉熵损失
        loss = self.criterion(pred, y_target)

        if model.is_trainable and optimizer is not None:
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            optimizer.step()

        loss_val, metric_scores = self.__calc_scores(pred, y_target)

        return loss_val, metric_scores

    def __val_step(self, model, inputs, optimizer):
        x, y_target, hidden = inputs
        pred = model.forward(x=x, hidden=hidden)
        loss_val, metric_scores = self.__calc_scores(pred, y_target)
        return loss_val, metric_scores

    def __prep_input(self, x):
        x = x.float().to(self.device)
        # (b, t, m, n, d) -> (b, t, d, m, n) 适应 ConvLSTM 的维度要求
        # x = x.permute(0, 1, 4, 2, 3)
        return x

    def __get_optimizer(self, model):
        if model.is_trainable:
            if self.optimizer == "adam":
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            else:
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        else:
            optimizer = None
        return optimizer

    def __calc_scores(self, pred, y_target):
        # 移除原有的正态逆变换，因为现在是分类任务
        loss = self.criterion(pred, y_target).detach().cpu().numpy()
        
        # ---------------- 【核心修改】计算像素级分类准确率 ----------------
        # 找出每个像素点概率最大的类别索引 (Batch, H, W)
        pred_classes = torch.argmax(pred, dim=1)
        correct_pixels = (pred_classes == y_target).float()
        accuracy = correct_pixels.mean().item()
        
        metric_scores = {"Accuracy": accuracy}
        return loss, metric_scores

    @staticmethod
    def get_metric_string(metric_scores):
        message = ""
        for key, score in metric_scores.items():
            message += f"{key}: {score:.4f}, "
        return message
