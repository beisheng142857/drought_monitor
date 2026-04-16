import time
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from sklearn.metrics import f1_score # 【新增】用于计算 Macro-F1

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, num_epochs, early_stop_tolerance, clip, optimizer,
                 learning_rate, weight_decay, momentum, device, class_weights=None, selected_dim=-1):
        self.num_epochs = num_epochs
        self.clip = clip
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.tolerance = early_stop_tolerance
        self.device = device
        
        # ---------------- 【核心优化】加入类别权重 ----------------
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # ---------------- 【核心优化】加入 Macro-F1 指标 ----------------
        self.metric_names = ["Accuracy", "Macro-F1"]

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
            print(f"\n\t --> Epoch:{epoch + 1}/{self.num_epochs} took {epoch_time:.3f} secs:\t"
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
                print(f"\n\tEarly exiting from epoch: {best_epoch}:\t"
                      f"Train_loss {best_train_loss:.5f}, {train_metric_str}\t"
                      f"Validation_loss: {best_val_loss:.5f}, {val_metric_str}")
                break
            torch.cuda.empty_cache()

        return (train_loss, val_loss), best_train_metric, best_val_metric

    def evaluate(self, model, batch_generator):
        """【新增】用于在测试集上评估模型"""
        model = model.to(self.device)
        model.eval()
        with torch.no_grad():
            test_loss, test_metric_scores = self.__step_loop(model=model,
                                                             generator=batch_generator,
                                                             mode='test',
                                                             optimizer=None)
        return test_loss, test_metric_scores
    
    def predict(self, model, batch_generator):
        """【新增】用于输出预测的干旱图，方便后续可视化"""
        model = model.to(self.device)
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for x, y in batch_generator.generate('test'):
                x = self.__prep_input(x)
                if hasattr(model, 'hidden'):
                    hidden = model.init_hidden(batch_size=x.shape[0])
                else:
                    hidden = None
                
                pred = model.forward(x=x, hidden=hidden)
                pred_classes = torch.argmax(pred, dim=1) # 获取分类结果
                
                predictions.append(pred_classes.cpu())
                targets.append(y.cpu())
                
        return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

    def __step_loop(self, model, generator, mode, optimizer):
        if mode in ['test', 'val']:
            step_fun = self.__val_step
            model.eval()
        else:
            step_fun = self.__train_step
            model.train()

        running_loss, running_metric_scores = 0, {key: 0 for key in self.metric_names}
        for idx, (x, y) in enumerate(generator.generate(mode)):
            print('\r\t{}:{}/{}'.format(mode, idx+1, generator.num_iter(mode)), flush=True, end='')

            if hasattr(model, 'hidden'):
                hidden = model.init_hidden(batch_size=x.shape[0])
            else:
                hidden = None

            x = self.__prep_input(x)
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
            
        pred = model.forward(x=x, hidden=hidden) 
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
        return x

    def __get_optimizer(self, model):
        if model.is_trainable:
            if self.optimizer == "adam":
                optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            else:
                optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            optimizer = None
        return optimizer

    def __calc_scores(self, pred, y_target):
        loss = self.criterion(pred, y_target).detach().cpu().numpy()
        
        pred_classes = torch.argmax(pred, dim=1)
        
        # 提取到 CPU 转为 numpy 以计算高级指标
        y_true_np = y_target.cpu().numpy().flatten()
        y_pred_np = pred_classes.cpu().numpy().flatten()
        
        # 计算 Accuracy
        accuracy = (y_true_np == y_pred_np).mean()
        
        # 计算 Macro-F1 (综合考虑所有类别的表现，避免大类吃小类)
        macro_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        
        metric_scores = {
            "Accuracy": accuracy,
            "Macro-F1": macro_f1
        }
        return loss, metric_scores

    @staticmethod
    def get_metric_string(metric_scores):
        message = ""
        for key, score in metric_scores.items():
            message += f"{key}: {score:.4f}, "
        return message