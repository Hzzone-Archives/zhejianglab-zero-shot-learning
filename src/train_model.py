import torch
import time
import copy
import sys
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from visdom import Visdom



def train_model(model, criterion, optimizer, dataloaders, num_epochs=25, task="classification"):

    scheduler = lr_scheduler.StepLR(optimizer, step_size=num_epochs//4, gamma=0.1)

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()


    model_name = model.__class__.__name__

    viz = Visdom(env=model_name)


    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    if task == "wordembeddings":
        target = dataloaders["train"].dataset.labelObject.get_class_wordembeddings()
    elif task == "attribute":
        pass

    try:
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 每一个迭代都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train(True)  # 设置 model 为训练 (training) 模式
                else:
                    model.train(False)  # 设置 model 为评估 (evaluate) 模式

                running_loss = 0.0
                running_corrects = 0

                # 遍历数据
                for bth_index, data in enumerate(dataloaders[phase]):
                    # 获取输入
                    if task == "classification":
                        inputs, labels, _ = data
                    else:
                        inputs, true_labels, labels = data

                    # inputs, labels = data
                    # print(inputs.size(), labels)

                    # 用 Variable 包装输入数据
                    if use_gpu:
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    # 设置梯度参数为 0
                    optimizer.zero_grad()

                    # 正向传递
                    outputs = model(inputs)

                    if task == "classification":
                        _, preds = torch.max(outputs.data, 1)
                        running_corrects += torch.sum(preds == labels.data).cpu().numpy()
                    else:
                        outputs_numpy = outputs.cpu().data.numpy()
                        mul_value = np.dot(outputs_numpy, target.T)
                        denorm1 = np.reshape(np.linalg.norm(outputs_numpy, axis=1), (-1, 1))
                        denorm2 = np.reshape(np.linalg.norm(target, axis=1), (1, -1))
                        denorm = np.dot(denorm1, denorm2)
                        cos_value = mul_value/denorm
                        tmp = np.reshape(np.argmax(cos_value, axis=1), (1, -1))
                        true_labels = np.reshape(true_labels.numpy(), (1, -1))
                        running_corrects += np.sum(tmp == true_labels)

                        labels = labels.reshape_as(outputs)

                    loss = criterion(outputs, labels)

                    # 如果是训练阶段, 向后传递和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 统计
                    running_loss += loss.data[0] * inputs.size(0)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.astype(np.float) / len(dataloaders[phase].dataset)

                viz.line(X=torch.FloatTensor([epoch]), Y= torch.FloatTensor([epoch_loss]), win='loss',
                         update='append' if epoch > 0 else None)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 深拷贝 model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()
    except KeyboardInterrupt:

        torch.save(best_model_wts, "../models/{}.pkl".format(model_name))
        print("training phase stop at {} epochs, save model at models/{}.pkl".format(epoch, model_name))
        sys.exit(0)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型的权重
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "../models/{}.pkl".format(model_name))
    writer.close()
    return model
