from model import *
from data import MyDataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import os, gc

if __name__ == '__main__':
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    # device = torch.device("cuda")
    # 清除gpu缓存
    gc.collect()
    torch.cuda.empty_cache()

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_datas = MyDataLoader(3)
    trainloader = all_datas.trainloader
    testloader = all_datas.testloader

    model = Model()
    model = model.to(device)

    # 损失函数
    loss_fun = torch.nn.CrossEntropyLoss()
    loss_fun = loss_fun.to(device)

    # 优化器
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    # 数据集大小
    train_data_size = len(trainloader) * 1
    test_data_size = 0
    if testloader != None:
        test_data_size = len(testloader) * 1
    print(train_data_size)
    print(test_data_size)
    # 训练参数
    epoch = 2
    # 训练和测试的步数，总训练和测试次数
    total_train_step = 0
    total_test_step = 0
    # 测试的损失和准确度
    total_train_loss = 0
    total_train_accuracy = 0
    total_test_loss = 0
    total_test_accuracy = 0
    best_test_accuracy = 0

    # 添加可视化tensorboard
    writer = SummaryWriter("Model_logs")
    writer.add_graph(model, torch.randn(1, 1, 105, 105).to(device))
    for current_epoch in range(epoch):
        print(f"第{current_epoch + 1}轮训练开始-------------------")
        # 训练
        model.train()
        for item in trainloader:
            data, targets = item

            data = data.to(device)
            targets = targets.to(device)

            output = model(data)
            loss = loss_fun(output, targets)
            # 优化模型参数
            optim.zero_grad()
            with torch.autograd.detect_anomaly():
                loss.backward()
            optim.step()

            total_train_loss += loss.item()
            total_train_step += 1
            accuracy = (output.argmax(1) == targets).sum()
            total_train_accuracy += accuracy
            # 打印相关内容，以便查看训练进度
            if total_train_step % 1000 == 0:
                print(f"训练次数: {total_train_step}, Loss: {loss}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                pass
        print(f"第{current_epoch + 1}轮,整体训练集上的Loss: {total_train_loss}")
        print(f"第{current_epoch + 1}轮,整体训练集上的正确率: {total_train_accuracy / train_data_size * 100} %")
        writer.add_scalar("train_accuracy_rate", total_train_accuracy / train_data_size, current_epoch + 1)
        total_train_accuracy = 0
        total_train_loss = 0

        # 测试
        model.eval()
        with torch.no_grad():
            for item in testloader:
                data, targets = item

                data = data.to(device)
                targets = targets.to(device)

                output = model(data)
                loss = loss_fun(output, targets.long())

                total_test_loss += loss.item()
                accuracy = (output.argmax(1) == targets).sum()
                total_test_accuracy += accuracy
                pass
            pass
        total_test_step += 1
        # 打印相关内容，以便查看训练效果
        print(f"第{current_epoch + 1}轮,整体测试集上的Loss: {total_test_loss}")
        print(f"第{current_epoch + 1}轮,整体测试集上的正确率: {total_test_accuracy / test_data_size * 100} %")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy_rate", total_test_accuracy / test_data_size, total_test_step)
        total_test_loss = 0

        # 保存模型
        torch.save(model,
                   f"Train/Model{(current_epoch + 1) % 11}.pth")
        print(f"模型已保存到,Train/Model{(current_epoch + 1) % 11}.pth--------------")
        if total_test_accuracy > best_test_accuracy:
            torch.save(model,
                       f"Best_Model.pth")
            print("模型已保存到Best_Model.pth--------------")
            best_test_accuracy = total_test_accuracy
            pass
        total_test_accuracy = 0
    writer.close()
pass
