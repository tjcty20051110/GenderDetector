from data_loader import train_loader, val_loader
from model_setup import model, optimizer, loss_fn
import torch  # 无需再导入Variable（PyTorch 2.x已整合）
from loss_graph import plot_plot
max_acc = 0.0  # 初始化用浮点数更规范

#初始化四个列表，用于储存损失曲线绘图所需的数据
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(50):
    train_loss, train_count, train_correct_num = 0., 0., 0.
    val_loss, val_count, val_correct_num = 0., 0., 0.
    # 训练阶段
    model.train()  # 显式开启训练模式（对Dropout/BatchNorm等层必要）
    for data in train_loader:
        img, label = data
        # 直接将Tensor移到GPU，无需Variable包装（PyTorch 2.x推荐写法）
        img = img.cuda()
        label = label.cuda()

        output = model(img)
        optimizer.zero_grad()  # 清空梯度
        loss = loss_fn(output, label)
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        train_loss += loss.item()
        # 计算正确数量：先在GPU上求和，再移到CPU转换为Python整数
        pred = torch.max(output, dim=1)[1]  # 获取预测类别
        train_correct_num += (pred == label).sum().cpu().item()
        train_count += img.size(0)  # 累计样本数

    # 打印训练指标（确保除法结果为浮点数）
    train_acc = train_correct_num / train_count
    train_avg_loss = train_loss / train_count
    # 计算训练指标并存储
    train_loss_list.append(train_avg_loss)  # 保存当前轮训练损失
    train_acc_list.append(train_acc)        # 保存当前轮训练准确率
    print(f"epoch: {epoch}, train_clf_acc: {train_acc:.4f}, train_loss: {train_avg_loss:.4f}")

    # 验证阶段
    model.eval()  # 显式开启评估模式（固定Dropout/BatchNorm等层）
    with torch.no_grad():  # 关闭验证阶段的梯度计算，节省内存和计算资源
        for data in val_loader:
            img, label = data
            img = img.cuda()
            label = label.cuda()

            output = model(img)
            loss = loss_fn(output, label)

            val_loss += loss.item()
            pred = torch.max(output, dim=1)[1]
            val_correct_num += (pred == label).sum().cpu().item()
            val_count += img.size(0)

    # 打印验证指标
    val_acc = val_correct_num / val_count
    val_avg_loss = val_loss / val_count
    val_loss_list.append(val_avg_loss)  # 保存当前轮验证损失
    val_acc_list.append(val_acc)  # 保存当前轮验证准确率
    print(f"val_clf_acc: {val_acc:.4f}, val_loss: {val_avg_loss:.4f}")

    # 保存最优模型
    if val_acc > max_acc:
        max_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    print(f"max_acc: {max_acc:.4f}\n")

#打印损失曲线
plot_plot(train_loss_list, train_acc_list, val_loss_list, val_acc_list)

