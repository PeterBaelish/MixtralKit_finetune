import torch
import torch.nn as nn
import torch.optim as optim

# 定义网络结构
class TwoLayerNet(nn.Module):
    def __init__(self, dim, D, hidden_dim):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(dim, D, bias=False)
        self.linear2 = nn.Linear(D, hidden_dim, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)  # 使用ReLU激活函数
        x = self.linear2(x)
        x = torch.sigmoid(x)  # 使用Sigmoid将输出限制在[0, 1]之间
        return x.round()  # 将Sigmoid输出四舍五入到最近的整数（0或1）

# 网络参数
dim = 10  # 输入维度
D = 20    # 第一层的大小
hidden_dim = 5  # 输出维度

# 创建网络实例
net = TwoLayerNet(dim, D, hidden_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 由于输出是二进制的，使用二元交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练数据准备
# 假设我们有一些随机生成的训练数据
x_train = torch.randn(100, dim)  # 100个训练样本
y_train = torch.randint(0, 2, (100, hidden_dim)).float()  # 100个随机的二进制标签

# 训练网络
epochs = 10
for epoch in range(epochs):
    for i in range(len(x_train)):
        # 前向传播
        outputs = net(x_train[i])
        loss = criterion(outputs, y_train[i])

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
