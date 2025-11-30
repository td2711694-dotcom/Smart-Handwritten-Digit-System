import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import tkinter as tk
from PIL import Image, ImageDraw
import os


# ================= 1. 定义卷积神经网络模型 (CNN) =================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：输入1通道(灰度)，输出10通道，卷积核5x5
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 第二层卷积：输入10通道，输出20通道，卷积核5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 池化层：2x2的最大池化
        self.pooling = nn.MaxPool2d(2)
        # 全连接层：将特征图展平后映射到10个分类
        self.fc = nn.Linear(320, 10)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        # 卷积 -> 激活 -> 池化
        x = self.pooling(self.relu(self.conv1(x)))
        x = self.pooling(self.relu(self.conv2(x)))
        # 展平
        x = x.view(batch_size, -1)
        # 全连接分类
        x = self.fc(x)
        return x


# ================= 2. 模型训练函数 =================
def train_model():
    print("正在准备数据和模型，请稍候（预计耗时 1-2 分钟）...")
    device = torch.device("cpu")  # 简单任务用CPU足够快，兼容性更好

    # 数据预处理：转为Tensor并归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 自动下载 MNIST 数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    # 训练 1 个 Epoch 即可达到演示效果 (95%+ 准确率)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'训练进度... Batch {batch_idx}/{len(train_loader)} \tLoss: {loss.item():.6f}')

    print("模型训练完成！正在启动手写识别系统...")
    return model


# ================= 3. 图形用户界面 (GUI) 类 =================
class App:
    def __init__(self, model):
        self.model = model
        self.model.eval()  # 切换到评估模式

        # 创建主窗口
        self.window = tk.Tk()
        self.window.title("基于CNN的手写数字智能识别系统")
        self.window.geometry("400x480")

        # 顶部提示标签
        self.label_hint = tk.Label(self.window, text="请在下方黑色区域手写数字 (0-9)", font=("微软雅黑", 12))
        self.label_hint.pack(pady=10)

        # 创建画布 (用于显示)
        self.canvas_width = 200
        self.canvas_height = 200
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        # 创建 PIL Image 对象 (用于模型输入)
        # 注意：这里创建纯黑背景的图像，模式'L'代表灰度
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # 绑定鼠标拖动事件
        self.canvas.bind("<B1-Motion>", self.paint)

        # 结果显示区域
        self.result_label = tk.Label(self.window, text="等待识别...", font=("微软雅黑", 20, "bold"), fg="#333")
        self.result_label.pack(pady=20)

        # 按钮区域
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=10)

        # 识别按钮
        tk.Button(btn_frame, text="立即识别", command=self.predict,
                  bg="#4CAF50", fg="white", font=("微软雅黑", 10), width=12, height=2).pack(side=tk.LEFT, padx=10)

        # 清除按钮
        tk.Button(btn_frame, text="清除画布", command=self.clear,
                  font=("微软雅黑", 10), width=12, height=2).pack(side=tk.LEFT, padx=10)

        self.window.mainloop()

    def paint(self, event):
        # 笔触半径
        r = 8
        x1, y1 = (event.x - r), (event.y - r)
        x2, y2 = (event.x + r), (event.y + r)
        # 在 GUI 画布上画 (视觉反馈)
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        # 在 PIL 图像上画 (数据记录) - 255 代表白色
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear(self):
        # 清除 GUI 画布
        self.canvas.delete("all")
        # 重置 PIL 图像
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="等待识别...", fg="#333")

    def predict(self):
        # 1. 调整图像大小：从 200x200 -> 28x28 (MNIST标准尺寸)
        img_resized = self.image.resize((28, 28))

        # 2. 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 增加 Batch 维度: [1, 28, 28] -> [1, 1, 28, 28]
        input_tensor = transform(img_resized).unsqueeze(0)

        # 3. 模型推理
        with torch.no_grad():
            output = self.model(input_tensor)
            # 获取概率最大的索引
            pred = output.argmax(dim=1, keepdim=True)
            confidence = torch.nn.functional.softmax(output, dim=1).max().item()

        # 4. 显示结果
        self.result_label.config(text=f"识别结果：{pred.item()}", fg="#E91E63")
        print(f"预测: {pred.item()}, 置信度: {confidence:.2f}")


# ================= 4. 主程序入口 =================
if __name__ == '__main__':
    # 检查是否安装了必要的库
    try:
        import torchvision
    except ImportError:
        print("错误：缺少必要的库。请在终端运行: pip install torch torchvision pillow")
        exit()

    # 第一步：获取训练好的模型
    trained_model = train_model()

    # 第二步：启动应用程序
    print("系统启动成功！")
    App(trained_model)