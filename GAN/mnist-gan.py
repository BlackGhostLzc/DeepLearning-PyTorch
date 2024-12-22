import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

batch_size = 128

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])
# transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) # 3 for RGB channels
# MNIST dataset
mnist = torchvision.datasets.MNIST(root='data',
                                   train=True,
                                   transform=transform,
                                   download=True)
# Data loader.py
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leakyrelu(self.map1(x))
        x = self.leakyrelu(self.map2(x))
        x = self.sigmoid(self.map3(x))  # 最后生成的是概率
        return x


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 激活函数

    def forward(self, x):
        x = self.relu(self.map1(x))
        x = self.relu(self.map2(x))
        x = self.tanh(self.map3(x))
        return x


image_size = 784
hidden_size = 400
latent_size = 256
# ----------
# 初始化网络
# ----------
D = Discriminator(input_size=image_size,
                  hidden_size=hidden_size,
                  output_size=1)
G = Generator(input_size=latent_size,
              hidden_size=hidden_size,
              output_size=image_size)
# -----------------------
# 定义损失函数和优化器
# -----------------------
learning_rate = 0.0003
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
d_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=50, gamma=0.9)
g_exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=0.9)

num_epochs = 500


# 定义辅助函数
def denorm(x):
    """
    用来还原图片, 之前做过标准化
    """
    out = (x + 1) / 2
    return out.clamp(0, 1)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


real_path = os.path.join(os.getcwd(), 'real-img')
fake_path = os.path.join(os.getcwd(), 'fake-img')

total_step = len(data_loader)
print(total_step)
# ------------------
# 一开始学习率快一些
# ------------------
for epoch in range(250):
    d_exp_lr_scheduler.step()
    g_exp_lr_scheduler.step()
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1)
        # print(images.shape)
        if images.shape[0] != 128 or images.shape[1] != 784:
            break
        # 创造real label和fake label
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)
        # ---------------------
        # 开始训练discriminator
        # ---------------------
        # 首先计算真实的图片
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs  # 真实图片的分类结果, 越接近1越好
        # 接着使用生成器训练得到图片, 放入判别器
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs  # 错误图片的分类结果, 越接近0越好, 最后会趋于1, 生成器生成的判别器判断不了
        # 两个loss相加, 反向传播进行优化
        d_loss = d_loss_real + d_loss_fake
        g_optimizer.zero_grad()  # 两个优化器梯度都要清0
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        # 开始训练generator
        # -----------------
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        g_loss = criterion(outputs, real_labels)  # 希望生成器生成的图片判别器可以判别为真
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        if (i + 1) % 200 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}, d_lr={:.6f},g_lr={:.6f}'
                .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                        real_score.mean().item(), fake_score.mean().item(),
                        d_optimizer.param_groups[0]['lr'], g_optimizer.param_groups[0]['lr']))
        # Save real images
        if (epoch + 1) == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(real_path, 'real_images.png'))
    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(fake_path, 'fake_images-{}.png'.format(epoch + 1)))
# Save the model checkpoints
torch.save(G.state_dict(), './G.ckpt')
torch.save(D.state_dict(), './D.ckpt')


