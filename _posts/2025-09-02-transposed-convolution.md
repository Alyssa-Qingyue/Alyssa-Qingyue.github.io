---
layout: post
title: 简析resunet
subtitle: 草履虫都能看懂
gh-repo: Alyssa-Qingyue/Alyssa-Qingyue.github.io
gh-badge: [star, fork, follow]
tags: [transposed, convolution]
comments: true
mathjax: true
author: Alyssa
---
# ResNet

ResNet（Residual Network，残差网络）是由微软研究院提出的一种深度卷积神经网络结构，首次应用于 **ImageNet 2015** 图像识别竞赛中，取得了巨大的成功。其核心思想是通过 **残差结构（Residual Block）** 有效解决了深层网络中的梯度消失和梯度爆炸问题，从而使得网络能够训练到超过 **100 层** 甚至 **1000 层**。

---
<img src="https://github.com/user-attachments/assets/91cc8680-5f3e-4219-b813-83326e31281c" alt="流程图" style="width:800px; height:auto;" />


## 1. ResNet 的整体结构

下图展示了不同深度的 ResNet（18 层、34 层、50 层、101 层、152 层）的基本结构：

- **前置层（conv1）**: 7×7 卷积核，步长为 2，输出大小 112×112。
- **残差层 1（conv2.x）**: 多个 3×3 卷积堆叠，输出 56×56。
- **残差层 2（conv3.x）**: 通道数加倍，输出 28×28。
- **残差层 3（conv4.x）**: 通道数继续加倍，输出 14×14。
- **残差层 4（conv5.x）**: 最深的残差模块，输出 7×7。
- **分类层**: 全局平均池化 + 全连接层，输出类别概率。

> **FLOPs** 随着层数增加而快速增长，例如 ResNet-18 需要约 1.8×10⁹ FLOPs，而 ResNet-152 需要约 11.3×10⁹ FLOPs。

---

## 2. 卷积层输出计算公式

卷积层输出特征图大小由下式决定：

$$
OutputSize = \left\lfloor \frac{InputSize + 2 \times Padding - KernelSize}{Stride} \right\rfloor + 1
$$

例如：
- 输入大小 224×224
- 使用 7×7 卷积核，步长 2，填充 3

则：
$$
OutputSize = \frac{224 + 2 \times 3 - 7}{2} + 1 = 112
$$

---

## 3. 池化层

### 最大池化（Max Pooling）

公式：
$$
y_{i,j} = \max\{x_{p,q} | p \in R_i, q \in R_j\}
$$

### 平均池化（Average Pooling）

公式：
$$
y_{i,j} = \frac{1}{|R|} \sum_{p \in R_i, q \in R_j} x_{p,q}
$$

**示例**：

| 输入值 | Max Pooling 输出 | Average Pooling 输出 |
|--------|----------------|--------------------|
| 10, 25, 31, 43 | 43 | 27.25 |
| 54, 24, 20, 95 | 95 | 48.25 |

---

## 4. 残差结构（Residual Block）

核心思想是引入 **shortcut/skip connection**：

- 传统卷积层： $y = F(x)$
- 残差卷积层： $y = F(x) + x$

残差块由以下部分组成：
1. 卷积（Conv）
2. 批归一化（Batch Normalization）
3. 激活函数 ReLU
4. 残差连接（Shortcut）

这种设计能够避免梯度消失，使网络更深时依旧可以收敛。

---

## 5. ResNet34 的网络结构（示例）

```
以 **ResNet-34** 为例，其主要层次如下：
输入: 224×224 图像
↓
7×7 conv, 64, stride 2 → 输出 112×112
↓
3×3 max pool, stride 2 → 输出 56×56
↓
conv2.x → [3×3, 64] × 3 → 输出 56×56
↓
conv3.x → [3×3, 128] × 4 → 输出 28×28
↓
conv4.x → [3×3, 256] × 6 → 输出 14×14
↓
conv5.x → [3×3, 512] × 3 → 输出 7×7
↓
全局平均池化 → 1×1×512
↓
全连接层 → 1000 类别
↓
Softmax → 分类概率
```
---

## 6. 分类层与 Softmax

最终分类层通常为一个全连接层（fc 1000），对应 **ImageNet 1000 类别**。输出向量经过 **Softmax 函数** 转换为概率分布：

$$
P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
$$

示例输出：
类别 1 → 0.90
类别 2 → 0.05
类别 3 → 0.02
...
类别 1000 → 0.01

---
```
from torch import nn
import torch.nn.functional as F

#基本残差块实现，继承自nn的子类
class ResidualBlock(nn.Module):
  #shortcut跳连，none表示恒等映射
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        #主路径
        self.left = nn.Sequential(
            #下采样，后面有批归一化，卷积偏置可以省略
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            #第一个卷积输出做二维归一化，加速收敛稳定训练
            nn.BatchNorm2d(outchannel),
            #inplace=True，就地修改以节省内存
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        #快捷通道
        self.right = shortcut

    def forward(self, x):
        residual = x if self.right is None else self.right(x)
        #主路径与残差相加
        out = self.left(x) + residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        #前端预处理模块stem（大卷积+BN+ReLU+池化)
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            #二维批归一化,专门为2
d图像特征图设计，在每通道上面归一化（用于卷积层输出（N,C,H,W),普通BN1d用于全连接层的输出（N,D))
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(inchannel=64, outchannel=64, block_num=3, stride=1, is_shortcut=False)
        self.layer2 = self._make_layer(inchannel=64, outchannel=128, block_num=4, stride=2)
        self.layer3 = self._make_layer(inchannel=128, outchannel=256, block_num=6, stride=2)
        self.layer4 = self._make_layer(inchannel=256, outchannel=512, block_num=3, stride=2)
        #全连接层，将最后的通道维度映射到num_classes
        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride, is_shortcut=True):
        if is_shortcut:
            shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        else:
            #恒等映射
            shortcut = None

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))
        for _ in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)
  #resnet前向传播函数
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```
## 参考文献
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
2. [ResNet 原论文 PDF](https://arxiv.org/abs/1512.03385)

---
