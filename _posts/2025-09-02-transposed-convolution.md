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
<img src="https://github.com/user-attachments/assets/91cc8680-5f3e-4219-b813-83326e31281c" alt="流程图" style="width:600px; height:auto;" />


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

## 参考文献
1. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
2. [ResNet 原论文 PDF](https://arxiv.org/abs/1512.03385)

---
