---
layout: post
title: 简析转置卷积
subtitle: 草履虫都能看懂
gh-repo: Alyssa-Qingyue/Alyssa-Qingyue.github.io
gh-badge: [star, fork, follow]
tags: [transposed, convolution]
comments: true
mathjax: true
author: Alyssa
---
# 转置卷积

## 转置卷积的运算步骤
1.在输入特征图元素间填充s-1行、列0<br>
2.在输入特征图四周填充k-p-1行、列0<br>
3.将卷积核参数上下、左右翻转<br>
4.做正常卷积运算（p=0，s=1）<br>

$$
H_{out} = (H_{in} - 1) \times stride[0] - 2 \times padding[0] + kernel\_size[0]
$$

$$
W_{out} = (W_{in} - 1) \times stride[1] - 2 \times padding[1] + kernel\_size[1]
$$
<img width="850" height="890" alt="image" src="https://github.com/user-attachments/assets/f9734b73-d6e9-42c4-b02e-aa8ee9d2c839" />


