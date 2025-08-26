# Unet学习笔记

## 摘 要
医学图像常常表现出病灶与周围组织之间对比度低且模糊的特征，即使在同
一种疾病中，病灶边缘和形状也存在较大差异，这给分割带来了巨大挑战。
因此，
精确分割病灶已成为评估患者病情和制定治疗方案的重要前提。
近年来， U-Net模
型在医学图像分割领域取得了显著成果，显著提升了分割性能，并被广泛应用于
语义分割任务，为一致性的定量病灶分析提供了有力的技术支持。
本文结合本科基础较弱学生的学习过程，提出了针对 U-Net的学习参考路径。
综合利用哔哩哔哩、 GitHub、知乎、arXiv等多种资源，对相关基础知识进行了较
为系统的整理与总结。
内容涵盖 U-Net的核心技术原理及优势、 U-Net++ 的主要特
性与改进方向；
在代码实战部分，展示了小规模医学细胞检测与目标检测任务的
实现过程，并探讨了在弱基础条件下，本科生如何有效应用所学知识解决基本科
研问题。
本研究为相关领域的研究人员提供了系统化的参考与借鉴， 并期待基于 U-Net
设计出更高效、更稳定的医学图像分割网络模型.<br>
关键词： 医疗，小目标检测， unet
## ABSTRACT
Medicalimagesoftenexhibitlowandblurredcontrastbetweenlesionsandsurround-
ing tissues, with substantial variation in lesion boundaries and shapes even within the
same disease, posing significant challenges for segmentation.
Therefore, precise lesion
segmentationhasbecomeacrucialprerequisiteforpatientconditionassessmentandtreat-
mentplanning.
Inrecentyears,theU-Netmodelhasachievedremarkablesuccessinthe
fieldofmedicalimagesegmentation,significantlyimprovingsegmentationperformance
and being extensively applied to semantic segmentation tasks, thereby providing robust
technicalsupportforconsistentquantitativelesionanalysis.
ThispaperproposesalearningroadmapforU-Nettailoredtoundergraduatestudents

with limited foundational knowledge.
By integrating resources from platforms such as
Bilibili,GitHub,Zhihu,andarXiv,wepresentasystematicsummaryofthefundamental
conceptsrelatedtoU-Net.
Thecontentcoversthecoretechnicalprinciplesandadvantages
of U-Net, as well as the key features and improvements of U-Net++.
In the practical
coding section, we demonstrate implementations for small-scale medical cell detection
andobjectdetectiontasks,anddiscussstrategiesforundergraduatestudentswithweaker
backgrounds to effectively apply acquired knowledge to address fundamental research
problems.
Thisworkprovidesasystematicreferenceforresearchersinrelatedfieldsandaims
to inspire the design of more efficient and stable medical image segmentation network
modelsbasedontheU-Netarchitecture.<br>
Key words: Medical；
objectdetection ；
unet
## 目录
1概述 ················································································· - 1<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.1核心技术栈简述 ······························································ 1<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.2优势（原始及发展版本） ··················································· - 2<br>
2模型架构：以医学细胞分析（小图像）为例 ·························· 3<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.1 unet 框架：实现基础 unet与unet++ ······································· 3<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.2 unet 核心部件 ································································ 4<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.3其他问题（纯基础可不看） ················································ - 4<br>
3数据处理与加载 ································································· 5<br>
&nbsp;&nbsp;&nbsp;&nbsp;3.1数据处理 ······································································ - 5<br>
&nbsp;&nbsp;&nbsp;&nbsp;3.2图像评估，损失函数 ························································ 5<br>
&nbsp;&nbsp;&nbsp;&nbsp;3.3其他基础 ······································································ - 6<br>
4代码主体 ··········································································· 7<br>
&nbsp;&nbsp;&nbsp;&nbsp;4.1 evaluate 工作流程 ···························································· 7<br>
&nbsp;&nbsp;&nbsp;&nbsp;4.2 predict 工作流程 ······························································ 7<br>
&nbsp;&nbsp;&nbsp;&nbsp;4.3蒟蒻的其他问题 ······························································ - 8<br>
5代码主体：  ····························································· 9<br>
&nbsp;&nbsp;&nbsp;&nbsp;5.
1准备数据 ······································································ - 9<br>
&nbsp;&nbsp;&nbsp;&nbsp;5.
2造Dataloder ··································································· 9<br>
&nbsp;&nbsp;&nbsp;&nbsp;5.
3配置优化器 /调度器/损失 ···················································· - 9<br>
&nbsp;&nbsp;&nbsp;&nbsp;5.
4训练循环（含混合精度、指标、可视化、验证、保存） ··············· 9<br>
&nbsp;&nbsp;&nbsp;&nbsp;5.
5其他问题 ······································································ - 10<br>

## 1概述
### 1.1核心技术栈简述
1.
核心：编码再解码。
先拼接特征，再加权采样，最后解码还原。
图1.
- 1 pipeline
2.
基本结构：初级版： （简单的特征融合与拼接）
图1.
- 2 basic
3.
升级结构U-net++（特征融合，能用全用，拼接全面，类比 densenet（cvpr- 2017
bestpaper ）
例如： 下采样卷积步长为 2， 升采样再加回去。
每一层与前面所有层都有联系。
deepsupervision 损失由多个位置计算，再更新
# - 1 图1.
3 upgrade
1.
2优势（原始及发展版本）
1.
overall：结构简单，速度快。
医学分割领域应用广泛
2.
优化版：更容易剪枝（每一层都有输出结果，且有单独监督训练：想图的最边
缘一斜列被去除了）
# - 2 2模型架构：以医学细胞分析（小图像）为例
源代码支持： https://github.
com/kuisu-GDUT/pytorch-cell-UNet.
git
## 2.
- 1 unet 框架：实现基础 unet与unet++
概述：定义两个网络类；
导入模块；
输入输出
重要函数：
1.
n_channels: 输入图像通道数（ e.
gRGB=3, 灰度=1）
2.
n_classes: 分割的类别数
3.
bilinear:上采样方式， true：双线性插值， false：转置卷积（转置卷积，就是普
通卷积的“逆操作” ，用来把小的特征图还原成更大的图像，常用于图像生成
和分割。
双线性插值：用周围 4个像素，按距离加权平均，来估算新像素值的
一种图像缩放方法 .
）
P(x,y)≈(x2−x)(y2−y)
(x2−x1)(y2−y1)Q11+(x−x1)(y2−y)
(x2−x1)(y2−y1)Q21+(x2−x)(y−y1)
(x2−x1)(y2−y1)Q12+(x−x1)(y−y1)
(x2−x1)(y2−y1)Q- 22


$$ (2.
1) $$


4.
deep_supervision: 是否使用深度监督（多层输出）
标准UNet类：
1.
编码器（下采样） （共四层） ： DoubleConv ：两次卷积 +BN+ReLU;Down ：最大
池化+DoubleConv
2.
解码器（上采样） （共四层） Up： 上采样+拼接（skipconnection ）+DoubleConv
3.
输出层：OutConv:1*- 1 卷积将通道数映射到类别数
4.
```
forward过程： 一路下采样特征， 再上采样并与编码层特征拼接， 最后输出 logits
（未经过 softmax/sigmoid ）
NestedUNet （也叫U-Net++）
1.
编码：和普通 unet类似
2.
解码：上采样＋多级跳越连接（第一层嵌套：用上采样深层特征来补充浅层
第二层：不仅用对称层，还用中间层第三，四层：加入更多节点，在最浅层融
合了来自不同深度和路径的信息）
3.
输出：多层输出，或者最后输出，看 deepsupervision
# - 3 ## 2.
2 unet 核心部件
DoubleConv ：两次卷积 +BN+ReLU
1.
两次卷积比一次卷积能提取更复杂的特征
2.
BN加快收敛、稳定训练
3.
RELU提供非线性能力
Down下采样
1.
Maxpool 减少尺寸
2.
DoubleConv: 提取下采样特征
UP上采样
1.
区分：双线性插值（ bilinear） ：轻量，但是通道数需要卷积调整
2.
反卷积（convtranspose2d(bilinear=false) 学习参数的上采样，计算更重）
3.
特别的：padding对齐+拼接上下采样：先计算上下 /左右差距；
用 F.
pad给小
特征图补 0，和大的对齐；
沿通道维度拼接
OutConv 输出，1*1卷积压缩到目标通道数
2.
3其他问题（纯基础可不看）
为什么最大池化有助于“看到更多特征” ？
1.
扩大感受野：从更大范围提取特征
2.
只保留最大值：保留最重要特征
3.
减少冗余，防止过拟合：参数变小；
保留关键特征，强泛化；
噪声不敏感
4.
不止输出层！输出层全局平均池化，或直接全连接层
# - 4 3数据处理与加载
3.
1数据处理
Pytorch自定义数据集类，加载图像和对应的分割掩码（常用于语义分割）读取图
像和mask（确保一一对应）
1.
basicdataset:transforms: 数据增强方法
2.
imagesd_ir 文件遍历去扩展名
缩放、归一化、格式转换
1.
转化为numpy数组格式
2.
缩放：image_nearest: 最邻近插值， 保持 mask离散值不被破坏；
image_bicubic:
双三次插值，适合普通图像
3.
通道调整：灰度图：扩展维度，变成 [1,H,W] ；
彩色图：转为 [C,H,W] 格式
4.
归一化【0，255】到【0，1】 、
（可选）数据增强输出 pytorch可直接使用的 image，masktensorps ：加了一个 car-
vanaDATASET 类专门处理车图分割吧啦吧啦
3.
2图像评估，损失函数
dice系数,[0,1]
1.
公式：$$Dice = 2|A∩B|$$
|A|+|B|
2.
遍历batch里面所有图并求平均
多类别（multiclass ）dice：多类别分割（遍历所有类别通道） dice loss=1-dice Lou
（交并比）
1.
公式：$$IoU = |A∩B|$$
|A∪B|
2.
sigmoid把模型输出转化为 0-1，阈值0.
5，转二值 mask
3.
intersection ：交集像素数；
union：并集像素数
4.
smooth：避免分母为 - 0
# 5 3.
3其他基础
什么是车图分割？
1.
其实就是一种图像分割 (ImageSegmentation) 任务， 目标是把图像中属于“车”
的像素和背景像素区分开来。
2.
CarvanaDataset 是对Carvana汽车数据集的封装，用来训练 U-Net之类的分割
模型
dice系数和lou系数的关系
1.
lou:关注交集在并集中的比例；
dice：强调交集与整体大小的相对关系
2.
Dice更敏感：当目标区域很小（例如医学影像中的小病灶） ， Dice对交集的比
例会更“放大” ，所以更能反映模型在小目标上的表现。
3.
IoU更严格：IoU要求交集在并集中的占比大，数值比 Dice更低，衡量标准
更“苛刻” 。
4.
总结：dice：训练损失， lou：评价指标
# - 6 4代码主体
函数签名与依赖
1.
net：分割模型（如 UNet/NestedUNet ） ，其forward返回通常是一个列表，这
里取最后一个输出。
2.
dataloader ：验证集的 DataLoader ，每个batch返回’image’: Tensor,’mask’: Ten-
sor。
3.
device：运行设备（ cuda或cpu） 。
4.
deep_supervision ：形参里有，但本函数里没有用到
## 4.
- 1 evaluate 工作流程
1.
建两个计量器，分别累计 Dice和IoU的加权平均。
2.
net.
eval() ：切换到评估模式（关闭 Dropout、BatchNorm ） 。
3.
记录验证批次数用于 tqdm进度条。
4.
遍历验证集， one_hot，注意参数顺序，二分类 nclassws 设为- 2
5.
no_grad下前向得到 mask_pred （取最后一个输出）
6.
二分类：sigmoid+0.
- 5 阈值；
多分类： argmax+one_hot
7.
计算Dice（多分类时忽略背景）与 IoU
8.
以batch大小为权重更新平均器
## 4.
- 2 predict 工作流程
其实是推理脚本，把训练好的 unet或者unet++应用到新图像上面 ,输出mask,
结果保存成图像或者可视化展示
1.
图像预处理： resize、归一化等。
unsqueeze 加batch维度
2.
前向推理
3.
概率图（多分类 softmax；
二分类 sigmoid）
4.
resize回原图大小，阈值化 one_hot
5.
getargs定义运行时参数；
输出文件名；
masktoimage （二分类 0，1灰白;多分
类argmax后按变好转灰度值）
# - 7 6.
主程序：eval(args.
model_name) ：字符串变为类对象（灵活选择模型） ；
加载
权重
7.
循环处理每张图片
4.
3蒟蒻的其他问题
one_hot
1.
把一个整数变成一个向量，这个向量的长度 =类别数，每个位置代表一个类
别，只有对应类别的那一位是 1，其余都是 0。
2.
语义分割里的 one_hot: 输入图像标签通常为 [N,H,W](n 张batch，h*w尺寸)。
onehot之后变成【 n,c,h,w】c=类别数
3.
意义：和模型输出对齐；
方便计算指标（如 dice，直接按通道做矩阵运算）
前向传播为什么不计算梯度
1.
不需要反向传播，只前向计算得到结果
2.
节省显存；
提速度
eval是Python内置函数，它能把字符串当作代码执行
# - 8 5代码主体： train.
py
5.
1准备数据
1.
数据增强：随机旋转 /翻转/对比度：提升泛化
2.
resize输入到input_h和w
5.
2造Dataloder
1.
构建数据集，切分
2.
特别的：droplast:丢弃最后一个不足 batch的批次
5.
3配置优化器 /调度器/损失
1.
本代码里面三种优化器任选（分割里 Adam/RMSprop 常见）
2.
调度器：ReduceLROnPlateau ： 指标 （这里用 dice） 提升停滞时降 LR。
；
CosineAn-
nealingLR ：余弦退火，常用于 warm-up/ 平滑收敛。
；
MultiStepLR ：在里程碑
epoch处乘以gamma衰减。
3.
AMP混合精度： autocast+GradScaler ，提速、降显存。
5.
4训练循环（含混合精度、指标、可视化、验证、保存）
1.
切换训练模式
2.
遍历batch
(a)qianxiang+deepsupervision （把每个输出的损失加权求和） +损失
(b)反向+更新（AMP）
(c)记录/进度条
(d)训练中期验证（用之前的 evaluate）+调度器step
(e)保存checkpoint
3.
命令行参数（模型与数据，深监督，输入大小等）
4.
main入口： 解析参数 →组装路径 （ dataset/imgs 、dataset/masks 、.
/checkpoints ） 。
；
生成config[’name’] 并创建checkpoints/name 目录；
把全部配置 yaml.
dump 存
到checkpoints/name/config.
yml 。
；
设日志、设备。
；
Model=eval(config[’arch’])
# - 9 动态选择 UNet或NestedUNet ，实例化：
5.
端到端数据流：图像 /掩码→dataset （增强，预处理） →dataloader→net.
train
循环：forward多可能输出 →CE+dice （ds加权求和） →AMP反传/更新→指
标统计/W&B→ 周期性evaluate→ 调度器→每epoch存checkpoint
5.
5其他问题
wandb：Weights&Biases
1.
实验管理与可视化平台
2.
用途：记录超参数（学习率、批大小、优化器等） ；
追踪指标（ loss,accuracy,
IoU,Dice.
.
.
） ；
可视化训练曲线（ loss曲线、lr曲线、指标变化趋势） ；
存储和
管理模型权重；
团队协作（多人实验对比）
为什么构建数据集要优先使用车图分割数据？
1.
规模大；
质量高；
多样性强；
实际价值高
2.
相比一些小众医疗分割 /自然分割数据，车图分割数据泛化性更强，优先级高
shuffle=True ： 每个epoch开始时， 随机打乱数据顺序 →避免模型记住样本顺序， 减
少过拟合。
shuffle=False ：数据固定顺序，一般用于验证集 /测试集（保证结果可重
复） 。
超参数要记录哪些，分别有什么用
1.
学习率（lr） ：控制每次参数更新的步长
2.
batchsize ：影响显存占用、收敛速度
3.
优化器类型（ Adam,SGD,RMSprop 等）
4.
调度器策略（ CosineAnnealing,StepLR …）
5.
epoch数量：训练轮次
6.
loss函数类型（ CE,Dice,BCEWithLogitsLoss 等）
7.
数据增强方法（翻转、旋转、裁剪）
8.
模型结构参数（深度监督、网络层数、是否使用 BN等）
分割模型中三种常见的优化器；
调度器 :主包还在学，过段时间出续集吧
10

