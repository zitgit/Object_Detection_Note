# AlexNet

## tricks
### ReLU Nolinearity
1. 训练速度上：tanh(x)和Sigmoid比较慢，ReLU达到低LOSS时的Epoch比较少
2. 防止梯度消失：Sigmoid右端变化缓慢，导数趋近于零，容易出现梯度消失，ReLU右端求导不为零
3. Relu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生

### Local Response Normolization
在第1，2卷积层后
### Overlapping Pooling
与LRN层同时存在，在第五个卷积层存在

池化技术的本质：

在尽可能保留图片空间信息的前提下，降低图片的尺寸，增大卷积核感受野，提取高层特征，同时减少网络参数量，预防过拟合。

### Dropout
随机失活：部署在FC层的第一二层，增加模型鲁棒性

dropout probability = 0.5

训练集尺度问题

### Architecture
五个卷积+三个全连接层

INPUT：输入224×224×3的RGB图片

OUTPUT：100个类别

卷积输出特征图公式：Fo=[Fin - k(kernel) + 2p(padding)]/s(stride) + 1

conv1：ReLU + Pool + LRN

conv2：ReLU + Pool + LRN

conv3：ReLU

conv4：ReLU

conv5：ReLU + Pool

具体操作：
https://www.cnblogs.com/alexanderkun/p/6917984.html

### Data Augmentation
训练阶段：Resize 256×256 取224×224随机剪裁

测试阶段：对测试集图像Resize 水平翻转×10

PCA方法修改通道的像素值，实现色彩扰动，提高了1个acc


## Results
ImageNet预训练模型 + 5 个CNN做平均

第一个卷积核的可视化：卷积核呈现不同的频率，方向，颜色。两个GPU分工学习。

特征由低级到高级。越深层次的卷积核越抽象。

GPU1：频率，方向的卷积核

GPU2：颜色

图像检索：相似图片的第二个全连接层输出特征向量的欧氏距离相近：4096维的特征


## Conclusion
•
深度与宽度可决定网络能力
Their capacity can be controlled by varying their depth and breadth.
1 Introduction p2
•
更强大 GPU 及更多数据可进一步提高模型性能
All of our experiments suggest that our results can be improved simply by waiting for faster GPUs and bigger
datasets to become available. 1 Introduction p5
•
图片缩放细节，对短边先缩放
Given a rectangular image, we first rescaled the image such that the shorter side was of length 256, and then
cropped out the central 256 × 256 patch from the resulting image.(2 Dataset p3)
•
ReLU 不需要对输入进行标准化来防止饱和现象，即说明 sigmoid/tanh 激活函数有必要对输入进行标准化
ReLUs
have the desirable property that they do not require input normalization to prevent them from
saturating(3.3 LRN p1)
•
卷积核学习到频率、方向和颜色特征
The network has learned a variety of frequency
and orientation selective kernels, as well as various colored
blobs.(6.1 p1)
•
相似图片具有“相近”的高级特征
If two images produce feature activation vectors with a small Euclidean separation, we can say that the higher levels
of the neural network consider them to be similar.(6.1 p3)
•
图像检索可基于高级特征，效果应该优于基于原始图像
This should produce a much better image retrieval method than applying autoencoders to the raw pixels.(6.1 p4)
•
网络结构具有相关性，不可轻易移除某一层
It is notable that our network’s performance degrades if a single convolutional layer is removed.(7 Discussion p1)
•
采用视频数据，可能有新突破
Ultimately we would like to use very large and deep convolutional nets on video sequences.(7 Discussion p2)


# VGG

## Architecture of VGG

VGG 11 到 19的变化过程
11 layers: conv3-64; maxpool; conv3-128; maxpool; conv3-256, conv3-256; maxpool; conv3-512, conv3-512; maxpool; conv3-512, conv3-512; maxpool; FC-4096; FC-4096; FC-1000
首先，从11层开始，都是由64卷积核上升至512个卷积核，卷积层数不断增加，在VGG16中做过在三个深层卷积网络：256，512，512中增加1×1卷积层的改进，目的是为了增加网络非线性映射能力，发现效果有所提升，最终改为3×3卷积尺寸，目的是为了增大感受野，最终得到了 22444+3的VGG19结构，卷积核依旧从64增加至512。

经过5个maxpooling的池化操作，224的图像变成最后7×7的尺寸。Feature Map的channel数翻倍直至512。3个FC进行分类输出。

### Memory and Params
INPUT: [224×224×3]; m: 224×224×3=150k; params: 0
CONV3-64: [224×224×64]; m: 224×224×64=3.2M; params:(3×3×3)×64=1728

内存计算：feature map的尺寸（size×channel）；参数计算：kernel size × 输入channel × 卷积核数目； 池化层没有参数

### Feature

1. 两个3×3卷积核等价于一个5×5卷积核，可以增大感受野的同时，增加特征抽象能力；然而参数数目更少


## Training Tricks
超参设置：BS=256; Momentum=0.9; L2=5×e-4; LR=0.01;
### Data Augmentation
训练过程中等比例缩放的最短边为S，S不小于224×224，S设定的方法为 要么固定为256/384，要么在[256, 512]中 随机取值

针对位置：随机剪裁出224×224的图像出来   

针对颜色：RGB三通道颜色扰动 

### Test time augmentation 多尺度测试
多尺度测试，引入超参Q

Q的取值有三种方式：

Q=[S-32, S, S+32](S为固定值)

Q=[S_min, 0.5×(S_min+S_max), S_max]


一张图片要截取224×224的图片，将大于224的部分分四个步长进行位移，水平和垂直都可以位移五次所以是5×5=25然后flip，每个scale的Q可以做50crops（5×5 regular grid with flip），三个scale做150次crops

### Dense Test 稠密测试
将FC层转换为卷积操作，变为全卷积网络，实现任意尺寸图片输入。

第一个FC层为 FC-4096; 输入为7×7×512的Feature Map；将他拉成一维：（1， 7×7×512），将他与FC全连接，向量参数量为7×7×512×4096; 输出向量的尺寸变为4096（个神经元）；

若第一个为7×7的卷积层；输入为7×7×512的Feature Map，不用拉成向量，要求输出为4096个神经元，卷积层就需要使用7×7的kernel size，4096的卷积核，就能得到4096个神经元作为输出（1×1×4096的FM）；

然后用1×1×4096的卷积层，同理最后一层卷积层就用1×1×1000的卷积层最终得到1000个神经元。

好处是：倘若图片输入不是224而是448，进入FC层会报错（14×14×512），但是三个卷积层不会报错


## Results
1. 误差随着深度加深而降低，19层误差饱和，不再下降
2. 增加1×1的卷积层有助于性能提升
3. 训练时加入尺度扰动有助于性能提升
4. 3×3 kernel size比5×5优秀
5. 在Test时使用Scale Jittering 且使用随机S值（导致随机Q值）有助于性能提升
6. 除了Multi-scale外，使用Multi-crop（224×224滑动三个scale的Q再flip得到150crops）和Dense（卷积层代替全连接层）结合可以提升精度

## Summary
• 堆叠小卷积核，加深网络，获得高精度

• 训练阶段，尺度扰动，获得高精度

• 测试阶段，多尺度及Dense+Multi crop，获得高精度


# Resnet

## Residual Block
残差block解决的问题：网络退化的现象：增加网络深度使训练集loss增大。（并非过拟合）

残差结构可以把低层的特征传到高层，以同等映射的方式学习低层特征，这样效果至少不会比浅层的网络效果差

## Architecture
1. 头部使用 stride = 2的卷积层迅速降低分辨率
2. 4阶段残差结构堆叠，深层次Resnet只是改变Building Block的堆叠层数；特别是14×14的FM阶段
3. 池化average pool和FC-1000作为输出
4. Bottleneck：第一个1×1下降1/4通道数，第二个1×1提升4倍通道数

## Training Tricks

Warmup：一个epoch（迭代400次）


# ResNeXT
## Architecture





























 
