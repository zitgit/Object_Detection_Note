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
INPUT：输入224*224*3的RGB图片
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
训练阶段：Resize 256*256 取224*224随机剪裁
测试阶段：Resize 水平翻转*10
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






 
