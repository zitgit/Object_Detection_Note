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

### Architecture
五个卷积+三个全连接层
INPUT：输入224*224*3的RGB图片
OUTPUT：100个类别
卷积输出特征图公式：Fo=[Fin - k(kernel) + 2p(padding)]/s(stride) + 1

conv1：ReLU
![](file:///E:/Machine_Vison/deepeye/03%20CV-baseline/%E5%AD%A6%E5%91%98%E7%94%A8%E8%B5%84%E6%96%99%E5%90%88%E9%9B%86/01Alexnet/alexnet.jpg)
