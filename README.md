# ml_softmax
机器学习_softmax的相关分析

## 使用 Softmax 计算多分类的梯度计算示例

### 问题设定

假设模型的 logit 输出为 \( z = [z_1, z_2, z_3] = [1.0, 2.0, 3.0] \)，真实的标签是类别 2，对应的 one-hot 编码为 \( y = [0, 1, 0] \)。

### Softmax 函数计算

Softmax 函数定义如下：
\[ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{3} e^{z_j}} \]

计算每个 logit 的 Softmax 输出：

\[ e^{z_1} = e^1 = 2.718 \]
\[ e^{z_2} = e^2 = 7.389 \]
\[ e^{z_3} = e^3 = 20.085 \]

总和：
\[ \sum_{j=1}^{3} e^{z_j} = 2.718 + 7.389 + 20.085 = 30.192 \]

每个类的概率：
\[ \sigma(z)_1 = \frac{e^1}{30.192} = \frac{2.718}{30.192} \approx 0.090 \]
\[ \sigma(z)_2 = \frac{e^2}{30.192} = \frac{7.389}{30.192} \approx 0.245 \]
\[ \sigma(z)_3 = \frac{e^3}{30.192} = \frac{20.085}{30.192} \approx 0.665 \]

### 交叉熵损失函数

交叉熵损失函数定义为：
\[ L = -\sum_{i=1}^{3} y_i \log(\sigma(z)_i) \]

对于我们的例子，真实标签是类别 2，所以：
\[ L = -\log(\sigma(z)_2) = -\log(0.245) \approx 1.407 \]

### 梯度计算

为了计算梯度，我们首先计算损失函数 \( L \) 对 Softmax 输出 \( \sigma(z)_i \) 的偏导数：
\[ \frac{\partial L}{\partial \sigma(z)_i} = -\frac{y_i}{\sigma(z)_i} \]

因为 \( y = [0, 1, 0] \)，我们得到：
\[ \frac{\partial L}{\partial \sigma(z)_1} = 0 \]
\[ \frac{\partial L}{\partial \sigma(z)_2} = -\frac{1}{0.245} \approx -4.082 \]
\[ \frac{\partial L}{\partial \sigma(z)_3} = 0 \]

接下来，我们需要计算 Softmax 输出 \( \sigma(z)_i \) 对 logits \( z_j \) 的偏导数。

对于 \( i = j \)：
\[ \frac{\partial \sigma(z)_i}{\partial z_i} = \sigma(z)_i (1 - \sigma(z)_i) \]

对于 \( i \neq j \)：
\[ \frac{\partial \sigma(z)_i}{\partial z_j} = -\sigma(z)_i \sigma(z)_j \]

结合链式法则，我们计算每个 \( z_i \) 对损失 \( L \) 的梯度：
\[ \frac{\partial L}{\partial z_i} = \sum_{k=1}^{3} \frac{\partial L}{\partial \sigma(z)_k} \cdot \frac{\partial \sigma(z)_k}{\partial z_i} \]

具体计算如下：

#### 对于 \( z_1 \)：

\[ \frac{\partial \sigma(z)_1}{\partial z_1} = \sigma(z)_1 (1 - \sigma(z)_1) \approx 0.090 (1 - 0.090) = 0.090 \times 0.910 \approx 0.0819 \]
\[ \frac{\partial \sigma(z)_2}{\partial z_1} = -\sigma(z)_2 \sigma(z)_1 \approx -0.245 \times 0.090 = -0.02205 \]
\[ \frac{\partial \sigma(z)_3}{\partial z_1} = -\sigma(z)_3 \sigma(z)_1 \approx -0.665 \times 0.090 = -0.05985 \]

因此：
\[ \frac{\partial L}{\partial z_1} = 0 \times 0.0819 + (-4.082) \times (-0.02205) + 0 \times (-0.05985) \approx 0.090 \]

#### 对于 \( z_2 \)：

\[ \frac{\partial \sigma(z)_1}{\partial z_2} = -\sigma(z)_1 \sigma(z)_2 \approx -0.090 \times 0.245 = -0.02205 \]
\[ \frac{\partial \sigma(z)_2}{\partial z_2} = \sigma(z)_2 (1 - \sigma(z)_2) \approx 0.245 (1 - 0.245) = 0.245 \times 0.755 \approx 0.185 \]
\[ \frac{\partial \sigma(z)_3}{\partial z_2} = -\sigma(z)_3 \sigma(z)_2 \approx -0.665 \times 0.245 = -0.163 \]

因此：
\[ \frac{\partial L}{\partial z_2} = 0 \times (-0.02205) + (-4.082) \times 0.185 + 0 \times (-0.163) \approx -0.755 \]

#### 对于 \( z_3 \)：

\[ \frac{\partial \sigma(z)_1}{\partial z_3} = -\sigma(z)_1 \sigma(z)_3 \approx -0.090 \times 0.665 = -0.05985 \]
\[ \frac{\partial \sigma(z)_2}{\partial z_3} = -\sigma(z)_2 \sigma(z)_3 \approx -0.245 \times 0.665 = -0.163 \]
\[ \frac{\partial \sigma(z)_3}{\partial z_3} = \sigma(z)_3 (1 - \sigma(z)_3) \approx 0.665 (1 - 0.665) = 0.665 \times 0.335 \approx 0.223 \]

因此：
\[ \frac{\partial L}{\partial z_3} = 0 \times (-0.05985) + (-4.082) \times (-0.163) + 0 \times 0.223 \approx 0.665 \]

### 最终梯度结果

将计算结果总结如下：

\[ \frac{\partial L}{\partial z_1} \approx 0.090 \]
\[ \frac{\partial L}{\partial z_2} \approx -0.755 \]
\[ \frac{\partial L}{\partial z_3} \approx 0.665 \]

通过这些步骤，我们手动计算了交叉熵损失函数相对于 logits 的梯度。这个示例展示了如何将 Softmax 函数与交叉熵损失函数结合起来，计算多分类问题中的梯度。
