# Time-Series-Forecasting

本项目主要使用了[Acea Smart Water Analytics challenge](https://www.kaggle.com/c/acea-water-prediction/)的数据作为实验数据。

## 数据处理

在`1. Data Processing.ipynb`中，主要利用pandas进行数据处理（包括：对日期属性进行排序、填充缺失值等），重采样等操作，为后续实验做准备。另外，利用ADF检验，我们可以判断时间序列是否具有稳定趋势。

## 传统方法

在`2. Traditional Models.ipynb`中，利用处理好的数据，我们分别尝试了朴素方法（Naive Approach）、滑动平均（Moving Average）、ARIMA和VAR的方法对时间序列进行预测。在必要时，需要利用差分的方法让时间序列变得稳定，再进行预测。

## LSTM

在`3. LSTM.ipynb`中，我们首先基于pytorch框架构建自己的Dataset类，在后续的实验中，调用DataLoader类，可以很方便地获取批量化的数据。我们同样基于pytorch框架构建LSTM网络，使用单向的LSTM对时间序列进行特征提取，使用Linear层作为最后的输出层。

在训练时，使用`nn.MSELoss`即均方损失作为损失函数，使用`Adam`优化器。

## Transformer

在`4. Transformer.ipynb`中，我们手动构建了训练集和测试集的batch，并基于pytorch框架构建了Transformer网络，使用Transformer的encoder对时间序列进行特征提取，使用Linear层作为最后的输出层。

在训练时，使用`nn.MSELoss`即均方损失作为损失函数，使用`Adam`优化器。
