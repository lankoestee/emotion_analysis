# TextCNN情感分析实验

![Huawei](https://img.shields.io/badge/Huawei-%23FF0000.svg?style=for-the-badge&logo=huawei&logoColor=white)![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

本仓库是基于华为有限公司基座课程[《自然语言处理》——情感分析](https://connect.huaweicloud.com/courses/learn/course-v1:HuaweiX+CBUCNXA028+Self-paced/about)的Tensorflow以及PyTorch实现版本。

## 实现功能

通过Tensorflow以及PyTorch框架实现了基于TextCNN实现简单的二分类情感分析，其测试集准确率与指导书中大致一致，能够达到91%左右。

此外，在PyTorch版本中，我们设计了4个不同的参数和模型文件以适合不同的任务的任务需求，且拥有不同的参数大小。

|  类型  |    平台    |  参数量  | 模型大小 | 测试集准确率 |   链接   |
| :----: | :--------: | :------: | :------: | :----------: | :------: |
| 原模型 | Tensorflow | 11038622 |   44MB   |    0.894     |          |
|   XL   |  PyTorch   | 3754934  |   15MB   |    0.903     |          |
|   L    |  PyTorch   | 1910684  |  7.6MB   |    0.891     |          |
|   M    |  PyTorch   |  416212  |  1.7MB   |    0.877     | 仓库内含 |
|   S    |  PyTorch   |  117366  |  469KB   |    0.853     | 仓库内含 |

## 快速使用

均在`tensorflow.ipynb`和`pytorch.ipynb`中有较为详细的说明，均支持重新训练和根据现有模型进行快速推理。

## 模型概览

本模型使用[ChnSentiCorp](https://www.sciencedirect.com/science/article/abs/pii/S0957417407001534)数据集进行训练和测试，通过[jieba](https://github.com/fxsjy/jieba)进行分词后将句子按照词数拓展或裁剪到指定长度。通过词向量嵌入后通过并行的多个CNN卷积层，获得不同的感受野。最后通过全连接层进行结果的输出。

![TextCNN](figure/model.svg)
