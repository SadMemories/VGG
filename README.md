# VGG
VGG 模型测试仓库
1. 初始化了VGG模型 在花分类数据集上操作
2. 准备向VGG模型中加入SE注意力机制，观察模型性能是否有提高

**训练时需要修改的路径：**
train.py: main 函数中的 data_root，将其改为自己数据集的路径

**花分类数据集的组织形式**
data/:  
\quad\quad train/:  
     daisy/:  
     dandelion/:  
     roses/:  
     sunflowers/:  
     tulips/:
   val/:  
     daisy/:  
     dandelion/:  
     roses/:  
     sunflowers/:  
     tulips/:  
上述的 / 表示文件夹，在最终的文件夹中存放着属于各类花的图片
