# FPN_Faster_rcnn


### 7月30日训练结果

|Models|mAP|scrap|fixedshape|reflect|scrach|
|-------------------|------|-------|-------------|---------|-------|
|Faster-RCNN-FPN+res101_v1|58.65%|78.27%|94.06%|60.21%|2.05%|
### 性能
在电脑上处理每张图片的时间，大约为0.8s左右，如下所示:
```
.../test_00132.jpg image cost 0.07894110679626465s    
.../test_00042.jpg image cost 0.08439826965332031s
.../test_00311.jpg image cost 0.07591986656188965s
.../test_00018.jpg image cost 0.07846617698669434s   
```
在docker上部署的tensorflow serving上进行测试时，从摄像头的帧率大概为每秒10帧。

## 环境
1、python3.5+ (anaconda recommend)                             
2、[opencv(cv2)](https://pypi.org/project/opencv-python/)   , [tfplot](https://github.com/wookayin/tensorflow-plot)             
3、tensorflow >= 1.10                   

## 预训练模型
[resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz) 放在下面的目录里

$PATH_ROOT/data/pretrained_weights.         

## 数据格式
```
├── VOCdevkit
│   ├── VOCdevkit_train
│       ├── Annotation
│       ├── JPEGImages
│   ├── VOCdevkit_test
│       ├── Annotation
│       ├── JPEGImages
```

## Compile
```  
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```




## Train

    
(1) 对于训练设置的更改可以去下面目录进行修改
```
$PATH_ROOT/libs/configs/cfgs.py
```
(2) 添加自己的数据集之后需要更改数据的种类以及对应关系。
```
$PATH_ROOT/libs/label_name_dict/lable_dict.py

```     
如果添加了自己数据作为新的数据名称的话(默认voc),同样需要更改下面目录的代码
 ```
 $PATH_ROOT/data/io/read_tfrecord.py 
 ```   

2、make tfrecord
```  
cd $PATH_ROOT/data/io/  
python convert_data_to_tfrecord.py --VOC_dir='/PATH/TO/VOCdevkit/VOCdevkit_train/' 
                                   
```     

3、train
```  
cd $PATH_ROOT/tools
python train.py
```


## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=.
``` 
## 生成计算图
训练完成之后得到的是**.ckpt**文件，需要把参数和图进行整合生成pb文件，首先生成的是用FrozenGraph 的API生成的计算图，以及计算图的节点文件。
运行代码之前需要更改 *CKPT_PATH*
```
cd $PATH_ROOT/tools/export_pbs
python exportPb.py
```
生成后的计算图可以直接部署到Android、iOS等移动设备上，而部署在tensorflow-serving上需要使用的是名为SavedModel的API，所以需要将上一步生成的pb文件进一步转换。
```
cd $PATH_ROOT/tools/export_pbs
python saved_model.py
```
至此，已经可以将文件部署到docker的tensorflow-serving上去了。
```
sudo docker run --runtime=nvidia -p 8500:8500 -p 8501:8501  --mount type=bind,source=/home/yimu/project/FPN_Tensorflow-master/output/Pbs/model/model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving:1.13.1

```
可以使用以下代码查看部署的模型输入和输出格式
```
 curl http://localhost:8501/v1/models/my_model/metadata
 ```
 后面直接使用客户端链接摄像头并观察瑕疵：
 ```
 cd $PATH_ROOT
 python client.py
 ```
 
