# 进度提示（标志：仓库程序文件还没有完善好）
- [x] 完成README-工程代码部分：整体代码提交与测试   
　　2018.6.6 完成代码提交     
　　2018.6.6 完成代码测试：[6步即可训练与测试](#程序运行)         
- [x] 完成README-Faster模型详解部分－2018.6.7
- [ ] 完成README-Pytorch使用部分
# 工程代码    
这个程序是基于Pytorch实现Faster-RCNN功能。    
参考代码链接：https://github.com/jwyang/faster-rcnn.pytorch.git    
参考代码特点：代码健壮，功能齐全，使用方便，过于庞大不方便阅读学习   
本代码目的：方便学习faster-rcnn细节，如果项目应用还是用参考代码比较好   
本代码特点：在保证基础功能的前提下，对数据处理部分进行整理，对模型部分进行注释    

### 开发环境  
Ubuntu16.04（i5-7500 + GTX 1070Ti ） + python3.5 + Pytorch0.3.0    

### 文件夹说明
1、Data：   
　　　picture_data/Annotations--存放图片标注的xml文件（手动存放）   
　　　picture_data/Images --存放用于训练与测试的图片（手动存放）   
　　　picture_data/cache --存放处理xml文件之后形成图片label信息文件（程序执行）   
　　　train.txt --存放训练图片序号（手动存放）  
　　　pretrained_model --用于存放VGG的预训练模型  
2、Output --存放训练完成的模型（程序执行）   
3、demo/images  -- 存放测试图片  
4、demo/result  -- 存放测试结果  
5、lib -- 模型各个部分的程序文件  

### 程序运行   
1.6步快速验证程序是否可行：   
　由于之前的反馈许多同学是在验证程序可行之后才开始读代码，可是准备数据集又不太方便   
　因此我提前做了一些准备工作方便同学快速验证程序的可行性，步骤如下：   
　1.下载代码到本地： git clone https://github.com/Liu-Yicheng/Faster-rcnn.git      
　2.安装程序需要的包：pip install -r requirements.txt      
　3.进入lib文件夹 ：cd lib     
　　编译程序需要用到的组件如nms与roi-pooling：sh make.sh    
　4.在下面的链接中下载VGG预训练模型，放入Data/pretrained_model文件夹   
　5.直接可以运行trainval.py（训练500次之后结束，loss值一般为0.1-0.15）   
　6.训练结束后直接可运行demo.py，在demo/result文件夹下可以看到它的预测结果   
2.测试结果图如下：

|class|xj|xjyz|jyz|tgsr|xlxj|dwq|xzsr|cls|mAP|
|---|---|---|---|---|---|---|---|---|---
|AP|0.44|0.87|0.78|0.71|0.87|0.68|0.78|0.65|0.738

　![result](https://github.com/Liu-Yicheng/Faster-rcnn/raw/master/Output/result.png)   
 
3.程序使用具体流程：---------------------训练过程-------------------------  
　　　　　　　　A.将需要训练与测试的图片放入Data/picture_data/Images文件夹   
　　　　　　　　　将XML文件放入Data/picture_data/Annotation文件夹     
　　　　　　　　（当数据集改变，需要删除Data/picture_data/cache里面的文件）     
　　　　　　　　B.训练图片的编号写入train.txt  
　　　　　　　　C.下载VGG预训练模型，放入Data/pretrained_model文件夹。  
　　　　　　　　　权重下载地址：[Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0)　　[VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth)  
　　　　　　　　　　　　　　　　[百度云](https://pan.baidu.com/s/1nHezTm6xKXjHYZXKHAl3KQ) 密码：or0m  
　　　　　　　　D.针对自己的项目修改traincal.py文件中的参数(如果用命令行的就输入参数)    
　　　　　　　　E.修改Data/pascal.py的141行，类别为用户的数据集类别           
　　　　　　　　F.运行trainval.py 开始训练         
　　　　　　　　----------------------测试过程---------------------------  
　　　　　　　　A.修改demo中的139行改为你训练之后的模型名称  
　　　　　　　　B.将测试图片放入demo/images文件夹  
　　　　　　　　C.运行demo.py进行测试，结果在demo/result中查看  
3.本代码方便学习faster内部细节与整个流程，但是用于项目中还是用参考代码  
  参考代码实现了多个模型与roi-crop，并且有计算map的函数等等，比较完备   

# Faster-Rcnn模型详解   
![picture1](https://github.com/Liu-Yicheng/Faster-rcnn/raw/master/Output/Faster.jpg)  
faster组件说明:   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿    
1.VGG16convrelu1--convrelu5(lib/model/faster_rcnn/vgg16.py):   
　　输入：图片（input_image）  　
　　过程：图片输入多个卷积层池化层  
　　输出：图片的特征图（H*W*512）   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   
2.generate_anchor(lib/model/rpn/generate_anchor.py)：   
　　输入：默认参数(base_size = 16, ratios =[0.5, 1, 2],scales = [8,16,32])   
　　过程：step1.构造一个参考ref_anchor[0,0,15,15]   
　　　　　step2.保持ref_anchor的中心点不变，将ref_anchor的面积按[0.5，1，2]缩放   
　　　　　　　　得到3种anchor：[-3.5,2,18.5,13.][0,0,15,15,][2.5,-3,12.5,18]   
　　　　　step3.将step2得到的3种anchor保持中心点不变，各自的边长缩放[8,16,32]倍   
　　　　　　　　如此得到9种anchor。   
　　输出：_anchors：9个中心点一致，面积不相同的anchor   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   
3.anchor_target_layer(lib/model/rpn/anchor_target_layer.py)：   
　　输入：feature_map（H*W*512），9种_anchor   
　　过程：step1.根据feature_map的面积(H*W),以上面每一个点为anchor中心，产生9个anchor   
　　　　　　　　最后得到H*W*9个anchor，这些anchor的位置是在输入图像上的位置   
　　　　　step2.在这些anchor中，只保留4个顶点都在图像内的anchor   
　　输出：anchors：H*W*9个anchor - 在图像外部的anchor   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   
4.bbox_overlaps_batch(lib/model/rpn/bbox_transfrom.py):      
　　输入：3输出的anchor，gt_boxes    
　　过程：生成anchor与gt_boxes的iou矩阵overlaps。   
　　　　　第i行第j列为第i个anchor与第j个gt_boxes的IOU值      
　　输出：overlaps   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   
overlaps->label(lib/model/rpn/anchor_target_layer.py):   
　　输入：overlaps  
　　过程：这个主要是通过anchor与gt的IOU矩阵确定anchor是正样本还是负样本。  
　　　　　step1.符合两类为正样本：若一个anchor与某个gt的IOU>0.7，为正样本。   
　　　　　　　　　　　　　　　　　与对gt来说，与它IOU最大的anchor为正样本。   
　　　　　step2.一个anchor与所有的gt的IOU<0.3,则这个anchor为负样本。  
　　　　　step3.统计前面的正样本数量与负样本。若正样本数量>128个，超出的部分  
　　　　　　　　则变为非正非负样本，不参与训练，剩下的负样本控制在128个。若正  
　　　　　　　　样本数量<128个，则选负样本使得正样本与负样本总和为256。  
　　　　　　　　最后得到所有的anchor的labels值  
　　输出：labels（代表生成的anchor里有无目标物体，分数为0，1，-1：1代表有物体，  
　　　　　0代表没有物体，-1代表不参与训练）  
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   	  
5.RPN_conv1(lib/model/rpn/rpn.py)：  
　　输入：feature_map（H*W*512）  
　　过程：将feature_map输入到卷积层（512个卷积核，每个卷积核大小为3，步长与填充都为1）  
　　输出：特征图rpn_conv1  
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿   
6.RPN_cls_score(lib/model/rpn/rpn.py):   
　　输入：特征图rpn_conv1  
　　过程：将rpn_conv1输入至卷积层（H*W*9个卷积核，卷积核大小为1，边长为1，填充为0）  
　　　　　这个卷积核边长为1的卷积层其实相当于全连接层，最后的输出为维度为[H*W*9]的向量  
　　　　　,每个元素代表这个anchor框内有无目标物体，与Fast-Rcnn部分不同的是，它只在乎这个  
　　　　　anchor中有没有目标物体，而不在乎这个物体属于哪一类。  
　　输出：rpn_cls_loss（代表rpn网络预测每个anchor里有目标物体的分数）  
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿    
7.Cross_entropy:  
　　输入：rpn_cls_score--(rpn网络预测由特征图每个anchor里有目标物体的分数)  
　　　　　labels--（代表特征图生成的anchor里的真正分数(有无目标物体)，分数  
　　　　　　　　　　为0，1，-1：1代表有物体，0代表没有物体，-1代表不参与训练）    
　　过程：rpn_cls_score是anchor有无物体的预测值，label是anchor有无物体的真值，两者利用交  
　　　　　叉熵损失函数计算损失值，在不断反向传播时，ron_cls_score会越来越向label值靠近，   
　　　　　这代表网络判断得越来越准。   
　　输出：rpn_box_loss   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿    
8.compute_target_batch(lib/model/rpn/bbox_transfrom.py):    
　　输入：3输出的anchor，gt_boxes(假设anchor数量为N，gt_boxes数量为M)   
　　过程：生成[N, M, 4]矩阵bbox_target.[i,j,4]代表第i个anchor的4个参数与第j个gt_boxes   
　　　　　的4个参数的真正偏移量。   
　　输出：bbox_target   
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿    
9.RPN_bbox_pred(lib/model/rpn/rpn.py):   
　　输入：特征图rpn_conv1    
　　过程：将rpn_conv1输入至卷积层（H*W*9*4个卷积核，卷积核大小为1，边长为1，填充为0）   
　　　　　这个卷积核边长为1的卷积层其实相当于全连接层，最后的输出为维度为[H*W*9*4]的向量   
　　　　　,代表特征图生成的H*W*9个anchor，每个anchor的参数为4个，分别是该anchor距离gt框的    
　　　　　中心点与长宽的预测偏移量。    
　　输出：rpn_bbox_pred（代表rpn网络预测每个anchor距离它最近的gt框的预测偏移量）    
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿    
10.smooth_l1_loss(lib/model/rpn/rpn.py):    
　　输入：rpn_bbox_pred--(rpn网络预测由特征图每个anchor里距离gt框的预测偏移量)    
　　　　　bbox_target--(特征图的每个anchor距离gt框的真正偏移量)    
　　过程：利用smooth_l1损失函数计算预测偏移量与真正偏移量的距离，经过不断反向传播，使得    
　　　　　预测偏移量不断靠近真正偏移量。这就代表网络预测出的anchor值愈来愈靠近gt框。     
　　　　　综合cross_entropy与smooth_l1_loss，使得rpn网络预测出的anchor值越来越靠近gt框     
　　　　　这为后面faster的预测打下了坚实的基础。     
　　输出：rpn_box_loss    
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿     
anchors+rpn_bbox_pred+rpn_cls_score ->rpn_output_rois(lib/model/rpn/proposal_layer.py)：     
　　输入：anchors：特征图产生的(H*W*9)个anchor     
　　　　　rpn_bbox_pred：rpn网络预测每个anchor距离它最近的gt框的预测偏移量    
　　　　　rpn_cls_score：rpn网络预测每个anchor内是否有物体的分数   
　　过程：将每个anchor的4个参数(中心点x,y,长,宽)与rpn_bbox_pred中对应的anchor偏移量进行运算，        
　　　　　得到rpn网络预测每个anchor的最终位置rpn_output_rois。    
　　　　　根据 rpn_cls_score分数从大到校将anchor进行排列，选取12000个候选框（测试时6000个）  
　　　　　将选取出来的12000个候选框先通过nms得到nms。    
　　输出：rpn_output_rois（训练时2000个，测试时300个）       
＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿         
此后进入Fast-Rcnn部分。与RPN部分相比，思路上大体是差不多的。差别在：       
1.输入的anchor上，rpn网络是人工产生的anchor而Fast-Rcnn部分是由rpn网络产生的rois作为anchor。       
2.在RCNN_cls_score上，rpn只在乎anchor里有没有目标物体，而Fast-RCNN部分再要对anchor中的物体属于        
　哪一类进行分类。     

# Pytorch使用  
### 构造模型  
1.调用模型  
　　ourmodel = models.alexnet()   
　　注：在torchvision.models中定义了vgg、alexnet、densenet、inception、resnet、squeezenet等常用的网络。   
　　　　我们使用调用语句就可以将定义的网络赋值给vgg。   
   
2.加载预训练模型  
　　static_dict = torch.load(pretrained_model)      
　　ourmodel.load_state_dict({k:v for k,v in static_dict.items() if k in ourmodel.state_dict()})    
　　注：pretrained_model是预训练好的模型   
　　　　模型的static_dict是一个字典，字典的key是模型层的变量名比如：features.0.weight   
　　　　代表着features块的第一层的weight；字典的value就是变量值。加载预训练模型必须自己的模型与   
　　　　预训练的模型的定义是一致的   

3.了解模型构造   
　　ourmodel = models.alexnet()   
　　print(ourmodel._modules)   
　　-----------------------result---------------------------------------------    
　　OrderedDict(   
　　[  ('features',    
　　　　　Sequential((0): Conv2d (3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))   
　　　　　　　　　　　(1): ReLU(inplace)   
　　　　　　　　　　　(2): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))   
　　　　　　　　　　　(3): Conv2d (64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))   
　　　　　　　　　　　(4): ReLU(inplace)      
　　　　　　　　　　　(5): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))    
　　　　　　　　　　　(6): Conv2d (192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    
　　　　　　　　　　　(7): ReLU(inplace)     
　　　　　　　　　　　(8): Conv2d (384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    
　　　　　　　　　　　(9): ReLU(inplace)     
　　　　　　　　　　　(10): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))    
　　　　　　　　　　　(11): ReLU(inplace)    
　　　　　　　　　　　(12): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1))    
　　　　　　　　　　)),     
　　　　('classifier',     
　　　　　Sequential((0): Dropout(p=0.5)    
　　　　　　　　　　　(1): Linear(in_features=9216, out_features=4096)    
　　　　　　　　　　　(2): ReLU(inplace)     
　　　　　　　　　　　(3): Dropout(p=0.5)    
　　　　　　　　　　　(4): Linear(in_features=4096, out_features=4096)     
　　　　　　　　　　　(5): ReLU(inplace)     
　　　　　　　　　　　))     
　　　])     

    -----------------------result---------------------------------------------     
　　从上面打印的结果可以看到，加载的模型一般分为两个部分：features，classifier。     
　　features主要是卷积池化层，classifier主要是全连接层。      

3.模型的拆解：    
　　ourmodel = models.alexnet()    
　　rebuild_model_feature = nn.Sequential(*list(ourmodel.features._modules.values())[:-1])     
　　rebuild_model_classifier = nn.Sequential(*list(ourmodel.classifier._modules.values())[:-1])    
　　注：-1：代表一般抛弃最后一层    
　　　　在许多应用场景下，我们需要将卷积池化层与全连接层分开，在它们中间加入许多自己的组件    
		
4.固定模型某些层参数：   
　　for layer in range(10):    
　　　for p in rebuild_model_feature[layer].parameters():    
　　　　p.requires_grad = False   
　　注：requirs_grad表示该层的参数固定    
	
5.模型重组(加入自己的模型层)   
　　重组的内涵我觉得是模型嵌套：A模型中加入B模型。实现有三个步骤：    
　　1.构造B模型类（初始化函数，向前传播函数）    
　　2.在A模型类初始化函数中加入B模型类     
　　3.在A模型的向前传播函数中实现B模型的嵌套，通俗点说是，A在向前传播过程中，它前半模型的输出成为    
　　　B模型的输入，B模型的输出成为A模型后半模型的输入。    
	  
6初始化模型部分层参数：   
　　B是继承了nn.module的模型类。    
　　weight初始化：1.B.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)    
　　　　　　　　　2.B.weight.data.normal_(mean, stddev)   
　　bias初始化：B.bias.data.zero_()    

7.保存自己训练的模型：   
　　torch.save({'model':ourmodel.static_dict   
　　　　　　　　'optimizer':optimizer.static_dict   
　　　　　　　　'***':***   
　　　　　　　　}, save_file)   
8.加载自己训练的模型：  
　　load_file = ''   
　　checkpoint = torch.load(load_name)   
　　ourmodel.load_state_dict(checkpoint['model'])   
　　optimizer.load_state_dict(checkpoint['optimizer'])   
	
9.在trainval文件里的常规套路：   
　　step1.构造数据采样类   
　　step2.构造数据类（处理原始数据，并提供iter或gettm去获取单个数据）   
　　step3.构造dataloader（提供一个batch的输入数据）   
　　step4.初始化tensorf_holder（相当于占位符）   
	
　　step5.构造模型    
　　step6.设定学习率    
　　step7.构造模型中参数集params：weight，bias的值与他们的正则化系数   
　　step8.设定optimizer    
	
　　step9.进行训练模型的加载    
　　step10.设定模型的模式（train还是eval）    
　　step11.训练    
　　　step11.1读取一个batch数据      
　　　　　step11.1.1 dataloader调用Sampler类获得一个batch的index值    
　　　　　step11.1.2 dataloader根据得到的index值调用数据类获取与之对应的数据    
　　　step11.2设定模型梯度为0    
　　　step11.3确定学习率的更新    
　　　step11.4数据输入至模型进行计算loss值   
　　　step11.5设定optimizer的梯度为0     
　　　step11.6进行反向传播    
　　　step11.7梯度更新    
	
	
	
	
	
	
	
	
	
	