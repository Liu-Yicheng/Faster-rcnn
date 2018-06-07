# 进度提示（标志：仓库程序文件还没有完善好）
- [x] 完成README-工程代码部分：整体代码提交与测试   
　　2018.6.6 完成代码提交     
　　2018.6.6 完成代码测试：[6步即可训练与测试](#程序运行)         
- [ ] 完成README-Faster模型详解部分
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
				
				


