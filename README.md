# CIRNet_TIP2022

Runmin Cong, Qinwei Lin, Chen Zhang, Chongyi Li, Xiaochun Cao, Qingming Huang, and Yao Zhao, CIR-Net: Cross-modality interaction and refinement for RGB-D salient object detection, IEEE Transactions on Image Processing, vol. 31, pp. 6800-6815, 2022.

# Results of CIR-Net
* Results:
 - We provide the resutls of our CIR-Net on six popular RGB-D SOD benchmark datasets, including STEREO797, NLPR, NJUD, DUT, LFSD and SIP.
 - The results can be download from: [Baidu Cloud](https://pan.baidu.com/s/1vZupiTkXI3ZTIYLrdEL9UQ) (password:1234)

# Pytorch Code of CIR-Net:
* Pytorch implementation of CIR-Net
* Pretrained model:
  - We provide our testing code. If you test our model, please download the pretrained model, unzip it, and put the checkpoint `CIRNet.pth` to `CIRNet_cpts/` folder 
  - Pretrained model using ResNet50 backbone:[Baidu Cloud](https://pan.baidu.com/s/1QUoGbqgaZhalwJxoDOpL8A) (password:1234)
  - Pretrained model using VGG16 backbone: [Baidu Cloud](https://pan.baidu.com/s/1tP3XFXhmAjC2Q3I8lC7TwQ) (password:1234)


## Requirements

* Python 3.7
* torch=1.10.1
* torchvision=0.11.2
* opencv-python
* Pillow

## Data Preprocessing
* Please download and put the train data to `data` folder.
* train data can be download from: [Baidu Cloud](https://pan.baidu.com/s/1NFt3eSpdNA99DuP9O5zpHA) (password:1234)
* test data can be download from: [Baidu Cloud](https://pan.baidu.com/s/1KVCLaXLrMZDUZDpYBd_SJA) (password:1234)

## Test
```
python3 CIRNet_test.py --backbone R50 --test_model CIRNet_R50.pth
```

## Train
```
python3 CIRNet_train.py --backbone R50
```

* You can find the results in the `test_maps` folder

# If you use our CIR-Net, please cite our paper:

# Contact Us
If you have any questions, please contact Runmin Cong (rmcong@bjtu.edu.cn) or Qinwei Lin (lqw22@mails.tsinghua.edu.cn).
