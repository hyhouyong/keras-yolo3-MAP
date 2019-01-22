# keras-yolo3-MAP
# 详细教你如何使用测试图片进行MAP评估
![](https://github.com/hyhouyong/keras-yolo3-MAP/blob/master/results/banshou.png)
# 前提：
        git clone https://github.com/hyhouyong/keras-yolo3.git
## groundtruths
### 1.生成真实值的框体文件(存放在from_kerasyolo3/version_...):
        python convert_keras-yolo3.py --gt test.txt 
### 2.将version_..文件夹移动到上一级目录下，并改名为groundtruths

# detections
### 1.将test.txt文件中的图片进行预测，将结果写入txt文件中
        python yolo_new.py -g config.yml --weights logs/000/xx.h5
* 注意：将model.py替换model_data的model.py
### 2.将生成的txt文件改为pred.txt
### 3.生成预测值的框体文件(存放在from_kerasyolo3/version_...)：
        python convert_keras-yolo3.py --pred pred.txt
### 4.将version_..文件夹移动到上一级目录下，并改名为detections

# AP + MAP
        python pascalvoc.py
####控制台输出：
        AP: 20.18% (banshou)
        mAP: 20.18%
####生成的results.txt文件，存放在result文件夹下
        Average Precision (AP), Precision and Recall per class:

        Class: banshou
        AP: 20.18%
        Precision: ['0.00', '0.00', '0.00', '0.00', '0.20', '0.17', '0.14', '0.12', '0.11', '0.10', '0.09', '0.08', '0.08', '0.07', '0.07', '0.06', '0.06', '0.06', '0.05', '0.05', '0.05', '0.05', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.03', '0.04', '0.05', '0.05', '0.06', '0.06', '0.06', '0.06', '0.06', '0.07', '0.07', '0.07', '0.07', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.06', '0.07', '0.07', '0.07', '0.07', '0.08', '0.08', '0.08', '0.09', '0.09', '0.10', '0.09', '0.09', '0.10', '0.11', '0.12', '0.12', '0.13', '0.14', '0.15', '0.15', '0.16', '0.17', '0.17', '0.18', '0.18', '0.19', '0.20', '0.20', '0.21', '0.21', '0.21', '0.21', '0.22', '0.22', '0.22', '0.23', '0.23', '0.22', '0.22', '0.22', '0.22', '0.22', '0.23', '0.23', '0.23', '0.23', '0.22', '0.23', '0.23', '0.23', '0.23', '0.24', '0.24', '0.24', '0.24', '0.24', '0.24', '0.24', '0.24', '0.24', '0.23', '0.23', '0.23', '0.23', '0.23', '0.23', '0.23', '0.23', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.22', '0.21', '0.22', '0.22', '0.22', '0.21', '0.21', '0.21', '0.21', '0.21', '0.21', '0.21', '0.21', '0.21', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.20', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.19', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.18', '0.17', '0.17']
        Recall: ['0.00', '0.00', '0.00', '0.00', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.02', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.04', '0.06', '0.08', '0.08', '0.10', '0.10', '0.10', '0.10', '0.10', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.12', '0.14', '0.14', '0.16', '0.16', '0.18', '0.18', '0.18', '0.20', '0.20', '0.22', '0.22', '0.22', '0.24', '0.27', '0.29', '0.31', '0.33', '0.35', '0.37', '0.39', '0.41', '0.43', '0.45', '0.47', '0.49', '0.51', '0.53', '0.55', '0.57', '0.59', '0.59', '0.59', '0.61', '0.63', '0.63', '0.65', '0.65', '0.65', '0.65', '0.65', '0.65', '0.67', '0.69', '0.69', '0.69', '0.69', '0.69', '0.71', '0.71', '0.73', '0.73', '0.76', '0.78', '0.78', '0.78', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.80', '0.82', '0.82', '0.82', '0.82', '0.82', '0.82', '0.82', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84', '0.84']


        mAP: 20.18%
reference：https://github.com/gustavovaliati/keras-yolo3 <br>
           https://github.com/Cartucho/mAP<br>
           https://github.com/rafaelpadilla/Object-Detection-Metrics
        
  
