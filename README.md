# Bone abnormality detection
Musculoskeletal disorders (MSD) affect over 1.7 billion people globally. Common MSD include fracture and dislocation. In this project, deep learning is applied to identify bone abnormality from front limb X-ray images. 

## Author:
Yu-Hsiang Chen

## SYNOPSIS:
train.py --data $datapath
inference.py --data $datapath --output $output_file_path
======
BEFORE YOU RUN:
Please download the checkpoint_2_.pth file from the drive below,
and put it in $datapath/.
https://drive.google.com/drive/folders/1frqpYjMYvr7ft8FJFt9NTP_kQ7BKx9bk?usp=sharing
======
## HOW TO RUN:
model training: 
```bibtex
python train.py--data $datapath
```
The model is trained with a pretrained RegnetX, and the trained model parameters is stored in [here](https://drive.google.com/drive/folders/1frqpYjMYvr7ft8FJFt9NTP_kQ7BKx9bk?usp=sharing) <br>
You can use the trained model to do inference directly.<br>
model inference: <br>
```bibtex
python inference.py --data $datapath --output $output_file_path
```
