# Bone abnormality detection
Musculoskeletal disorders (MSD) affect over 1.7 billion people globally. Common MSD include fracture and dislocation. In this project, deep learning is applied to identify bone abnormality from front limb X-ray images. 

Author: Yu-Hsiang Chen
Date:2021/12/1
=====
SYNOPSIS:

train.py --data $datapath
inference.py --data $datapath --output $output_file_path
======
BEFORE YOU RUN:

Please download the checkpoint_2_.pth file from the drive below,
and put it in $datapath/.
https://drive.google.com/drive/folders/1frqpYjMYvr7ft8FJFt9NTP_kQ7BKx9bk?usp=sharing
======
HOW TO RUN:

python inference.py --data $datapath --output $output_file_path
The out.csv file containing all predictions of testing data will locate in $output_file_path/.
