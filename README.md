# Bone abnormality detection
Musculoskeletal disorders (MSD) affect over 1.7 billion people globally. Common MSD include fracture and dislocation. In this project, deep learning is applied to identify bone abnormality from front limb X-ray images. 

## Author:
Yu-Hsiang Chen

## HOW TO RUN:
model training: 
```bibtex
python train.py--data $datapath
```
The model is trained with a pretrained RegnetX on a dataset from [Kaggle](https://www.kaggle.com/competitions/bone-abnormality-classification/overview), and the trained model parameters is stored in [here](https://drive.google.com/drive/folders/1frqpYjMYvr7ft8FJFt9NTP_kQ7BKx9bk?usp=sharing) <br>
You can use the trained model to analyze data directly.<br><br>
model inference: <br>
```bibtex
python inference.py --data $datapath --output $output_file_path
```
