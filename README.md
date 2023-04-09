# OnsetDetection
This repo includes the code base for a onset detection challenge by the CP

## Installation
1. Clone the repo
2. Install the requirements
``` pip install -r requirements.txt ```
## Prepare the data
## Run the training script
``` python train.py --data_dir <path to data> --model_dir <path to save model> --log_dir <path to save logs> --batch_size <batch size> --num_epochs <number of epochs> --learning_rate <learning rate> -gpu <gpu id>```
## Run the inference script
``` python inference.py --data_dir <path to data> --model_dir <path to pretrained model> ```
## Run the evaluation script
``` python evaluate.py --data_dir <path to data> --model_dir <path to pretrained model> ```