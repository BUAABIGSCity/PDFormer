# [AAAI2023] PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction

This is a PyTorch implementation of Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction (**PDFormer**) for traffic flow prediction, as described in our paper: [Jiawei Jiang](https://github.com/aptx1231)\*, [Chengkai Han](https://github.com/NickHan-cs)\*, Wayne Xin Zhao, Jingyuan Wang,  **[Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/25556)**, AAAI2023.

> \* Equal Contributions.


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=pdformer-propagation-delay-aware-dynamic-long) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd7)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7?p=pdformer-propagation-delay-aware-dynamic-long) 

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pdformer-propagation-delay-aware-dynamic-long/traffic-prediction-on-pemsd8)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd8?p=pdformer-propagation-delay-aware-dynamic-long) 

<img src="./framework.png" width="75%">

## Requirements

Our code is based on Python version 3.9.7 and PyTorch version 1.10.1. Please make sure you have installed Python and PyTorch correctly. Then you can install all the dependencies with the following command by pip:

```shell
pip install -r requirements.txt
```

## Data

The dataset link is [Google Drive](https://drive.google.com/drive/folders/176Uogr_kty02NQcM9gB2ZT_ngulEhb0H?usp=share_link). You can download the datasets and place them in the `raw_data` directory.

All 6 datasets come from the [LibCity](https://github.com/LibCity/Bigscity-LibCity) repository, which are processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format. The only difference with the datasets provided by origin LibCity repository [here](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing) is that the filename of the datasets are differently.

Note that our model would calculate a **DTW matrix** and a **traffic pattern set** for each dataset, which is time-consuming. Therefore, we have provided DTW matrices and traffic pattern sets of all datasets in `./libcity/cache/dataset_cache/`.

## Train & Test

You can train and test **PDFormer** through the following commands for 6 datasets. Parameter configuration (**--config_file**) reads the JSON file in the root directory. If you need to modify the parameter configuration of the model, please modify the corresponding **JSON** file.

```shell
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS04 --config_file PeMS04
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS08 --config_file PeMS08
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS07 --config_file PeMS07
python run_model.py --task traffic_state_pred --model PDFormer --dataset NYCTaxi --config_file NYCTaxi --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model PDFormer --dataset CHIBike --config_file CHIBike --evaluator TrafficStateGridEvaluator
python run_model.py --task traffic_state_pred --model PDFormer --dataset T-Drive --config_file T-Drive --evaluator TrafficStateGridEvaluator
```

If you have trained a model as above and only want to test it, you can set it as follows (taking PeMS08 as an example, assuming the experiment ID during training is $ID):

```shell
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS08 --config_file PeMS08 --train false --exp_id $ID
```

**Note**: By default the result recorded in the experiment log is the average of the first n steps. This is consistent with the paper (configured as **"mode": "average"** in the JSON file). If you need to get the results of each step separately, please modify the configuration of the JSON file to **"mode": "single"**.

## Contributors

<a href="https://github.com/NickHan-cs"><img src="https://avatars.githubusercontent.com/u/59010369?v=4" width=98px></img></a> <a href="https://github.com/aptx1231"><img src="https://avatars.githubusercontent.com/u/35984903?v=4" width=98px></img></a>

## Reference Code

Code based on [LibCity](https://github.com/LibCity/Bigscity-LibCity) framework development, an open source library for traffic prediction.

## Cite

If you find the paper useful, please cite as following:

```
@inproceedings{pdformer,
  title={PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction},
  author={Jiawei Jiang and 
  		  Chengkai Han and 
  		  Wayne Xin Zhao and 
  		  Jingyuan Wang},
  booktitle = {{AAAI}},
  publisher = {{AAAI} Press},
  year      = {2023}
}
```

If you find [LibCity](https://github.com/LibCity/Bigscity-LibCity) useful, please cite as following:

```
@inproceedings{libcity,
  author    = {Jingyuan Wang and
               Jiawei Jiang and
               Wenjun Jiang and
               Chao Li and
               Wayne Xin Zhao},
  title     = {LibCity: An Open Library for Traffic Prediction},
  booktitle = {{SIGSPATIAL/GIS}},
  pages     = {145--148},
  publisher = {{ACM}},
  year      = {2021}
}
```

