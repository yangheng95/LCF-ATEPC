# LCF-ATEPC

codes for our paper [A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction](https://arxiv.org/abs/1912.07976)

> LCF-ATEPC，面向中文及多语言的ATE和APC联合学习模型，基于PyTorch和pytorch-transformers.

> LCF-ATEPC,  a multi-task learning model for Chinese and multilingual-oriented ATE and APC task, based on PyTorch

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-multi-task-learning-model-for-chinese/aspect-based-sentiment-analysis-on-semeval)](https://paperswithcode.com/sota/aspect-based-sentiment-analysis-on-semeval?p=a-multi-task-learning-model-for-chinese)

# Note
This latest version of LCF-ATEPC has been integeated in [pyabsa](https://github.com/yangheng95/pyabsa/blob/release/examples/aspect_term_extraction/extract_aspects.py). Please follow pyabsa to get up-to-date functions and tutorials.

LCF-ATEPC模型进行方面抽取与情感预测的用法请见[这里](https://github.com/yangheng95/pyabsa/blob/release/examples/aspect_term_extraction/extract_aspects.py)。

## Here are the usages of LCF-ATEPC provided in pyabsa

Install this repo by `pip install pyabsa`.


# Aspect Term Extraction and Polarity Classification (ATEPC)
## Quick Start

### 1. Import necessary entries

```
from pyabsa.functional import ATEPCModelList
from pyabsa.functional import Trainer, ATEPCTrainer
from pyabsa.functional import ABSADatasetList
from pyabsa.functional import ATEPCConfigManager
```

### 2. Choose a base param config

```
config = ATEPCConfigManager.get_atepc_config_english()
```

### 3. Specify an ATEPC model and alter some hyper-parameters (if necessary)

```
atepc_config_english = ATEPCConfigManager.get_atepc_config_english()
atepc_config_english.num_epoch = 10
atepc_config_english.evaluate_begin = 4
atepc_config_english.log_step = 100
atepc_config_english.model = ATEPCModelList.LCF_ATEPC
```

### 4. Configure runtime setting and running training

```
laptop14 = ABSADatasetList.Laptop14

aspect_extractor = ATEPCTrainer(config=atepc_config_english, 
                                dataset=laptop14
                                )
```

### 5. Aspect term extraction & sentiment inference

```
from pyabsa import ATEPCCheckpointManager

examples = ['相比较原系列锐度高了不少这一点好与不好大家有争议',
            '这款手机的大小真的很薄，但是颜色不太好看， 总体上我很满意啦。'
            ]
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='chinese',
                                                               auto_device=True  # False means load model on CPU
                                                               )

inference_source = pyabsa.ABSADatasetList.SemEval
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source, 
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )
```
### 6. Aspect term extraction & sentiment inference output format (方面抽取及情感分类结果示例如下):

```
Sentence with predicted labels:
关(O) 键(O) 的(O) 时(O) 候(O) 需(O) 要(O) 表(O) 现(O) 持(O) 续(O) 影(O) 像(O) 的(O) 短(B-ASP) 片(I-ASP) 功(I-ASP) 能(I-ASP) 还(O) 是(O) 很(O) 有(O) 用(O) 的(O)
{'aspect': '短 片 功 能', 'position': '14,15,16,17', 'sentiment': '1'}
Sentence with predicted labels:
相(O) 比(O) 较(O) 原(O) 系(O) 列(O) 锐(B-ASP) 度(I-ASP) 高(O) 了(O) 不(O) 少(O) 这(O) 一(O) 点(O) 好(O) 与(O) 不(O) 好(O) 大(O) 家(O) 有(O) 争(O) 议(O)
{'aspect': '锐 度', 'position': '6,7', 'sentiment': '0'}

Sentence with predicted labels:
It(O) was(O) pleasantly(O) uncrowded(O) ,(O) the(O) service(B-ASP) was(O) delightful(O) ,(O) the(O) garden(B-ASP) adorable(O) ,(O) the(O) food(B-ASP) -LRB-(O) from(O) appetizers(B-ASP) to(O) entrees(B-ASP) -RRB-(O) was(O) delectable(O) .(O)
{'aspect': 'service', 'position': '7', 'sentiment': 'Positive'}
{'aspect': 'garden', 'position': '12', 'sentiment': 'Positive'}
{'aspect': 'food', 'position': '16', 'sentiment': 'Positive'}
{'aspect': 'appetizers', 'position': '19', 'sentiment': 'Positive'}
{'aspect': 'entrees', 'position': '21', 'sentiment': 'Positive'}
Sentence with predicted labels:
```
Check the detailed usages in [ATE examples](https://github.com/yangheng95/PyABSA/tree/release/examples/aspect_term_extraction) directory.


## Requirement

* Python >= 3.7
* PyTorch >= 1.0
* transformers >= 4.5.1
* Set `use_bert_spc = True` to improve the APC performance while only APC is considered.

## Training
We use the configuration file to manage experiments setting.

Training in batches by experiments configuration file, refer to the [experiments.json](experiments.json) to manage experiments.

Then, 
```sh
python train.py --config_path experiments.json
```

## About dataset

If you want to build your dataset, please find the description of the dataset [here](https://github.com/yangheng95/LCF-ATEPC/issues/25)

## Out of Memory

Since BERT models require a lot of memory. If the out-of-memory problem while training the model, here are the ways to mitigate the problem:
1. Reduce the training batch size ( train_batch_size = 4 or 8 )
2. Reduce the longest input sequence ( max_seq_length = 40 or 60 )
3. Set `use_unique_bert = true` to use a unique BERT layer to model for both local and global contexts

## Model Performance

We made our efforts to make our benchmarks reproducible. However, the performance of the LCF-ATEPC models fluctuates and any slight changes in the model structure could also influence performance. Try different random seed to achieve optimal results.

### Performance on Chinese Datasets

![chinese](assets/Chinese-results.png)

### Performance on Multilingual Datasets

![multilingual](assets/multilingual-results.png)

### Optimal Performance on Laptop and Restaurant Datasets

![semeval2014](assets/SemEval-2014-results.png)

## Model Architecture
![lcf](assets/lcf-atepc.png)

## Notice

We cleaned up and refactored the original codes for easy understanding and reproduction.
However, we didn't test all the training situations for the refactored codes. If you find any issue in this repo,
You can raise an issue or submit a pull request, whichever is more convenient for you.

Due to the busy schedule, some module may not update for long term, such as saving and loading module for trained models, inferring module, etc. If possible, we sincerely request for someone to accomplish these work.

## Citation

If this repository is helpful to you, please cite our paper:

    @misc{yang2019multitask,
        title={A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction},
        author={Heng Yang and Biqing Zeng and JianHao Yang and Youwei Song and Ruyang Xu},
        year={2019},
        eprint={1912.07976},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

## Licence

MIT License

