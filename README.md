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

To use our models, you may need download `en_core_web_sm` by

`python -m spacy download en_core_web_sm`

### Aspect Extraction Output Format (方面术语抽取结果示例如下):
```
Sentence with predicted labels:
It(O) was(O) pleasantly(O) uncrowded(O) ,(O) the(O) service(B-ASP) was(O) delightful(O) ,(O) the(O) garden(B-ASP) adorable(O) ,(O) the(O) food(B-ASP) -LRB-(O) from(O) appetizers(B-ASP) to(O) entrees(B-ASP) -RRB-(O) was(O) delectable(O) .(O)
{'aspect': 'service', 'position': '7', 'sentiment': 'Positive'}
{'aspect': 'garden', 'position': '12', 'sentiment': 'Positive'}
{'aspect': 'food', 'position': '16', 'sentiment': 'Positive'}
{'aspect': 'appetizers', 'position': '19', 'sentiment': 'Positive'}
{'aspect': 'entrees', 'position': '21', 'sentiment': 'Positive'}
Sentence with predicted labels:
How(O) pretentious(O) and(O) inappropriate(O) for(O) MJ(O) Grill(O) to(O) claim(O) that(O) it(O) provides(O) power(O) lunch(B-ASP) and(O) dinners(B-ASP) !(O)
{'aspect': 'lunch', 'position': '14', 'sentiment': 'Negative'}
{'aspect': 'dinners', 'position': '16', 'sentiment': 'Negative'}

```

## Quick Start

1. Convert APC datasets to ATEPC datasets

```
from pyabsa import convert_apc_set_to_atepc

convert_apc_set_to_atepc(r'../apc_usages/datasets/restaurant16')
```

2. Training for ATEPC

```
from pyabsa import train_atepc

# see hyper-parameters in pyabsa/main/training_configs.py
param_dict = {'model_name': 'lcf_atepc',
              'batch_size': 16,
              'seed': 1,
              'device': 'cuda',
              'num_epoch': 5,
              'optimizer': "adamw",
              'learning_rate': 0.00002,
              'pretrained_bert_name': "bert-base-uncased",
              'use_dual_bert': False,
              'use_bert_spc': False,
              'max_seq_len': 80,
              'log_step': 30,
              'SRD': 3,
              'lcf': "cdw",
              'dropout': 0,
              'l2reg': 0.00001,
              'polarities_dim': 3
              }

# Mind that polarities_dim = 2 for Chinese datasets, and the 'train_atepc' function only evaluates in last few epochs

train_set_path = 'atepc_datasets/restaurant14'
save_path = '../atepc_usages/state_dict'
aspect_extractor = train_atepc(parameter_dict=param_dict,  # set param_dict=None to use default model
                               dataset_path=train_set_path,  # file or dir, dataset(s) will be automatically detected
                               model_path_to_save=save_path,
                               auto_evaluate=True,  # evaluate model while training if test set is available
                               auto_device=True  # Auto choose CUDA or CPU
                               )

```

3. Extract aspect terms
```
from pyabsa import load_aspect_extractor

examples = ['But the staff was so nice to us .',
            'But the staff was so horrible to us .',
            r'Not only was the food outstanding , but the little ` perks \' were great .',
            'It took half an hour to get our check , which was perfect since we could sit , have drinks and talk !'
            ]
            
# Download the provided pre-training model from Google Drive
model_path = 'state_dict/lcf_atepc_cdw_rest14_without_spc'

aspect_extractor = load_aspect_extractor(trained_model_path=model_path,
                                         auto_device=True)

atepc_result = aspect_extractor.extract_aspect(examples,
                                               print_result=True,
                                               pred_sentiment=True)
# print(atepc_result)

```

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

