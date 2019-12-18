# LCF-ATEPC

> codes for paper [A Multi-task Learning Model for Chinese-oriented Aspect Polarity Classification and Aspect Term Extraction](https://arxiv.org/abs/1912.07976)

> LCF-ATEPC，面向中文及多语言的ATE和APC联合学习模型。

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

## Requirement

* python 3.6 / 3.7
* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) >= 1.2.0


## Training
> We use the configuration file to manage experiments setting

Traininng on Chinese-oriented Review dataset: complete the experiments configuration [exp-batch-chinese.json](./exp-batch-chinese.json) and run 

```sh
python batch_train_chinese.py
```

Else traininng on English or multilingual Review dataset: complete the experiments configuration [exp-batch.json](./exp-batch.json) and run 

```sh
python batch_train.py
```

## Model Architecture
![lcf](assets/lcf-atepc.png)

## Notice

We cleaned up and refactored the original codes for easy understanding and reproduction.
Due to the busy schedule, we didn't test all the training situations. If you find any issue in this repo,
You can raise an issue or submit a pull request, whichever is more convenient for you.

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

