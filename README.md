## CIL Project 2020: Twitter Sentiment Analysis

1. [ Description. ](#desc)
2. [ Structure. ](#struct)
3. [ Models. ](#models)
4. [ Deployment. ](#depl)
4. [ Links. ](#links)

<a name="desc"></a>
### Description

Labelling small texts as **positive or negative** is a key-challenge in Natural Language Understanding. 
In this work we combined different methods and used transfer learning on a dataset of 2.5M Tweets to classify them
between these two categories of sentiment with the highest accuracy. We achieved an accuracy of **90.32** with our best model.


<a name="models"></a>
### Models

> BERT
>
> RoBERTa
>
> XLNET

<a name="struct"></a>
### Structure

- `src/` contains the sample and the full *dataset*, the *preprocessor* and some *helpers* to convert the dataset in some other formats.
- Each model has its own package `model/`

<a name="depl"></a>
### Deployment on Leonhard

#### Dependencies 

- `pip install -r requirements`
- python --version `3.7.4`
- gcc --version `6.3.0`
- cuda --version `10.1.0`

#### Commands 

- copy files remotely: `scp -r $MODEL_NAME$/* julient@login.leonhard.ethz.ch:`

**On Leonhard cluster ([tutorial](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters))**

- prepare cluster: 
````
module load hdf5/1.10.1
module load eth_proxy
module load python_gpu/3.7.4 
module load gcc/6.3.0
````
- launch job: `bsub -n 4 -W 24:00 -R "rusage[mem=20000,ngpus_excl_p=1]" python $MODEL_NAME$.py`


<a name="links"></a>
### Links

[BERT paper](https://arxiv.org/abs/1810.04805)

[BERT github repository](https://github.com/google-research/bert)

[RoBERTa paper](https://arxiv.org/abs/1907.11692)

[RoBERTa github repository](https://github.com/pytorch/fairseq/tree/master/examples/roberta)

[XLNET paper](https://arxiv.org/abs/1906.08237)

[XLNET github repository](https://github.com/zihangdai/xlnet)