## Bert

### Structure

 ```` 
 
1. configurations/ (contains the different model configurations)
2. data/ (contains the data specified in the config file)
3. bert.py (execution script - loading, traing, prediction)
4. dataset.py (torchtext object of our dataset)
    
 ````

### Dependencies 

- `pip install -r requirements`
- python --version `3.7.4`
- gcc --version `6.3.0`
- cuda --version `10.1.0`

### Commands 

- copy files remotely: `scp -r bert/* julient@login.leonhard.ethz.ch:project/bert_proc/`
- create dataset: `python dataset.py` (change main accordingly)
- launch training: `python bert.py` 

**On Cluster ([tutorial](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters))**

- prepare cluster: 
````
module load hdf5/1.10.1
module load eth_proxy
module load python_gpu/3.7.4 
module load gcc/6.3.0
````
- launch job: `bsub -n 4 -W 24:00 -R "rusage[mem=20000,ngpus_excl_p=1]" python bert.py`
