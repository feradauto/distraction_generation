# NLP: Distraction Generation


## Reproduce results

### Environment

Create environment
```bash
mkvirtualenv "nlp"
```
Load modules
```bash
module load python_gpu/3.7.4
module load eth_proxy
module load tmux/2.6
```
Activate environment
```bash
source $HOME/.local/bin/virtualenvwrapper.sh
workon "nlp"
```
Load environment variable for the location of the data
```bash
export NLP_DATA="/cluster/home/fgonzalez/nlp/data/"
```
Clone the repo and install requirements
```bash
pip install requirements.txt
```
### Datsets
Download the datasets from https://github.com/Yifan-Gao/Distractor-Generation-RACE/tree/master/data and place them in $NLP_DATA/distractor/

### Training
In order to reproduce results of T5 small you should run the next command

```bash
bsub -n 6 -W 19:59 -o t5_small -R "rusage[mem=16384, ngpus_excl_p=1]" python train_small.py
```

To reproduce results of T5 base you just need to change the parameter that indicates the type of model
```bash
bsub -n 6 -W 19:59 -o t5_base -R "rusage[mem=16384, ngpus_excl_p=1]" python train_small.py  --MODEL t5-base
```

To train the model with the modified loss function the command is
```bash
bsub -n 6 -W 19:59 -o t5_cos_0.4 -R "rusage[mem=16384, ngpus_excl_p=1]" python train_loss_bleu.py --LAMBDA 0.4
```

You can also specify the parameters of the model directly in the file configuration.py

### Evaluation
For evaluation you should run the next command specifying the experiment number of the previous trained model.
Example assuming the experiment number "1625497756"
```bash
bsub -n 6 -W 19:59 -o eval_t5_cos_0.4 -R "rusage[mem=16384, ngpus_excl_p=1]" python evaluate.py --LAMBDA 0.4 --MODEL_ID 1625497756
```


