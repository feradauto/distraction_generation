# NLP: Distraction Generation


## Reproduce results

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

### Evaluation
For evaluation you should run the next command specifying the experiment number of the previous trained model.
Example assuming the experiment number "1625497756"
```bash
bsub -n 6 -W 19:59 -o eval_t5_cos_0.4 -R "rusage[mem=16384, ngpus_excl_p=1]" python evaluate.py --LAMBDA 0.4 --MODEL_ID 1625497756
```

### Datsets

