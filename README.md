# Training Instruction

## Generate Pseudo-labels for Age and Gender

Arguments:

* $1: Path
* $2: Device
* $3: Batch size
* $4: NCPU
* $5: SEED

> Training model with different seed for ensemble

```shell
export COVIPATH=/mnt/sdb1/covid_data
export BATCH=8
./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH 4 111
./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH 4 222
./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH 4 333
./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH 4 444
```

## Generate Pseudo-labels for Results

Arguments:

* $1: Path
* $2: Device
* $3: Model
* $4: Batch size
* $5: NCPU
* $6: SEED

> Training multiple model with different seed for ensemble

```shell
export COVIPATH=/mnt/sdb1/covid_data
export BATCH=8
./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH 4 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH 4 222
./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH 4 333
./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH 4 444

./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH 4 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH 4 222
./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH 4 333
./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH 4 444

./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH 4 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH 4 222
./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH 4 333
./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH 4 444

./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH 4 111
./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH 4 222
./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH 4 333
./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH 4 444

./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH 4 111
./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH 4 222
./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH 4 333
./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH 4 444
```

## Run Contrastive Training Then Fine-tuning and Evaluation

**This block run after the first two blocks.**

Arguments:

* $1: Path
* $2: Device
* $3: Model
* $4: Batch size (pretrain)
* $5: Batch size (finetune)
* $6: Mixup (pretrain)
* $7: Mixup (finetune)
* $8: NCPU
* $9: SEED

> Train with different mixup ratio for pretrain and finetune

```shell
export COVIPATH=/mnt/sdb1/covid_data
export BATCH1=5
export BATCH2=8

./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 0.8 0.6 4 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 0.8 0.0 4 111

./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 0.8 0.6 4 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 0.8 0.0 4 111

./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 0.0 0.6 4 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 0.0 0.0 4 111

./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 0.0 0.6 4 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 0.0 0.0 4 111

```
