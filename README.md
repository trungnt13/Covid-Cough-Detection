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
export NCPU=4

./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH $NCPU 111
./run_pseudo_gen.sh $COVIPATH cuda:0 $BATCH $NCPU 222
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
export NCPU=4

./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH $NCPU 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_ecapa $BATCH $NCPU 222

./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH $NCPU 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_xvec $BATCH $NCPU 222

./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH $NCPU 111
./run_pseudo_res.sh $COVIPATH cuda:0 simple_langid $BATCH $NCPU 222

./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH $NCPU 111
./run_pseudo_res.sh $COVIPATH cuda:0 domain_xvec $BATCH $NCPU 222

./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH $NCPU 111
./run_pseudo_res.sh $COVIPATH cuda:0 domain_ecapa $BATCH $NCPU 222
```

## Run Contrastive Training Then Fine-tuning and Evaluation

**This block run after the first two blocks.**

Arguments:

* $1: Path
* $2: Device
* $3: Model
* $4: Batch size (pretrain)
* $5: Batch size (finetune)
* $6: cut
* $7: seed

> Train with different mixup ratio for pretrain and finetune

```shell
export COVIPATH=/mnt/sdb1/covid_data
export BATCH1=5
export BATCH2=8
export NCPU=4

./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_ecapa $BATCH1 $BATCH2 15 111

./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 15 111
./run_contrastive.sh $COVIPATH cuda:0 contrastive_xvec $BATCH1 $BATCH2 15 111

```
