# Generate paper titles

This is the DSVM branch of the repo. This code can be run directly from a machine of your choosing.

## Usage

#### Ingest model
```bash
cd ./scripts
python ingest.py --start_date 2015-01-01 --end_date 2015-01-05
```

#### Preprocess model
```bash
cd ./scripts
python preprocess.py
```

#### Train model
```bash
cd ./scripts
python nmt.py train --train-src='../data/train/abstract' --train-tgt='../data/train/title' --dev-src='../data/valid/abstract' --dev-tgt='../data/valid/title' --vocab='../data/vocab/vocab' --cuda
```

For quick testing add the arguments `--max-epoch 5` and `--valid-niter 10`
```bash
cd ./scripts
python nmt.py train --train-src='../data/train/abstract' --train-tgt='../data/train/title' --dev-src='../data/valid/abstract' --dev-tgt='../data/valid/title' --vocab='../data/vocab/vocab' --cuda --max-epoch 10 --valid-niter 10
```

### License

This work is licensed under a Creative Commons Attribution 4.0 International License.