SHELL := /bin/bash
SRC = automl/*.py
DATA = data/processed/train.csv test/data/dftest.csv
MLFLOW_RUN = mlflow run . --no-conda

all: train serve
train: train_autogluon
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train
train_h2o: $(SRC) $(DATA)
	$(MLFLOW_RUN) -e train_h2o

train_autogluon: $(SRC) $(DATA)
	$(MLFLOW_RUN) -e train_autogluon

data/processed/train.csv test/data/dftest.csv: automl/preprocess.py
	$(MLFLOW_RUN) -e preprocess

serve: tmp/run_env.sh
	. tmp/run_env.sh && scripts/mlflow_serve.sh

.PHONY: test
test:
	. tmp/run_env.sh && pytest test/test.py

devenv: conda*.yml
	conda env update -f conda-dev.yml
	conda env update -f conda.yml
	python --version && java -version

cleancode:
	bash scripts/clean_code.sh

clean:
	rm -Rf tmp
