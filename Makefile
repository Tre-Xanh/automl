# SHELL := /bin/bash
SRC = automl/*.py
DATA = data/processed/train.csv data/test/dftest.csv
PREP_MODEL = models/prep.model
MLFLOW_RUN = mlflow run .

all: train serve
train: train_autogluon
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train

train_h2o: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) -e train_h2o

train_autogluon: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) -e train_autogluon

preproc $(DATA) $(PREP_MODEL): automl/preprocess.py
	$(MLFLOW_RUN) -e preprocess

devenv: conda*.yml
	conda env update -f conda.yml
	conda env update -f conda-dev.yml
	python --version 
	gcc --version
	java -version

cleancode:
	bash scripts/clean_code.sh

clean:
	rm -Rf tmp
