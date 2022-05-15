# include .env
# export

SHELL := /bin/bash
SRC = */*.py
DATA = data/processed/train.csv data/test/dftest.csv
PREP_MODEL = models/prep.model
DEV_ENV ?= automl
MLFLOW_RUN = mamba run -n $(DEV_ENV) mlflow run

all: train
train: train_autogluon
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train

train_h2o: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) h2o_mlflow -e train_h2o

train_autogluon: $(SRC) $(DATA) $(PREP_MODEL)
	$(MLFLOW_RUN) autogluon_mlflow -e train_autogluon

preproc $(DATA) $(PREP_MODEL): common/preprocess.py
	$(MLFLOW_RUN) common -e preprocess

devenv: conda*.yml
	mamba env update -f conda-dev.yml -n $(DEV_ENV)
	python --version
	gcc --version

cleancode:
	bash scripts/clean_code.sh

clean:
	rm -Rf tmp
