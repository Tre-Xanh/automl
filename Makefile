# include .env
# export

SHELL := /bin/bash
SRC = */*.py
DATA = data/processed/train.csv data/test/dftest.csv
PREP_MODEL = models/prep.model
DEV_ENV ?= automl
MLFLOW_RUN = mamba run -n $(DEV_ENV) --no-capture-output --live-stream mlflow run

all: preproc train
train: train_autogluon train_h2o log_coordinator
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train

log_coordinator:
	$(MLFLOW_RUN) coordinator_mlflow -e log_coordinator

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

include .env
include .trained.env
export

DOCKER_NAME=automl
API_PORT?=5000
CONTAINER_MODELS=container/models

list_artifacts:
	tree $(MLFLOW_AUTOGLUON)

serve: serve_docker

serve_model:
	mlflow models serve -m $(MLFLOW_AUTOGLUON)

test: test_autogluon

test_h2o:
	mlflow run h2o_mlflow -e test

test_autogluon:
	mlflow run autogluon_mlflow -e test

build_docker:
	rm -Rf $(CONTAINER_MODELS)
	mkdir -p $(CONTAINER_MODELS)
	cp -R $(MLFLOW_AUTOGLUON) $(CONTAINER_MODELS)/MLFLOW_AUTOGLUON
	cp -R $(MLFLOW_H2OAUTOML) $(CONTAINER_MODELS)/MLFLOW_H2OAUTOML
	cp -R $(MLFLOW_COORDINATOR) $(CONTAINER_MODELS)/MLFLOW_COORDINATOR
	docker-compose build

serve_docker: build_docker
	docker-compose up --remove-orphans # --no-recreate

scale_docker:
	docker-compose scale predictorA=3 predictorB=3 coordinator=1

down:
	docker-compose down
