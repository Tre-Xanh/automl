# include .env
# export

SHELL := /bin/bash
SRC = */*.py
DATA = data/processed/train.csv data/test/dftest.csv
PREP_MODEL = models/prep.model
DEV_ENV ?= automl
MLFLOW_RUN = mamba run -n $(DEV_ENV) --no-capture-output --live-stream mlflow run

all: preproc train
train: train_autogluon train_h2o log_gateway
tmp/run_env.sh: $(SRC) $(DATA)
	$(MAKE) train

log_gateway:
	$(MLFLOW_RUN) mlflow_gateway -e log_gateway

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

list_artifacts:
	tree $(MLFLOW_AUTOGLUON)

serve: serve_docker

serve_model:
	mlflow models serve -m $(MLFLOW_AUTOGLUON)

test:
	mlflow run common -e test

test_h2o:
	mlflow run h2o_mlflow -e test

test_autogluon:
	mlflow run autogluon_mlflow -e test

build_docker:
	docker-compose build

serve_docker: build_docker
	docker-compose up --remove-orphans --no-recreate

scale_docker:
	docker-compose scale predictorA=3 predictorB=3 gateway=1

down:
	docker-compose down
