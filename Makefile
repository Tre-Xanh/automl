SHELL := /bin/bash

all: serve
train tmp/run_env.sh: data/processed/train.csv automl/*.py
	mlflow run . -e train_h2o

train_autogluon: automl/*.py
	mlflow run . -e train_autogluon

data/processed/train.csv test/data/dftest.csv: automl/preprocess.py
	mlflow run . -e preprocess

serve: tmp/run_env.sh
	. tmp/run_env.sh && scripts/mlflow_serve.sh

.PHONY: test
test: tmp/run_env.sh test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py

test_load_model: tmp/run_env.sh test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py::test_load_model

test_api: tmp/run_env.sh test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py::test_api

devenv: conda*.yml
	conda env update -f conda-dev.yml
	conda env update -f conda.yml
	/opt/conda/bin/activate automl && python --version && java -version

cleancode:
	bash scripts/clean_code.sh

clean:
	rm -Rf tmp
