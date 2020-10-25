SHELL := /bin/bash

all: serve_h2o
train_h2o: tmp/run_env.sh
tmp/run_env.sh: automl/*.py train_data
	mlflow run . -e train_h2o

train_autogluon: automl/*.py train_data
	mlflow run . -e train_autogluon

train_data: data/processed/train.csv data/preprocess/test.csv
data/processed/train.csv:
data/preprocess/test.csv:
test/data/dftest.csv:
	mlflow run . -e preprocess

serve:
	. tmp/run_env.sh && scripts/mlflow_serve.sh

test: test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py
	
test_load_model: test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py::test_load_model
	
test_api: test/data/dftest.csv
	. tmp/run_env.sh && pytest test/test.py::test_api

devenv: conda*.yml
	conda env update -f conda-dev.yml
	conda env update -f conda.yml
	/opt/conda/bin/activate automl && python --version && java -version


clean:
	rm -Rf tmp

.PHONY: test
