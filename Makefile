all: clean serve

serve: tmp/serve.sh
	bash tmp/serve.sh

test: tmp/test.sh
	bash tmp/test.sh
	
tmp/serve.sh tmp/test.sh:
	mlflow run .

clean:
	rm -Rf tmp