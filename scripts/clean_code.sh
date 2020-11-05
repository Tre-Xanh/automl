set -ex
shopt -s globstar

cd automl

autoflake --in-place --remove-all-unused-imports --remove-unused-variables \
    **/*.py

isort --atomic -m 3 .

black .