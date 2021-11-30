IMG_NAME=automl
mlflow models build-docker -m "${MLFLOW_MODEL}" -n ${IMG_NAME}
docker run -p 5001:8080 ${IMG_NAME}
