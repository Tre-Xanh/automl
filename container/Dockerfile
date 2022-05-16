FROM mambaorg/micromamba

ARG MODEL4CONTAINER
ENV MLFLOW_MODEL=/models
COPY --chown=$MAMBA_USER:$MAMBA_USER ${MODEL4CONTAINER}/conda.yaml ${MLFLOW_MODEL}/conda.yaml
ENV ENV_NAME=app
RUN micromamba create -n ${ENV_NAME} -y -f ${MLFLOW_MODEL}/conda.yaml && \
    micromamba clean --all --yes

COPY --chown=$MAMBA_USER:$MAMBA_USER ${MODEL4CONTAINER} ${MLFLOW_MODEL}
CMD mlflow models serve -m ${MLFLOW_MODEL} --env-manager=local
