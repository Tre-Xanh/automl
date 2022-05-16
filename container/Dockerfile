FROM mambaorg/micromamba

COPY --chown=$MAMBA_USER:$MAMBA_USER conda-serve.yml /tmp/env.yaml
ENV CONDA_ENV=mlflow_exec
RUN micromamba create -n ${CONDA_ENV}  && \
    micromamba install -n ${CONDA_ENV} -y -f /tmp/env.yaml && \
    micromamba clean --all --yes

ENV MLFLOW_MODEL=/models
# COPY --chown=$MAMBA_USER:$MAMBA_USER ${MODEL4CONTAINER} ${MLFLOW_MODEL}
CMD micromamba run -n ${CONDA_ENV} mlflow models serve -m ${MLFLOW_MODEL}
