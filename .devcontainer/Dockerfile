FROM continuumio/miniconda3

# Options for common setup script
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && /bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends gcc make python3-dev

RUN curl -fsSL https://get.docker.com -o get-docker.sh &&\
    sh get-docker.sh && docker --version

# RUN which conda && conda update -n base -c defaults conda
RUN conda install mamba -y -n base -c conda-forge
COPY conda-dev.yml ./
RUN mamba env update -f conda-dev.yml -n py37
COPY conda.yml ./
RUN mamba env update -f conda.yml -n py37