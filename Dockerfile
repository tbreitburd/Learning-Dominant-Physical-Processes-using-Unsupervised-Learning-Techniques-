FROM continuumio/miniconda3

RUN mkdir -p Project_24

COPY . /Project_24

WORKDIR /Project_24

RUN apt-get update && apt-get install -y \
    gfortran

RUN conda env update -f environment.yml --name Project_24

RUN apt-get update && apt-get install -y \
    git \
    dvipng texlive-latex-extra texlive-fonts-recommended cm-super

RUN echo "conda activate Project_24" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN git init
RUN pip install pre-commit
RUN pre-commit install
