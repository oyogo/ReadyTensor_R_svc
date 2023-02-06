FROM rocker/r-ver:4.1.0
#FROM rstudio/plumber

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
  git-core \
  libssl-dev \
  libcurl4-gnutls-dev \
  curl \
  libsodium-dev \
  libxml2-dev 


RUN R -e "install.packages('data.table', dependencies=T)"
RUN R -e "install.packages('rjson', dependencies=T)"
RUN R -e "install.packages('plumber', dependencies=T)"
RUN R -e "install.packages('jsonlite', dependencies=T)"
RUN R -e "install.packages('tidyr', dependencies=T)"
RUN R -e "install.packages('e1071', dependencies=T)"
RUN R -e "install.packages('FeatureHashing', dependencies=T)"


COPY ./ModellingLogistic /opt/ModellingLogistic
WORKDIR /opt/ModellingLogistic

ENV PATH="opt/ModellingLogistic:${PATH}"

RUN chmod +x train &&\
    chmod +x predict &&\
    chmod +x serve

