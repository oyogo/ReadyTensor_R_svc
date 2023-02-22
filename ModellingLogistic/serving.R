#!/usr/bin/env Rscript
library(data.table)
library(rjson)
library(readr)
library(e1071)
library(superml)
library(glue)
#json list(auto_unbox=TRUE)

source("preprocessor.R") # calling this so that we can access the hashing function for use in creating predictor sparse matrix

#* @post /infer
#* @serializer json list(auto_unbox=TRUE)
function(req) {
  
    df <- req$postBody
    parsed_df <- rjson::fromJSON(df)
    #dfr <- (parsed_df$instances
    #dfr <-  as.data.frame(do.call(rbind, parsed_df$instances))
    dfr <- data.table::rbindlist(parsed_df$instances)
    svc_model <- readr::read_rds("./../ml_vol/model/artifacts/svcmodel.rds")
    resvar <- readr::read_rds("./../ml_vol/model/artifacts/response_variable.rds")
    thefeatures <- readr::read_rds("./../ml_vol/model/artifacts/features.rds")
    id <- readr::read_rds("./../ml_vol/model/artifacts/id.rds")
    
    #print(str(dfr))
    newdf <- subset(dfr, select = -c(eval(as.name(paste0(resvar))),eval(as.name(paste0(id)))))

    modelmat_pred <- hashing(df=newdf, features=thefeatures)
    predicted <- predict(svc_model,newdata=modelmat_pred,probability=TRUE)
    predicted <- data.table(predicted)
     names(predicted) <- "probabilities"
     predicted <- cbind(dfr,predicted)

    # where the probabilities returned are <0.5 put 0 otherwise 1.
    predicted <- setDT(predicted)[, predictions:=0][probabilities>0.5, predictions:=1]
    cols <- c(eval(id),"probabilities","predictions")
    predicted <- predicted[,..cols]
    predicted

}