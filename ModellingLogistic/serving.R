#!/usr/bin/env Rscript
library(data.table)
library(rjson)
library(readr)
library(randomForest)
#json list(auto_unbox=TRUE)

#* @post /infer
#* @serializer json list(auto_unbox=TRUE)
function(req) {
  
    df <- req$postBody
    parsed_df <- rjson::fromJSON(df)
    dfr <-  as.data.frame(do.call(cbind, parsed_df))
    
    rf_logistic <- readr::read_rds("./../ml_vol/model/artifacts/rfmodel.rds")
    resvar <- readr::read_rds("./../ml_vol/model/artifacts/response_variable.rds")
    id <- readr::read_rds("./../ml_vol/model/artifacts/id.rds")
    newdf <- subset(dfr, select = -c(eval(as.name(paste0(resvar))),eval(as.name(paste0(id)))))
    predicted <- predict(rf_logistic,newdata=newdf, type="response")
     predicted <- data.table(predicted)
     names(predicted) <- "probabilities"
     predicted <- cbind(dfr,predicted)

    # where the probabilities returned are <0.5 put 0 otherwise 1.
    predicted <- setDT(predicted)[, predictions:=0][probabilities>0.5, predictions:=1]
    cols <- c(eval(id),"probabilities","predictions")
    predicted <- predicted[,..cols]
    predicted

}