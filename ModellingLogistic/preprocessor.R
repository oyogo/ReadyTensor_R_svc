
#####*****Preprocessing script*****######
#####* This is where we process the data, say imputing NA's with mean plus any other transformation that can be done. 
#####* This function will be sourced in the training script before fitting the model. 

library(data.table) # Opted for this, 1. Because its really fast 2. dplyr conflicted with plumber
library(rjson) # for handling json data
library(FeatureHashing)
library(superml)
library(glue)


hashing <- function(df,features){
  
  mat <- hashed.model.matrix(features,
                             df, hash.size = 2 ^ 10,
                             create.mapping = TRUE)
  return(mat)
  
}

preprocessing <- function(fname_train,fname_schema,genericdata,dataschema){ 

  # get the response variable and store it as a string to a variable
  varr <- dataschema$inputDatasets$binaryClassificationBaseMainInput$targetField
  
# drop the id field 
## get the field name and store it as a variable
idfieldname <- dataschema$inputDatasets$binaryClassificationBaseMainInput$idField

# get the predictor fields from the dataschemaa. 
predictor_fields <- data.frame(dataschema$inputDatasets$binaryClassificationBaseMainInput$predictorFields)

# convert the dataframe to data.table for munging :- don't want to use dplyr
predictor_fields <- setDT(predictor_fields)

# melt the data.table into long format so as to filter numeric columns. 
predictor_fields <- melt(predictor_fields,measure.vars=patterns(fieldNames="fieldName",dataTypes="dataType"))
#predictor_fields$fieldNames <- gsub("%", "x", predictor_fields$fieldNames)
# filter the numeric columns 
num_vars <- predictor_fields[dataTypes=="NUMERIC",.(fieldNames)]

# categorical variables
cat_vars <- predictor_fields[dataTypes=="CATEGORICAL",.(fieldNames)]

catcols <- as.vector(cat_vars$fieldNames)
v <- num_vars$fieldNames

# loop through the numeric columns and replace na values with mean of the same column in which the na appears.
for (coll in v){

 genericdata <-  genericdata[, (coll) := lapply(coll, function(x) {
    x <- get(x)
    x[is.na(x)] <- mean(x, na.rm = TRUE)
    x
  })]

}

# for the missing values in categorical variables replace them with mode
my_mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

for (cat_coll in cat_vars) {
  genericdata <- as.data.frame(genericdata)
  genericdata[is.na(genericdata[,cat_coll]),cat_coll] <- my_mode(genericdata[,cat_coll], na.rm = TRUE)

}

var <- as.symbol(varr)

# label encode the response variable
lbl <- LabelEncoder$new()

genericdata[,c(glue({var}))] <- lbl$fit_transform(genericdata[,c(glue({var}))])

#genericdata <- setDT(genericdata)[,(glue({var})) := factor(glue({var}), ordered = FALSE)]

data_withid <- genericdata # will need this for the prediction output(we need the id column for aligning the predictions with the respective id)
data_noid <- subset(genericdata,select = -c(eval(as.name(paste0(idfieldname)))))

features <- c(catcols,v)
output_vector <- data_noid[,glue({var})]
# encode the categorical variables and create a matrix for the xgboost training
modelmat <- hashing(df = data_noid, features = features)

saveRDS(features,"./../ml_vol/model/artifacts/features.rds")

return(list(modelmat,varr,data_withid,output_vector))
  
}

#head(preprocessing(genericdata = genericdata, dataschema = dataschema)[[1]])
