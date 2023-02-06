# Data agnostic Random Forest Regression (binary prediction) with docker compose and Plumber API  

* The idea of this project is to have a data agnostic random forest logistic regression that trains, predicts and serves as a docker compose service.
 So basically, the use just needs to attach the folder with the data with a binary response variable then run train script to train the model and save it as an artifact back to the attached volume, test the model to save the predictions as an output in the attached volume and run the serve script to run a plumber API which returns a json output of predictions. 
 
* The volume to be attached contains the csv files and the data schema.    

To try this locally on your pc see the steps below: 

1. Your data folder   
Note: ensure you name it : *ml_vol*      
should have your files with the following pattern     
```
*_train.csv
*_test.csv
*_schema.json 

```
    

1. Edit the docker compose yml file below accordingly(path to your data directory). Put it in the project directory. 
```
version: "3"
services:
  rforest_regressor:
    build: .
    volumes:
      - /path/to/data/dir/ml_vol:/modellingLogistic/ml_vol
    ports:
      - 8000:8000
    working_dir: /modellingLogistic
    command: tail -f /dev/null # keep the container running

```

2. Navigate to your project directory and run the following command.   

The command below starts your service. (the tag _-d_ is for running it in detached mode.)   

```
docker compose up -d

``` 

3. Running your script inside the container.    

```
docker compose exec -it rforest_regressor ./train

```
Note: logimodel is the name of the service in docker compose yml.    

To test the model you'll just replace ./train with ./test   

4. Starting a web server 

```
docker compose exec -it rforest_regressor ./serve 
```

Once the plumber API starts you can now open another terminal and paste the following: 

```
curl localhost:8000/infer --header "Content-Type: application/json" \
  --request POST \
  --data @/path/to/your/data/testjsn.json
```

_*Note: ensure you change the path to data accordingly!*_