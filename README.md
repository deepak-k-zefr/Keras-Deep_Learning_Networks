# Keras-Deep_Learning_Networks

## Deploying your Keras model in the CLOUD
* Create a new project in google cloud
* Enable Google Cloud ML service in API mananger
* Install Google Cloud SDK
* gcloud init 

Put Keras model in cloud and use it from anywhere in the world:
* Export model
  * Create bucket
  ```
  gsutil mb -l us-central1 gs://keras-class-99
  ```
  
  * Upload model into google cloud using cmdline (use unique name for bucket)
  ```
  gsutil cp -R exported_model/* gs://keras-class-99/earnings_v1/
  ```
  
  * Specify ml-engine to create model, specify name (earnings)
  ```
  gcloud ml-engine models create earnings --regions us-central1
  ```
  earnings is now a placeholder
  
 * Create first version of model
 ```
   gcloud ml-engine versions create v1 --model=earnings --origin=gs://keras-class-99/earnings_v1/
 ```
 
 * Try to make a prediction using a sample file: 
```
  gcloud ml-engine predict --model=earnings --json-instances=sample_input_prescaled.json 
```
  
OUTPUT:
EARNINGS
[0.17124266922473907]

## USE CLOUD BASED ML FROM OTHER PLACES
* CREATE CREDENTIALS (api manager--->Service account(role= project-->viewer))
download the credentials and add it to same folder as project
* Provide credentials and Make request to make prediction
