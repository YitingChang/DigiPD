#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Train and Deploy a Model
# https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-python-sdk/scikit_learn_randomforest/Sklearn_on_SageMaker_end2end.ipynb


# In[1]:


import datetime
import time
import tarfile

import boto3
import pandas as pd
import numpy as np
import pickle

import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn


sm_boto3 = boto3.client("sagemaker")

sess = sagemaker.Session()

region = sess.boto_session.region_name

bucket = sess.default_bucket()  # this could also be a hard-coded bucket name

role = get_execution_role()

print("Using bucket " + bucket)


# ## Prepare data

# In[ ]:


# Option 1: Load data from S3
subfolder = ''
conn = boto3.client('s3')
contents = conn.list_objects(Bucket=bucket, Prefix=subfolder)['Contents']

# Get train data
my_file = 'X_train_is_severe_tremor.pkl'
s3client = boto3.client('s3')
response = s3client.get_object(Bucket=bucket, Key=my_file)
body = response['Body']
df_X_train = pickle.loads(body.read())
# Get test data
my_file = 'X_test_is_severe_tremor.pkl'
response = s3client.get_object(Bucket=bucket, Key=my_file)
body = response['Body']
df_X_test = pickle.loads(body.read())

# Save train and test data
df_X_train.to_csv("X_train_is_severe_tremor.csv")
df_X_test.to_csv("X_test_is_severe_tremor.csv")


# In[2]:


# Option 2: Load data directly from SageMaker
df_X_train = pd.read_csv("X_train_is_severe_tremor.csv") 
df_X_test = pd.read_csv("X_test_is_severe_tremor.csv") 


# In[ ]:


df_X_train.head()


# In[ ]:


df_X_test.head()


# In[3]:


# Get feature names for training
feature_names = list(df_X_train.columns[1:-1])
all_feature_names = feature_names[0]
for i, name in enumerate(feature_names[1:]):
    all_feature_names = all_feature_names + " " + name


# In[4]:


print(feature_names)


# In[5]:


print(all_feature_names)


# In[ ]:


# Send data to S3. SageMaker will take training data from S3
trainpath = sess.upload_data(
    path="X_train_is_severe_tremor.csv", bucket=bucket, key_prefix="sagemaker/sklearncontainer"
)

testpath = sess.upload_data(
    path="X_test_is_severe_tremor.csv", bucket=bucket, key_prefix="sagemaker/sklearncontainer"
)


# In[ ]:


print(trainpath)
print(testpath)


# In[ ]:





# In[ ]:


# Get feature names
all_feature_names = 'x__absolute_sum_of_changes x__autocorrelation__lag_2 x__autocorrelation__lag_3 x__change_quantiles__f_agg_mean__isabs_True__qh_06__ql_00 x__change_quantiles__f_agg_mean__isabs_True__qh_08__ql_00 x__change_quantiles__f_agg_mean__isabs_True__qh_10__ql_00 x__cid_ce__normalize_False x__fft_aggregated__aggtype_skew x__mean_abs_change x__number_crossing_m__m_1 x__partial_autocorrelation__lag_4 z__absolute_sum_of_changes z__autocorrelation__lag_2 z__autocorrelation__lag_3 z__autocorrelation__lag_4 z__autocorrelation__lag_5 z__cid_ce__normalize_False z__fft_aggregated__aggtype_skew z__mean_abs_change z__partial_autocorrelation__lag_4 age_group_40 age_diagnosis_group_20 tremor_No disrupted_sleep_No'


# In[ ]:


# Get train and test file paths
trainpath = "s3://sagemaker-us-east-1-856191922375/sagemaker/sklearncontainer/X_train_is_severe_tremor.csv"
testpath = "s3://sagemaker-us-east-1-856191922375/sagemaker/sklearncontainer/X_test_is_severe_tremor.csv"


# ## Write a Script Mode script

# In[ ]:


get_ipython().run_cell_magic('writefile', 'script_is_severe_tremor.py', '\nimport argparse\nimport joblib\nimport os\n\nimport numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import recall_score, precision_score, f1_score\n\ndef model_fn(model_dir):\n    clf = joblib.load(os.path.join(model_dir, "model.joblib"))\n    return clf\n\nif __name__ == "__main__":\n\n    print("extracting arguments")\n    parser = argparse.ArgumentParser()\n\n    # hyperparameters sent by the client are passed as command-line arguments to the script.\n    # to simplify the model I don\'t use all sklearn LogisticRegression hyperparameters\n    parser.add_argument("--penalty", type=str, default=\'l1\')\n    parser.add_argument("--solver", type=str, default=\'liblinear\')\n    parser.add_argument("--C", type=int, default=0.75)\n    parser.add_argument("--max_iter", type=int, default=50)\n\n    # Data, model, and output directories\n    parser.add_argument(\'--output-data-dir\', type=str, default=os.environ.get(\'SM_OUTPUT_DATA_DIR\'))\n    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))\n    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))\n    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))\n    parser.add_argument("--train_file", type=str, default="X_train_is_severe_tremor.csv")\n    parser.add_argument("--test_file", type=str, default="X_test_is_severe_tremor.csv")\n    parser.add_argument(\n        "--features", type=str\n    )  # in this script we ask user to explicitly name features\n    parser.add_argument(\n        "--target", type=str\n    )  # in this script we ask user to explicitly name the target\n    \n\n    args, _ = parser.parse_known_args()\n\n    print("reading data")\n    train_df = pd.read_csv(os.path.join(args.train, args.train_file))\n    test_df = pd.read_csv(os.path.join(args.test, args.test_file))\n\n    print("building training and testing datasets")\n    X_train = train_df[args.features.split()]\n    X_test = test_df[args.features.split()]\n    y_train = train_df[args.target]\n    y_test = test_df[args.target]\n    \n\n    # train\n    print("training model")\n    model = LogisticRegression(\n        random_state=7, solver=args.solver, penalty=args.penalty, C=args.C, max_iter=args.max_iter\n    )\n    \n    model.fit(X_train, y_train)\n\n    # print performance metrics\n    print("validating model")\n    y_pred = model.predict(X_test)\n    F_score = f1_score(y_test, y_pred).round(2)\n    Precision = precision_score(y_test, y_pred, average = None).round(2)\n    Recall = recall_score(y_test, y_pred, average = None).round(2)  \n    print("F1=" + str(F_score))\n    print("Recall=" + str(Recall[1]))\n    print("Precision=" + str(Precision[1]))\n\n    # persist model\n    path = os.path.join(args.model_dir, "model.joblib")\n    joblib.dump(model, path)\n    print("model persisted at " + path)')


# ## SageMaker Training

# In[ ]:


# Use the Estimator from the SageMaker Python SDK
from sagemaker.sklearn.estimator import SKLearn

FRAMEWORK_VERSION = "0.23-1"

sklearn_estimator = SKLearn(
    entry_point="script_is_severe_tremor.py",
    role=get_execution_role(),
    instance_count=1,
    instance_type="ml.c5.xlarge",
    framework_version=FRAMEWORK_VERSION,
    base_job_name="lg-scikit",
    hyperparameters={
        "penalty": 'l1',
        "solver": 'liblinear',
        "C": 0.75,
        "max_iter": 50,
        "features": all_feature_names,
        "target": "target",
    },
)


# In[ ]:


# Train my estimator
sklearn_estimator.fit({"train": trainpath, "test": testpath}, wait=False)


# ## Deploy to a real-time endpoint

# In[ ]:


# Deploy my estimator to a SageMaker Endpoint and get a Predictor
predictor = sklearn_estimator.deploy(instance_type='ml.c5.large',
                                     initial_instance_count=1)


# In[ ]:


print(predictor.endpoint)


# # Evaluate the model

# In[ ]:


# Invoke with the Python SDK
from sklearn.metrics import recall_score, precision_score, f1_score
X_test = df_X_test[feature_names]
y_test = df_X_test["target"]
y_pred = predictor.predict(X_test)
F_score = f1_score(y_test, y_pred).round(2)
Precision = precision_score(y_test, y_pred, average = None).round(2)
Recall = recall_score(y_test, y_pred, average = None).round(2)  
print("F1=" + str(F_score))
print("Recall=" + str(Recall[1]))
print("Precision=" + str(Precision[1]))


# In[ ]:


print(y_pred)


# In[ ]:


X_test


# In[ ]:


print(X_test.iloc[2532])


# In[ ]:


print(X_test.iloc[2532].values)


# ## Delet the endpoint when it is finished

# In[ ]:


sm_boto3.delete_endpoint(EndpointName=predictor.endpoint)


# In[ ]:





# In[ ]:





# In[ ]:





# # Alternative: Launching a training with boto3

# In[ ]:


# first compress the code and send to S3

source = "source.tar.gz"
project = "scikitlearn-train-from-boto3-script3"

tar = tarfile.open(source, "w:gz")
tar.add("script3.py")
tar.close()

s3 = boto3.client("s3")
s3.upload_file(source, bucket, project + "/" + source)


# In[ ]:


from sagemaker import image_uris

FRAMEWORK_VERSION = "0.23-1"

training_image = image_uris.retrieve(
    framework="sklearn",
    region=region,
    version=FRAMEWORK_VERSION,
    py_version="py3",
    instance_type="ml.c5.xlarge",
)
print(training_image)


# In[ ]:


# launch training job

response = sm_boto3.create_training_job(
    TrainingJobName="sklearn-boto3-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    HyperParameters={
        "penalty": 'l1',
        "solver": 'liblinear',
        "C": "0.75",
        "max_iter": "50",
        "sagemaker_program": "script3.py",
        "features": all_feature_names,
        "target": "target",
        "sagemaker_submit_directory": "s3://" + bucket + "/" + project + "/" + source,
    },
    AlgorithmSpecification={
        "TrainingImage": training_image,
        "TrainingInputMode": "File",
#         "MetricDefinitions": [
#             {"Name": "median-AE", "Regex": "AE-at-50th-percentile: ([0-9.]+).*$"},
#         ],
    },
    RoleArn=get_execution_role(),
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": trainpath,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
        {
            "ChannelName": "test",
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": testpath,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        },
    ],
    OutputDataConfig={"S3OutputPath": "s3://" + bucket + "/sagemaker-sklearn-artifact/"},
    ResourceConfig={"InstanceType": "ml.c5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10},
    StoppingCondition={"MaxRuntimeInSeconds": 86400},
    EnableNetworkIsolation=False,
)

print(response)


# # Alternative: Deploy with Python SDK using s3 artifacts

# In[ ]:


sklearn_estimator.latest_training_job.wait(logs="None")

sklearn_estimator.latest_training_job.wait(logs="None")
artifact = sm_boto3.describe_training_job(
    TrainingJobName=sklearn_estimator.latest_training_job.name
)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifact persisted at " + artifact)


# In[ ]:


from sagemaker.sklearn.model import SKLearnModel

model = SKLearnModel(
    model_data=artifact,
    role=get_execution_role(),
    entry_point="script.py",
    framework_version=FRAMEWORK_VERSION,
)
predictor = model.deploy(instance_type="ml.c5.large", initial_instance_count=1)


# # Alternative: Evaluate the model/ Invoke with boto3

# In[ ]:


runtime = boto3.client("sagemaker-runtime")


# In[ ]:


# csv serialization
response = runtime.invoke_endpoint(
    EndpointName=predictor.endpoint,
    Body=df_X_test[feature_names].to_csv(header=False, index=False).encode("utf-8"),
    ContentType="text/csv",
)

print(response["Body"].read())


# In[ ]:


# Test json input (for Postman demo)
# Postman
# Useful csv to json convertor 
# https://www.convertcsv.com/csv-to-json.htm


# In[ ]:


# API URL: https://glk8nycew8.execute-api.us-east-1.amazonaws.com/test/


# In[18]:


j_input2= {
   "x__absolute_sum_of_changes":[1488.69349,761.29459],
   "x__autocorrelation__lag_2":[0.332961406,0.336792538],
   "x__autocorrelation__lag_3":[0.056601148,-0.05283795],
   "x__change_quantiles__f_agg_mean__isabs_True__qh_06__ql_00":[0.777561198,0.404807666],
   "x__change_quantiles__f_agg_mean__isabs_True__qh_08__ql_00":[0.791964439,0.425873892],
   "x__change_quantiles__f_agg_mean__isabs_True__qh_10__ql_00":[0.902785622,0.461670461],
   "x__cid_ce__normalize_False":[48.42932994,23.63544937],
   "x__fft_aggregated__aggtype_skew":[0.905069722,1.140701363],
   "x__mean_abs_change":[0.902785622,0.461670461],
   "x__number_crossing_m__m_1":[196,34],
   "x__partial_autocorrelation__lag_4":[-0.144102058,-0.269839603],
   "z__absolute_sum_of_changes":[1791.55672,707.07886],
   "z__autocorrelation__lag_2":[0.117929966,0.353439424],
   "z__autocorrelation__lag_3":[-0.129876771,-0.060176124],
   "z__autocorrelation__lag_4":[-0.204337165,-0.388713367],
   "z__autocorrelation__lag_5":[-0.231590606,-0.598858816],
   "z__cid_ce__normalize_False":[56.63622761,22.92053236],
   "z__fft_aggregated__aggtype_skew":[0.653764017,1.011008381],
   "z__mean_abs_change":[1.086450406,0.428792517],
   "z__partial_autocorrelation__lag_4":[-0.384899525,-0.418456489],
   "age_group_40":[0,0],
   "age_diagnosis_group_20":[1,1],
   "tremor_No":[0,0],
   "disrupted_sleep_No":[1,1]
}


# In[19]:


pdObj = pd.DataFrame.from_dict(j_input2)


# In[28]:


runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="lg-scikit-2022-08-13-21-27-41-457",
    Body=pdObj[feature_names].to_csv(header=False, index=False).encode("utf-8"),
    ContentType="text/csv",
)
results = json.loads(response['Body'].read().decode())
predicted_labels = []
for i,result in enumerate(results):
    pred=int(result)
    predicted_label = 'sever tremor' if pred == 1 else 'no sever tremor'
    print(predicted_label)
    predicted_labels.append(predicted_label) 


# In[ ]:




