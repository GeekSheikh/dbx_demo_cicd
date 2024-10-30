import os
from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from databricks.sdk.core import Config

# load_dotenv()

## Set the environment variable IS_LOCAL_ENV to "true" if running on local environment
## In VSCode, I do this settings --> terminal.integrated.env --> edit json --> add env vars here
if bool(os.environ.get("IS_LOCAL_ENV")):
    print("Driver is running on local environment")
    
    ## Proxy settings / env vars go here
    
    config = Config(
        profile = "field-eng",
        cluster_id = "1030-174504-b6mprahm"
    )
    spark = DatabricksSession.builder.sdkConfig(config).getOrCreate()
else:
    print("Driver is running on Databricks")
    spark = SparkSession.builder.appName("dbx_demo_cicd").getOrCreate()
    