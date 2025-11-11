## Goal
Create a simple jupyter notebook at /simple_ml/iris.ipynb that will allow me to demo mlflow logging, automl, and hyperopt. I am a Databricks employee and trying to sell our customer on Databricks, as such, everything will be demoed in Databricks. The outcome of the notebook you build should be an optimal model (trained and tuned by automl and hyperopt) trained on the iris dataset.

## Requirements & Details

* Today's date is Nov 11 2025. All references you use should be recent.
* The spark environment has already been configured and is accessible via "from spark_env import spark" You must use this for spark
    * Note: this environment is Databricks connect so the driver is local but we are live connected to Databricks environment (already configured)
* The source dataset is the traditional ml dataset "iris" and is located in Databricks' Unity Catalog at "main.tomes_gen.iris"
    * The target field is species and it's a multi-class (3) classifier
* The training must be logged with mlfow using autolog
    * mlflow.set_tracking_uri("databricks")
    * mlflow.set_experiment("iris_demo")
    * mlflow.sklearn.autolog()
* Train using sklearn and automl to train classifiers and find the best model
    * The sklearn parameters and fit should all be within 
    ```python
    with mlflow.start_run()
        ...
    ```
    * Use automl and test at least the following models:
        * Support Vector Machine (SVM), Logistic Regressor, DecisionTreeClassifier, RandomForestClassifier
    * be sure to track set parameters for each model
    * An example notebook with a Regressor is at the link below. You can use it as an example
        * https://docs.databricks.com/aws/en/notebooks/source/mlflow/mlflow-quick-start-python-mlflow-3.html
* Use hyperopt to fine tune the hyperparamters in parallel
    * Part two in the notebook example below provides an example of hyperopt implementation
        * https://assets.docs.databricks.com/_extras/notebooks/source/hyperopt-spark-ml.html
* You have access to a context7 tool for looking up documentation and code syntax, use it as necessary to help you with the code for 
    * mlflow 3.6.0
    * sklearn classifiers
    * databricks automl
    * etc