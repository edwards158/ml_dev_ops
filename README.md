# Machine Learning DevOps NanoDegree

This repository contains some my work from the Machine Learning DevOps Nanodegree.
The Machine Learning DevOps Engineer Nanodegree program focuses on the software engineering
fundamentals needed to successfully streamline the deployment of data and machine-learning models
in a production-level environment.  Objectives:
- Implement production-ready Python code/processes for deploying ML models outside of cloud-based environments facilitated by tools such as AWS SageMaker, Azure ML, etc.
- Engineer automated data workflows that perform continuous training (CT) and model validation within a CI/CD pipeline based on updated data versioning
- Create multi-step pipelines that automatically retrain and deploy models after data updates
- Track model summary statistics and monitor model online performance over time to prevent model-degradation


## Course 1

### Clean Code Principles
Develop skills that are essential for deploying production machine learning models. First, you will put your coding best practices on autopilot by learning how to use PyLint and AutoPEP8. Then you will further expand your Git and Github skills to work with teams. Finally, you will learn best practices associated with testing and logging used in production settings to ensure your models can stand the test of time.

### Course Project: [Predict Customer Churn with Clean Code](https://github.com/edwards158/ml_dev_ops/tree/main/proj-customer-churn-clean-code) &nbsp;
Identify credit card customers most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented and tested).

## Course 2

### Building a Reproducible Model Workflow
This course empowers the students to be more efficient, effective, and productive in modern, real-world ML projects by adopting best practices around reproducible workflows. In particular, it teaches the fundamentals of MLops and how to:
- create a clean, organized, reproducible, end-to-end machine learning pipeline from scratch using MLflow
- clean and validate the data using pytest 
- track experiments, code, and results using GitHub and Weights & Biases
- select the best-performing model for production
- deploy a model using MLflow.

### Course Project: [Build ML Pipeline](https://github.com/edwards158/nd0821-c2-build-model-workflow-starter) &nbsp;
Write a Machine Learning Pipeline to solve the following problem: a property management in New York needs to estimate the typical price for a given property based on the price of similar properties. The company receives new data in bulk every week, so the model needs to be retrained with the same cadence, necessitating a reusable pipeline. Write an end-to-end pipeline covering data fetching, validation, segregation, train and validation, test, and release. Run it on an initial data sample, then re-run it on a new data sample simulating a new data delivery.















