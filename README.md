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
- Create a clean, organized, reproducible, end-to-end machine learning pipeline from scratch using MLflow
- Clean and validate the data using pytest 
- Track experiments, code, and results using GitHub and Weights & Biases
- Select the best-performing model for production
- Deploy a model using MLflow.

### Course Project: [Build ML Pipeline](https://github.com/edwards158/nd0821-c2-build-model-workflow-starter) &nbsp;
Write a Machine Learning Pipeline to solve the following problem: a property management in New York needs to estimate the typical price for a given property based on the price of similar properties. The company receives new data in bulk every week, so the model needs to be retrained with the same cadence, necessitating a reusable pipeline. Write an end-to-end pipeline covering data fetching, validation, segregation, train and validation, test, and release. Run it on an initial data sample, then re-run it on a new data sample simulating a new data delivery.

## Course 3

### Building a Reproducible Model Workflow
This course teaches students how to deploy a machine learning model into production. En route to that goal, students will learn how to put the finishing touches on a model by taking a fine-grained approach to model performance, checking bias and ultimately writing a model card. Students will also learn how to version control their data and models using Data Version Control (DVC). In the last piece of preparation for deployment, students will learn Continuous Integration and Continuous Deployment accomplished
using GitHub Actions and Heroku. Finally, students will learn how to write a fast, type-checked and autodocumented API using FastAPI.

Learning outcomes:
- Performance Testing and Preparing a Model for Production
- Data and Model Versioning
- CI/CD
- API Deployment with FastAPI

### Course Project: [Deploying a Scalable ML Pipeline in Production](https://github.com/edwards158/fastapi-heroku) &nbsp;
Deploy a machine learning model on Heroku. Use Git and DVC to track code, data and model while developing a simple classification model on the Census Income Data Set. After creating the model, the finalize the model for production by checking its performance on slices and writing a model card encapsulating key knowledge about the model. Implement a Continuous Integration and Continuous Deployment framework and ensure pipeline passes a series of unit tests before deployment. Lastly, an API will be written using FastAPI and tested locally. After successful deployment, the API will be tested live using the requests module.











