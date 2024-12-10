# Hadoop-based platform for analysis and classification of streaming Reddit data

## Table of Contents
- [Description](#description)
- [Requirement](#requirement.txt)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description
HTwitt is a Hadoop-based platform designed to analyze and visualize Reddit data in real-time and batch modes. The framework addresses the challenges posed by the volume, variety, and velocity of big data by leveraging machine learning techniques such as Naive Bayes classification and n-Gram models.
HTwitt achieves a 94.28% accuracy in classifying tweets related to natural disasters like landslides. The platform integrates scraped data for data ingestion and decision-making, ensuring a comprehensive approach to disaster management, sentiment analysis, and other big data applications.
Key features of HTwitt include:

Real-time and batch data processing capabilities
Scalable and efficient performance, as highlighted by experimental results
Integration of machine learning algorithms for high-accuracy classification
Seamless data ingestion from scrapped sources
Versatile analytics and visualization capabilities

Future developments for HTwitt aim to incorporate deep learning models and expand its cross-platform capabilities, further enhancing its value in the field of big data analysis and disaster management.

## Requirement
- In requirement.txt you should add following libraries
    - praw
    - pyspark
    - hdfs
    - datetime
    - nltk
    - matplotlib
    - sklearn
    - pyspark
    - json
    
## Installation
IF Apache Pyspark, hdfs and nltk is not installed in the system, install it from the internet and do the required setup

## Dataset
We have used the Nasa_landslide_analysis to train the model (Classifier)
Also, the data scrapped from the Reddit using Reddit API to classify the posts using trained classifier.

## Usage
To run the code, follow the below process.
1) To Collect the data from the reddit -  Run -> python3 main.py
Using the above step the data will be scrapped from the Reddit API and will also store in the hadoop system.

2) To view if the parsed data is successfully stored in the hdfs or not -  Run -> python3 verify_hadoop_data.py
This step will check if the data is stored in the hdfs or not and also will display the format of the data stored.

3) To retrieve the data from the hadoop system and use it for the downstream task (Classification) - Run -> python3  grouped.py
This step will take out the data from hdfs in json form and then will use it to the classification task after some preprocessing

4) To perform the classification-  Run -> python3 classifier.py
This step will do the classification using naive-bayes with different set of hyperparameters

5) To perform the classificatoin (improvement using DL) - Run ->python3 classifier_dl.py
This step will do the classification using Deep learning methods. 

## License
No copyright.
