# Pattern-Recognition-and-Machine-Learning
CS5691 at IIT Madras


## Assignments

The data for the assignments are available in the assignment folder and the questions along with the coded solution is available in the notebook. 

## Hackathon

The goal of the hackathon was to build a recommendation system that reccomends biking tours to bikers ranked according to their preference. The data for the hackathon can be obtained on the [Kaggle competition page](https://www.kaggle.com/c/prml-data-contest-nov-2020/data) or can be installed using the Kaggle API command `kaggle competitions download -c prml-data-contest-nov-2020`. 

The code provided for the hackthon uses catboost and light-gbm to compute the probability of biker liking a tour and an ensemble is used to obtain the final predicitons. Mean Average Precision @k is used as the scoring method. The code provided results in a score of 0.718 on the private leaderboard.
