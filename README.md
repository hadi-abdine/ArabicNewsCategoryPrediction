# Arabic News Category Prediction

***Aurhors: Hadi Abdine, Ahmad Chamma, Youssef Farhat.***

In this project we are going to be looking into Arabic Natural Language Processing (ANLP). Arabic is the fifth most talked language in the world and is considered as the official language in 26 countries which qualifies it to be an important langauge. Because of that, a lot of textual data written in arabic language is generated whether it's from socail media, a news website or google searches etc... Also, Arabic in its standard form, known as Modern Standard Arabic (MSA), is one of the 6 official languages of the United Nations.

Despite its importance and significance in the world, it has yet to have a major breakthrough in the NLP applications. Arabic has received comparatively little attention in modern computational linguistics. It is indeed a challenging topic due to the complexity of the language, its rich morphology, as well as the presence many different dialects. But with the constant advancement in the NLP field, as well as deep learning and big data, and with proper research and dedication, the creation of a great ANLP model will be considered a breakthrough in the computational liguistics.

What we proposed in this challenge is to develop a prediction system to predict the category of the news of an arabic website using an nlp model, whether it's local news, international, sports related etc... A simple way to benefit from this is to use it, for example, to categorize the huge amount of data coming from Twitter and thus it will be much easier for a client to find the news related to his category in mind.

#### Getting Started

1. Set up the workflow environment by installing the ramp-workflow library (if not already done) with the following command:
```
$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
```
For more information on the [RAMP](http:www.ramp.studio) ecosystem go to
[`ramp-worflow`](https://github.com/paris-saclay-cds/ramp-workflow).

2. Get started on this RAMP challenge with the [dedicated notebook](ArabicNewsCategoryPrediction_starting_kit.ipynb)

- to test the starting kit submission (`submissions/starting_kit`), run the command:
```
ramp_test_submission --submission starting_kit
```
- to test for example `random_forest` any other submission in `submissions`, run the command:
```
ramp_test_submission --submission random_forest
```