# Practical Work 3

## General

- Authors: Fabien Léger & Mauro Santos
- Course: ARN, HEIG-VD
- Teaching staff: Professor Stephan Robert & Assistant Yasaman Izadmehr
- Date: 27.03.2026

## Information

### Files

- EEG_mouse_data_1.csv - mouse number 1
- EEG_mouse_data_2.csv - mouse number 2
- EEG_mouse_data_test.csv - mouse number 3 (only for testing part 3)
- ...

### Guidelines

- 25 features
- Normalize data
- 3-fold cross validation (3 different instances with validation in each)
- Plot training and validation loss 
- Confusion matrix
- F1-score (of each class and ‘micro’, there is a parameter for « micro », check sklearn) 

## Part 1 - Separation awake/sleep

### Model

Firstly, we decided to group the two mice together to have a bigger dataset and mix the data more.

In this first part, we only classify mice states between awake and non-awake. We can do this by grouping n-rem and rem
sleep stages into "sleep" and have "awake" be the other group. Thanks to that, we can use a single neuron as output.

- output = 1 => 'w'
- output = 0 => 's'

We can then use the sigmoid function to normalize our data between 0 and 1. Then, a basic threshold at 0.5 to separate
our results obtained through the output neuron.

## Part 3 - Competition

### Ideas

- Batch
- More hidden layers