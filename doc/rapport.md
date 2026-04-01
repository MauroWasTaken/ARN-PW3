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

## Notes

We explored a bit the data to better fit our algorithms. Here is a list of different steps done.

Firtly, by taking the same graph we did in practical work 1, we can check how the frequencies are distributes. We can
then try to find better faatures than take the first 25.

![frequencies2data.png](images/frequencies2data.png)

Thanks to this graph, it is pretty obvious that there is only a real difference between the first 10 frequencies. It
is probably possible to further reduce the number of features but we're gonna keep it as a good base for now.

## Part 1 - Separation awake/sleep

### Model

Firstly, we decided to group the two mice together to have a bigger dataset and mix the data more.

In this first part, we only classify mice states between awake and non-awake. We can do this by grouping n-rem and rem
sleep stages into "sleep" and have "awake" be the other group. Thanks to that, we can use a single neuron as output.

- output = 1 => 'w'
- output = 0 => 's'

We can then use the sigmoid function to normalize our data between 0 and 1. Then, a basic threshold at 0.5 to separate
our results obtained through the output neuron.

### Performances result

| Idx | Learning rate | Momentum | nb epochs | loss | Nb neurons | F1 (micro) | Notes |
|-----|---------------|----------|-----------|------|------------|------------|-------|
| 0   | 0.1           | 0.8      | 100       | mse  | 8          | 78.59%     |       |
| 1   | 0.01          | 0.9      | 100       | mse  | 16         | 76.79%     |       |
| 2   | 0.001         | 0.8      | 150       | mse  | 32         | 59.61%     |       |
| 3   | 0.5           | 0.8      | 200       | mse  | 8          | 80.33%     |       |
| 4   | 0.2           | 0.8      | 400       | mse  | 8          | 82.63%     |       |
| 5   | 0.2           | 0.8      | 1000      | mse  | 8          | 85.18%     |       |

It seems we get the best results with a high enough learning rate. We can higher the number of epoch to have a
better f1 score. This doesn't seem to learn the training data too much so it's not overfitting yet. One thing to
take into account is simply that it's not viable for now to train on such a long period of time.

For the rest of this part, we will use the no5 as our baselines as 85% is already a good start.

### Training history plot

![part1_training](images/part1_training.png)

As we can see, the training went decently well with a gradual descent. The only real problem is the time it takes.
Because of that, it is quite hard to make tests on it. We could have batching to improve that time but this would be
implemented in part 3.

A strange aspect is that after 600 epochs we start to have better results for validation data than training. This
could be because our validation data is easier to guess or we simply learned it through our choice of parameters.

Again, there is a rapid descent for the first 200 epochs but it has a harder time later on. This could be improved
through other ways.

### Analysis of results

With only 2 outputs possible, it is quite easy to implement the code for it. We can now check the confusion matrices
to see better how our model performs.

![part1_confusion_matrix_fold1](images/part1_confusionmatrix_fold1.png)

![part1_confusion_matrix_fold2](images/part1_confusionmatrix_fold2.png)

![part1_confusion_matrix_fold3](images/part1_confusionmatrix_fold3.png)

![part1_confusion_matrix_global](images/part1_confusionmatrix_global.png)

- AccuracySleep​ = 11775 / 16503 ​≈ 0.713 = 71.3%
- AccuracyAwake = 23034 / 24360 ≈ 0.946 = 94.6%

It is quite apparent even in the confusion matrices that when a mouseis awake, we guess mostly right with a 94.6%
accuracy. The problem comes from when it sleeps. We only have a success rate of 71.3% letting about 3/10 guesses
wrong. This is certainly what's pejorating our results.

Ways to improve this model are numerous. The problems range from a slow learning process to quite poor results even
with many epochs and decent parameters. We can hope to fix most of these problems in part3 with new ideas to
implement. It is worrying for part2 results because we only got so much f1 score already. We can predict it could
be lower because we have a new class or that the error will simply divide itself in the new class.

## Part 2 - Separation awake/rem/non-rem

### Model

### Testing

| Exp | Layers | Units     | Activation | Optimizer | LR     | Batch | Epochs | Loss                     | F1 (micro) | Notes  |
|-----|--------|-----------|------------|-----------|--------|-------|--------|--------------------------|------------|--------|
| 0   | 1      | [4]       | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 76.73%     |        |
| 1   | 1      | [16]      | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 83.95%     |        |
| 2   | 1      | [32]      | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.25%     |        |
| 3   | 1      | [64]      | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.55%     |        |
| 4   | 2      | [32, 16]  | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.47%     |        |
| 5   | 2      | [64, 32]  | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.36%     | deeper |
| 6   | 2      | [128, 64] | relu       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.26%     |        |
| 7   | 2      | [16, 32]  | relu       | Adam      | 0.0005 | 32    | 100    | categorical_crossentropy | 84.30%     |        |
| 8   | 2      | [32]      | tanh       | Adam      | 0.001  | 32    | 100    | categorical_crossentropy | 84.21%     |        |

## Part 3 - Competition

### Ideas

- Batch
- More hidden layers