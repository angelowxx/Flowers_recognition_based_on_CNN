##### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Fast Networks Track

##### What are the number of learnable parameters in your model?
Fast Networks Track - 99 448
Large Networks Track - N/A

##### Briefly describe your approach
Fast Networks Track - I have applied data augmentation, cross validation, early stop. I also tried to apply multiple convolution kernel in one layer, each with different receptive field to try to learn different spacial feature
Large Networks Track - N/A

##### Command to train your model from scratch
Fast Networks Track - python -m src.main -m FastCNN
Large Networks Track - N/A

##### Command to evaluate your model
Fast Networks Track - python -m src.evaluate_model -m FastCNN -p augmented_model -d resize_to_128x128
Large Networks Track - N/A