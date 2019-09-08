# GAIN Framework 

GAIN framework allows model to focus on a specific areas of object by changing the attention maps (Grad-CAM).  
The flow of the framework is summarized as follow: 
- First, we should [register](./src/models/gain.py#L33) the `feed-forward` and `backward` of the last convolution layer (or block).
- Model generate the [attention maps](./src/models/gain.py#L98-L131) (Grad-CAM) from `forward features` and backward `features` then normalize it by 
using threshold and `sigmoid` function.
- Now, the attention maps covers almost important information of the object. We want to tell the model those kind of areas
are important for the task. It can be done by [applying attention maps](./src/models/gain.py#L141) into the original image. 
Let imagine that the `masked_image` now is containing useless information. When we [feed](./src/models/gain.py#L143) `masked_image` into the model again, 
we expect that the prediction score is as low as possible. That is the idea of `attention mining` in the paper.
- The losses are computed at [GAINCriterionCallback](./src/callbacks.py#L16)


In this implementation, I select [resnet50](./src/models/gain.py#L11) as the base-model to perform GAIN framework. 
You can change the backbone and it's gradient layer as you want. 

# Train GAIN

```bash
bash bin/train_gain.sh 
``` 