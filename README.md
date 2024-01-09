# Parkinson's Disease classification using the Travelling Model paradigm
<div align="center">

</div>

<p align="center">
<img src="fig1_version4.png?raw=true">
</p>


Implementation for distributed learning using the Travelling Model for a Parkinson's disease classifier that is under review by the Frontiers in Artificial Intelligence: "[A multi-center distributed learning approach for Parkinson's disease classification using the travelling model paradigm] (link:coming soon).

Our code here is based on the investigation of a Parkinson's disease classification using non-identical distribution across 83 centers.

If you find our framework, code, or paper useful to your research, please cite us!
```
@inproceedings{}
```
```
```

### Abstract 
Distributed learning is a promising alternative to central learning for machine learning (ML) model training, overcoming data-sharing problems in healthcare. Previous studies exploring federated learning (FL) or the travelling model (TM) setup for medical image-based disease classification often relied on large databases with a limited number of centers or simulated artificial centers, raising doubts about real-world applicability. This study develops and evaluates a convolution neural network (CNN) for Parkinson’s disease classification using data acquired by 83 diverse real centers around the world, mostly contributing small training samples. Our approach specifically makes use of the TM setup, which has proven effective in scenarios with limited data availability but has never been used for image-based disease classification. Our findings reveal that TM is effective for training CNN models, even in complex real-world scenarios with variable data distributions. After sufficient training cycles, the TM-trained CNN matches or slightly surpasses the performance of the centrally trained counterpart (AUROC of 83\% vs. 80\%). Our study highlights, for the first time, the effectiveness of TM in 3D medical image classification, especially in scenarios with limited training samples and heterogeneous distributed data. These insights are relevant for situations where ML models are supposed to be trained using data from small or remote medical centers, and rare diseases with sparse cases. The simplicity of this approach enables a broad application to many deep learning tasks, enhancing its clinical utility across various contexts and medical facilities. 

## PD classifier centralized baseline
We use the state-of-the-art simple fully convolutional network (SFCN) (doi: 10.1016/J.MEDIA.2020.101871) as our deep learning architecture. The Adam optimizer with an initial learning rate of 0.001, an exponential decay after every epoch, and batch size 5 was used during training. The best model (lowest binary cross entropy testing loss) was saved for evaluation based on early stopping with patience of 10 epochs. 
The code used is in: 
```bash
├── centralized_pd.py

```
To run this code, you will need to change the params variable to match the size of your image data and to read the correct column for your labels. After you update the file you can save and run:
```
python centralized_pd.py -fn_train ./training_set.csv -fn_test ./testing_set.csv -model_name ./best_model_pd
```

## Travelling model - distributed learning
Our travelling model code used the same network architecture with two modifications: 
1. Training takes place at one center at a time based on the travelling order (you need to change the code if you want fixed order comment-out lines 128 and 129)
2. LR was set to 0.0001 because some centers have less than five MRI scans available for training.
```bash
├──distributed_learning.py

```
To run this code, you will need to change the params variable to match the size of your image data and to read the correct column for your labels. After you update the file you can save and run:
```
python distributed_learning.py -s ./best_model -c 30
```


## Evaluation
The code used for evaluation is in: 
```bash
├──inference_pd.py

```

## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090

Full environment in `requirements.txt`.


## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=PD-travelling-model).
