

from models.multi_layer_perceptron import MultiLayerPerceptron
from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    datasets= ['ArticularyWordRecognition'], 
    seeds= list(range(1, 11)),
    used_model = MultiLayerPerceptron
)