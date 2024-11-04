from models.fully_convolutional_network_1d_intermediate import FullyConvolutionalNetwork1DIntermediate
from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'AtrialFibrillation', 'StandWalkJump', 'EigenWorms'], 
    datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'AtrialFibrillation', 'StandWalkJump', 'EigenWorms'],
    seeds= list(range(1, 11)),
    used_model = FullyConvolutionalNetwork1DIntermediate,
    is_multimodal=True
)