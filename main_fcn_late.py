from models.fully_convolutional_network_1d_late import FullyConvolutionalNetwork1DLate
from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump', 'EigenWorms'], 
    datasets= ['BasicMotions'],
    seeds= list(range(1, 11)),
    used_model = FullyConvolutionalNetwork1DLate,
    is_multimodal=True
)