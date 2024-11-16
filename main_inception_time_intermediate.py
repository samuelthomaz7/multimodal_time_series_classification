from models.inception_time_intermediate import InceptionTimeIntermediate
from utils_file import training_nn_for_seeds



training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump', 'EigenWorms'], 
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump'], 
    datasets= ['AtrialFibrillation', 'StandWalkJump', 'RacketSports', 'ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'ArticularyWordRecognition'], 
    seeds= list(range(1, 11)),
    used_model = InceptionTimeIntermediate,
    is_debbug=False,
    num_ensembles=5
)   