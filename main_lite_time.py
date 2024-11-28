from models.lite import LITE
from utils_file import training_nn_for_seeds



training_nn_for_seeds(
    datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'ArticularyWordRecognition', 'AtrialFibrillation', 'StandWalkJump'], 
    # datasets=['AtrialFibrillation', 'StandWalkJump', 'RacketSports', 'ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'ArticularyWordRecognition'],
    seeds= list(range(1, 11)),
    used_model = LITE,
    is_debbug=True,
    num_ensembles=5
)