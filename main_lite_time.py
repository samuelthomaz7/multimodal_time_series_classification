from models.lite import LITE
from utils_file import training_nn_for_seeds



training_nn_for_seeds(
    # datasets= ['ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS', 'RacketSports', 'AtrialFibrillation', 'StandWalkJump'], 
    datasets=['ACSF1', 'AtrialFibrillation', 'StandWalkJump', 'RacketSports', 'ArticularyWordRecognition', 'BasicMotions', 'Cricket', 'NATOPS'],
    seeds= list(range(1, 11)),
    used_model = LITE,
    is_debbug=False,
    num_ensembles=5
)