

from utils_file import training_nn_for_seeds


training_nn_for_seeds(
    datasets= ['ArticularyWordRecognition'], # type: ignore
    seeds= list(range(1, 11)),
    used_model = None
)