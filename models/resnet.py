from models.nn_model import NNModel


class ResNet(NNModel):

    def __init__(self, dataset_name, train_dataset, test_dataset, metadata, model_name = 'ResNet', random_state=42, device='cuda', is_multimodal=False) -> None:
        super().__init__(dataset_name, train_dataset, test_dataset, metadata, model_name, random_state, device, is_multimodal)