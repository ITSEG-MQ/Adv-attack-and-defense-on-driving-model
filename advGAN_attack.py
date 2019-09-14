import numpy as np 
np.random.seed(0)
import torch
torch.manual_seed(0)
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN.advGAN import AdvGAN_Attack
from model import BaseCNN, build_vgg16, Nvidia, Vgg16
from data import UdacityDataset, Rescale, Preprocess, ToTensor
from advGAN.advGAN_Uni import AdvGAN_Uni_Attack

def advGAN_Attack(model_name, target_model_path, target, train_dataset, universal=False):
    image_nc=3
    epochs = 60
    batch_size = 64
    BOX_MIN = 0
    BOX_MAX = 1
    # target = 0.2
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    image_size = (128, 128)

    if 'baseline' in target_model_path:
        targeted_model = BaseCNN().to(device)
        #image_size = (128, 128)
    elif 'nvidia' in target_model_path:
        targeted_model = Nvidia().to(device)
        #image_size = (66, 200)
    elif 'vgg16' in target_model_path:
        targeted_model = Vgg16().to(device)
       #image_size = (224, 224)

    targeted_model.load_state_dict(torch.load(target_model_path))
    targeted_model.eval()

    if not universal:
        advGAN = AdvGAN_Attack(
                            device,
                            targeted_model,
                            model_name,
                            target,
                            image_nc,
                            BOX_MIN,
                            BOX_MAX)
    else:
        advGAN = AdvGAN_Uni_Attack(
                                device,
                                targeted_model,
                                model_name,
                                image_size,
                                target,
                                image_nc,
                                BOX_MIN,
                                BOX_MAX)
    advGAN.train(train_dataset, epochs)
    return advGAN
if __name__ == "__main__":
    # advGAN_Attack('baseline',0.3,univeral=True)
    pass