
from torchvision import models
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym


class CustomExtractorLL(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomExtractorLL, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        #feature_size = 128
        for key, subspace in observation_space.spaces.items():
            feature_size = 128
            if key in ["proprioception", "task_obs"]:
                
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())

            elif key == "image":
                n_input_channels = subspace.shape[0]  # channel last
                
                
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                    
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                
            elif key == "image_global":
                feature_size = 256
                n_input_channels = subspace.shape[0]  # channel last
                
                
                
                cnn = models.resnet18(pretrained=True)
               
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                    
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
                
            else:
                continue
            

            total_concat_size += feature_size

       
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size


    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            
            
        
        return th.cat(encoded_tensor_list, dim=1)




class CustomExtractorHL(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomExtractorHL, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            feature_size = 128
            
            if key == "valid_actions":
                
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU()) 

            elif key == "task_obs_hl":
               
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU()) 
                
            elif key == "image_global":
                feature_size = 256
                n_input_channels = subspace.shape[0]  # 
                cnn = models.resnet18(pretrained=True)
                
                
                test_tensor = th.zeros([n_input_channels, subspace.shape[1], subspace.shape[2]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                    
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            else:
                continue
            
            total_concat_size += feature_size



       
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size
        

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            
            encoded_tensor_list.append(extractor(observations[key]))

        
        return th.cat(encoded_tensor_list, dim=1)


