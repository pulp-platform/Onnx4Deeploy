import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.inject.enum import FloatToIntImplType
from brevitas.quant.scaled_int import (
    Int8ActPerTensorFloat,
    Int32Bias,
    Int8WeightPerTensorFloat,
    Int8WeightPerChannelFloat
)
from brevitas.core.function_wrapper.stochastic_round import StochasticRoundSte
    
class StochasticInt8WeightPerChannelFloat(Int8WeightPerChannelFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class StochasticInt8WeightPerTensorFloat(Int8WeightPerTensorFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class StochasticInt8ActPerTensorFloat(Int8ActPerTensorFloat):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class StochasticInt32Bias(Int32Bias):
    float_to_int_impl_type=FloatToIntImplType.STOCHASTIC_ROUND

class QLiteCNN(nn.Module):
    def __init__(self,   
                batch_size: int = 1,
                input_channels: int = 1,
                num_classes: int = 10, 
                dropout: float = 0.0):# ignored
    
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.fc_channels = 160  # Fixed: 10 * 4 * 4 = 160

        self.convAndLinQuantParams = {
            "bias": True,
            "weight_bit_width": 8,
            "bias_quant": Int32Bias,
            "input_quant": Int8ActPerTensorFloat,
            "weight_quant":Int8WeightPerChannelFloat, #no channel wise support in deeploy yet.
            #"weight_quant":Int8WeightPerTensorFloat,
            "output_quant": None,
            "return_quant_tensor": True
        }
        
        self.convAndLinQuantParamsOut = {
            "bias": True,
            "weight_bit_width": 8,
            "bias_quant": Int32Bias,
            "input_quant": Int8ActPerTensorFloat,
            "weight_quant":Int8WeightPerChannelFloat,# no channel wise support in deeploy yet.
            #"weight_quant": Int8WeightPerTensorFloat,
            "output_quant": Int8ActPerTensorFloat,
            "return_quant_tensor": True
        }
        super(QLiteCNN, self).__init__()
        # Convolutional layers
        # self.inputQuant = qnn.QuantIdentity(
        #     act_quant=Int8ActPerTensorFloat, return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(
                    in_channels=input_channels,
                    out_channels=20,
                    kernel_size=(5,5),
                    **self.convAndLinQuantParams
        )
        self.relu1 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Output: (20, 12, 12)
        self.conv2 = qnn.QuantConv2d(20,
                                     10,
                                     kernel_size=(1, 1), 
                                     **self.convAndLinQuantParams)
        # Output: (10, 12, 12)
        self.relu2 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.pool2 = nn.MaxPool2d(kernel_size=2)  # Output: (10, 6, 6)

        self.conv3 = qnn.QuantConv2d(10, 12, kernel_size=(3, 3),
                                    **self.convAndLinQuantParams) # Output: (12, 4, 4)
        
        self.relu3 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.conv4 = qnn.QuantConv2d(12, 10, kernel_size=(1, 1),
                                     **self.convAndLinQuantParams)
        # Output: (10, 4, 4)
        self.relu4 = qnn.QuantReLU(bit_width=8, return_quant_tensor=True)

        self.fc = qnn.QuantLinear(self.fc_channels, num_classes,
                                    **self.convAndLinQuantParamsOut)  # Output: num_classes

    def forward(self, x):

        # Convolutional layers with ReLU activation and pooling
        # compute min and max of input  and scale for quantization debugging
        # if isinstance(x, torch.Tensor):
        #     print(f"Input tensor shape: {x.shape}, dtype: {x.dtype}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        # print(f"After input quantization: shape: {x.shape}, dtype: {x.dtype}, min: {x.min().item():.4f}, max: {x.max().item():.4f}")
        # print(f"scale of input quantizer: {self.inputQuant.act_quant.scale().item():.6f}")
        # x = self.inputQuant(x)
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.pool1(x)  # Output: (20, 12, 12)
        x = self.conv2(x)
        x = self.relu2(x)

        x = self.pool2(x)  # Output: (10, 6, 6)
        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)  # Output: (10, 4, 4)
        x = self.relu4(x)

        # Flatten the feature map
        x = x.flatten(start_dim=1)  # Flatten to (batch_size, 10 * 4 * 4)
        # Fully connected layer
        x = self.fc(x)

        return x
