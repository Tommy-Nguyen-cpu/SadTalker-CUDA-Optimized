import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.facerender.TensorRTWrapper import TensorRTWrapper


class MappingNet(nn.Module):
    def __init__(self, coeff_nc, descriptor_nc, layer, num_kp, num_bins):
        super( MappingNet, self).__init__()

        self.layer = layer
        nonlinearity = nn.LeakyReLU(0.1)

        self.first = nn.Sequential(
            torch.nn.Conv1d(coeff_nc, descriptor_nc, kernel_size=7, padding=0, bias=True))

        for i in range(layer):
            net = nn.Sequential(nonlinearity,
                torch.nn.Conv1d(descriptor_nc, descriptor_nc, kernel_size=3, padding=0, dilation=3))
            setattr(self, 'encoder' + str(i), net)   

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.output_nc = descriptor_nc

        self.fc_roll = nn.Linear(descriptor_nc, num_bins)
        self.fc_pitch = nn.Linear(descriptor_nc, num_bins)
        self.fc_yaw = nn.Linear(descriptor_nc, num_bins)
        self.fc_t = nn.Linear(descriptor_nc, 3)
        self.fc_exp = nn.Linear(descriptor_nc, 3*num_kp)

    def forward(self, input_3dmm):
        out = self.first(input_3dmm)
        for i in range(self.layer):
            model = getattr(self, 'encoder' + str(i))
            out = model(out) + out[:,:,3:-3]
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        #print('out:', out.shape)

        yaw = self.fc_yaw(out)
        pitch = self.fc_pitch(out)
        roll = self.fc_roll(out)
        t = self.fc_t(out)
        exp = self.fc_exp(out)

        return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 't': t, 'exp': exp}

class MappingTensorRT():
    def __init__(self, model_path):
        self.model = TensorRTWrapper()
        if ".onnx" in model_path.lower():
             self.model.create_engine(onnx_file_path=model_path)
             self.model.save_engine("mapping.engine")
        else:
            self.model.load_engine(model_path)

    def __call__(self, input_3dmm):
            yaw = np.zeros((2, 66), dtype=np.float32)
            pitch = np.zeros((2, 66), dtype=np.float32)
            roll = np.zeros((2, 66), dtype=np.float32)
            t = np.zeros((2, 3), dtype=np.float32)
            exp = np.zeros((2, 45), dtype=np.float32)

            outputs = self.model(input_3dmm.cpu().numpy(), [yaw, pitch, roll, t, exp])

            outputs = [output.to("cuda") for output in outputs]
            # print(f"Pred mean: {pred}")
            # print(f"yaw: {outputs[0].shape}")
            # print(f"pitch: {outputs[1].shape}")
            # print(f"roll: {outputs[2].shape}")
            # print(f"t: {outputs[3].shape}")
            # print(f"exp: {outputs[4].shape}")

            return {'yaw': outputs[0], 'pitch': outputs[1], 'roll': outputs[2], 't': outputs[3], 'exp': outputs[4]}