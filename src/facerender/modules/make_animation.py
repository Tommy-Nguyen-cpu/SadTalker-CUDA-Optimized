from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from time import time

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}


# TODO: Okay, so I found a workaround that would allow me to convert the generator to tensorrt, but the model now outputs a black screen.
# TODO: This is likely due to some issue with tensorrt/the format of the data. Will have to dig into it.
def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False):
    with torch.no_grad():
        predictions = []
        # generator = generator.half() if use_half else generator

        start = time()
        kp_canonical = np.empty((4, 15, 3), dtype = np.float32)
        input_image = source_image.cpu().numpy().astype(np.float32)
        input_image = np.concatenate((input_image, input_image), axis=0)
        kp_canonical = kp_detector(input_image, kp_canonical)
        kp_canonical = kp_canonical.to("cuda")
        kp_canonical = {"value" : kp_canonical[2:4, :, :]}
        print(f"kp_canonical took: {time() - start}")

        start = time()
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)
        # torch.onnx.export(mapping, args={"input_3dmm" : source_semantics}, f="../scripts/mapping.onnx", export_params=True, opset_version=20)
        # print(f"yaw: {he_source['yaw']}\n pitch: {he_source['pitch']}\n roll: {he_source['roll']}\n t: {he_source['t']}\n exp: {he_source['exp']}")
        print(f"Mapping took: {time() - start}")
        print(f"source semantic shape: {source_semantics.shape}")
        # print(f"Mapping output shape: {he_source.shape}")

        print(f"Running grid sample {target_semantics.shape[1]} times!")
        for frame_idx in tqdm(range(target_semantics.shape[1]), 'Face Renderer:'):
            # still check the dimension
            # print(target_semantics.shape, source_semantics.shape)
            target_semantics_frame = target_semantics[:, frame_idx]
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
            if roll_c_seq is not None:
                he_driving['roll_in'] = roll_c_seq[:, frame_idx] 
            
            kp_driving = keypoint_transformation(kp_canonical, he_driving)
                
            kp_norm = kp_driving
            if use_half:
                source_image = source_image.half()
                kp_source = {k: v.half() for k, v in kp_source.items()}
                kp_norm = {k: v.half() for k, v in kp_norm.items()}
            
            start = time()
            out = np.zeros(source_image.shape, dtype = np.float16)
            print(f"output: {out.shape}")
            print(f"source image: {source_image.shape}")
            print(f"kp_source: {kp_source['value'].shape}")
            print(f"kp_norm: {kp_norm['value'].shape}")
            generator_input = [source_image.cpu().numpy().astype(np.float16), kp_source['value'].cpu().numpy().astype(np.float16), kp_norm['value'].cpu().numpy().astype(np.float16)]
            out = generator(generator_input, out)
            
            # out = generator(source_image, kp_source['value'], kp_norm['value'])
            # if frame_idx == 0:
            #     print(f"Input shape: {source_image.shape}")
            #     torch.onnx.export(generator, args={"source_image" : source_image, "kp_source" : kp_source['value'], "kp_driving" : kp_norm['value']}, f="../scripts/no_sample_grid_generator.onnx", export_params=True, opset_version=20)

            # print(f"Generator took: {time() - start}")
            # torch.onnx.export(generator, args={"source_image" : source_image, "kp_source" : kp_source, "kp_driving" : kp_norm}, f="face_render.onnx", export_params=True, opset_version=20)
            # print("Finished saving!")
            # print(f"Source image shape: {source_image.shape}")
            # break
            '''
            source_image_new = out['prediction'].squeeze(1)
            kp_canonical_new =  kp_detector(source_image_new)
            he_source_new = he_estimator(source_image_new) 
            kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
            kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
            out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
            '''
            predictions.append(out)
            # print(f"out: {out}")
            # print(f"out len: {len(out)}")
        predictions_ts =  torch.stack(predictions, dim=1)
    return predictions_ts

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video