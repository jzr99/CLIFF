import numpy as np
import torch
from models.smpl import SMPL
from common import constants

device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

# motion data dir
file_dir = '/media/ubuntu/hdd/CLIFF/test_samples/pair18_basketball18_4/_cliff_hr48_infill.npz'
save_dir = '/media/ubuntu/hdd/CLIFF/test_samples/pair18_basketball18_4/all.npz'
# load motion
results = np.load(file_dir)
# import pdb; pdb.set_trace()

joints_all = []
verts_all = []
kpts_all = []
for person_id in range(2):
    joints_person = []
    verts_person = []
    kpts_person = []
    detection_all = results['detection_all']
    _, idx = torch.sort(torch.tensor(detection_all[:, 0]))
    sorted_detection_all = detection_all[idx]
    sorted_smpl_joints = results['pred_joints'][idx]
    sorted_smpl_2d_kpts = results['kpts'][idx]
    sorted_smpl_verts = results['verts'][idx]
    for i in range(len(sorted_detection_all)):
            frame_id = sorted_detection_all[i][0]
            person = sorted_detection_all[i][-1]
            if person_id == person:
                # choose_frame.append(int(frame_id))
                # choose_index.append(len(choose_index))
                joints_person.append(sorted_smpl_joints[i])
                verts_person.append(sorted_smpl_verts[i])
                kpts_person.append(sorted_smpl_2d_kpts[i])
    joints_person = np.stack(joints_person, axis=0)
    verts_person = np.stack(verts_person, axis=0)
    kpts_person = np.stack(kpts_person, axis=0)
    joints_all.append(joints_person)
    verts_all.append(verts_person)
    kpts_all.append(kpts_person)
joints = np.stack(joints_all, axis=0)
verts = np.stack(verts_all, axis=0)
kpts = np.stack(kpts_all, axis=0)
print("joints", joints.shape)
print("verts", verts.shape)
print("kpts", kpts.shape)
np.savez(save_dir, pj2d_org=kpts, joints=joints, verts=verts)

# pred_betas = torch.from_numpy(results['shape']).float().to(device)
# pred_rotmat = torch.from_numpy(results['pose']).float().to(device)
# pred_cam_full = torch.from_numpy(results['global_t']).float().to(device)

# smpl_model = SMPL(constants.SMPL_MODEL_DIR).to(device)

# load smpl model
# smpl_model = smplx.create(constants.SMPL_MODEL_DIR, "smpl").to(device)
# smpl_model = SMPL(model_path='./data/smpl', gender='MALE', batch_size=1).eval().to(device)
