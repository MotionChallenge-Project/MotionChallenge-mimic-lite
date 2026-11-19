"""MimicLite batched observations, history and author PD commands."""
import json
import re
from pathlib import Path
import torch
from motion_challenge.networks.torch_graph import TensorGraph
from motion_challenge.simulation.tensor_math import normalize, conjugate, multiply, rotate, matrix


def vector(value, names, device):
    if isinstance(value, dict):
        values = []
        for name in names:
            matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
            if len(matches) > 1: raise ValueError("Ambiguous named joint configuration")
            values.append(matches[0] if matches else 0.)
        value = values
    result = torch.as_tensor(value, device=device, dtype=torch.float64)
    return result.expand(len(names)) if result.ndim == 0 else result


def heading(q):
    m = matrix(q)
    x, z = m[..., :, 0], m[..., :, 2]
    direction = torch.where((torch.linalg.vector_norm(x[..., :2], dim=-1) > .1)[..., None],
                            x[..., :2], z[..., :2]*torch.where(x[..., 2:3] < 0, 1., -1.))
    angle = torch.atan2(direction[..., 1], direction[..., 0]) / 2
    zero = torch.zeros_like(angle)
    return torch.stack((zero, zero, angle.sin(), angle.cos()), dim=-1)


class TensorMimicLite:
    def __init__(self, root, runtime, joint_names, slots, device, converted_weights):
        self.device = torch.device(device)
        cfg = json.loads((Path(root)/"policy.json").read_text())
        if cfg["anchor_body_name"] != "pelvis": raise ValueError("This checkpoint uses the root pelvis reference")
        names = cfg["policy_joint_names"]
        self.order = [joint_names.index(n) for n in names]
        self.inverse = [names.index(n) for n in joint_names]
        self.default = vector(cfg["default_joint_pos"], names, device)
        self.scale = vector(cfg["action_scale"], names, device)
        self.kp = vector(cfg["joint_kp"], names, device)[self.inverse]
        self.kd = vector(cfg["joint_kd"], names, device)[self.inverse]
        self.clip = cfg.get("action_clip")
        self.current = runtime["reference_offsets_s"].index(0.)
        self.model = TensorGraph(converted_weights, device, runtime["weights_sha256"]).eval()
        self.buffers = [torch.zeros((slots, 17, n), device=device, dtype=torch.float64) for n in (3,3,29,29)]
        self.actions = torch.zeros((slots,3,29), device=device, dtype=torch.float64)
        self.last = torch.zeros((slots,29), device=device)
        self.heading_offset = torch.zeros((slots,4), device=device, dtype=torch.float64)
        self.fresh = torch.ones(slots, device=device, dtype=torch.bool)
        self.lags = [0,1,2,3,4,8,16]

    def reset(self, slots):
        self.last[slots] = 0
        self.actions[slots] = 0
        self.fresh[slots] = True

    def observations(self, state, reference):
        q = state.joint_position_rad[:,self.order].double()
        qd = state.joint_velocity_rad_s[:,self.order].double()
        rotation = normalize(state.root_orientation_xyzw.double())
        inv = conjugate(rotation)
        omega = rotate(inv, state.root_angular_velocity_world_rad_s.double())
        gravity = rotate(inv, torch.tensor([0.,0.,-1.], device=self.device, dtype=torch.float64))
        positions = reference.root_position_world_m.double()
        rotations = normalize(reference.root_orientation_xyzw.double())
        offset = multiply(heading(rotations[:,self.current]), conjugate(heading(rotation)))
        self.heading_offset.copy_(torch.where(self.fresh[:,None], offset, self.heading_offset))
        for history, value in zip(self.buffers, (omega,gravity,q,qd)):
            history.copy_(torch.where(self.fresh[:,None,None], value[:,None], history))
            history[:,1:] = history[:,:-1].clone()
            history[:,0] = value
        self.fresh.zero_()
        self.actions[:,1:] = self.actions[:,:-1].clone()
        self.actions[:,0] = self.last
        policy = torch.cat([h[:,self.lags].flatten(1) for h in self.buffers]+[self.actions.flatten(1)],dim=-1)
        origin = positions[:,self.current].clone()
        origin[:,2] = 0
        local = rotate(conjugate(heading(rotations[:,self.current]))[:,None], positions-origin[:,None])
        orientation = matrix(multiply(conjugate(multiply(self.heading_offset, rotation))[:,None], rotations))[...,:2,:]
        joints = reference.joint_position_rad[:,:,self.order].double()
        command = torch.cat((local.flatten(1),orientation.flatten(1),joints.flatten(1)),dim=-1)
        return {"policy":policy.float(), "command":command.float()}

    @torch.inference_mode()
    def act(self, state, reference):
        raw = self.model(**self.observations(state, reference))["action"]
        action = raw.clamp(-self.clip, self.clip) if self.clip is not None else raw
        self.last.copy_(action)
        target = (self.default+self.scale*action)[:,self.inverse]
        zero = torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1).float()
