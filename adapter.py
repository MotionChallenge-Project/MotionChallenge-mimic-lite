"""MimicLite deployment observation contract expressed directly against SDK packets."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController


def projected_heading(rotation):
    # Author convention: horizontal x-axis, with a signed z-axis fallback.
    matrix=rotation.as_matrix()
    x,z=matrix[:,0],matrix[:,2]
    heading=x[:2] if np.linalg.norm(x[:2])>.1 else z[:2]*(1 if x[2]<0 else -1)
    return R.from_euler('z',np.arctan2(heading[1],heading[0]))


class MimicLite(ONNXController):
    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        q,qd,omega,gravity,rotation=self.proprio(state)
        self.buffers=[np.tile(v,(17,1)) for v in [omega,gravity,q,qd]]
        self.action_buffer=np.zeros((3,29))
        self.history_steps=[0,1,2,3,4,8,16]
        self.current=self.runtime['reference_offsets_s'].index(0.)
        ref=reference[self.current]
        self.heading_offset=projected_heading(R.from_quat(ref.root_orientation_xyzw))*projected_heading(rotation).inv()

    def observations(self,state,reference):
        q,qd,omega,gravity,rotation=self.proprio(state)
        self.buffers=[np.concatenate([v[None],h[:-1]],axis=0) for h,v in zip(self.buffers,[omega,gravity,q,qd])]
        self.action_buffer=np.concatenate([self.last[None],self.action_buffer[:-1]],axis=0)
        policy=np.concatenate([h[self.history_steps].reshape(-1) for h in self.buffers]+[self.action_buffer.reshape(-1)])
        now=reference[self.current]
        anchor=next(b for b in now.bodies if b.name==self.cfg['anchor_body_name'])
        origin=np.asarray(anchor.position_world_m).copy();origin[2]=0
        local=projected_heading(R.from_quat(anchor.orientation_xyzw)).inv().apply(np.array([f.root_position_world_m for f in reference])-origin)
        orientation=((self.heading_offset*rotation).inv()*R.from_quat([f.root_orientation_xyzw for f in reference])).as_matrix()[:,:2,:]
        joints=np.array([f.joint_position_rad for f in reference])[:,self.order]
        return {'policy':policy,'command':np.r_[local.reshape(-1),orientation.reshape(-1),joints.reshape(-1)]}


def create(root,runtime):return MimicLite(root,runtime)
