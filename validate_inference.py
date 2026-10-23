"""Compare the independent adapter to MimicLite's pinned author deployment code."""
import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from adapter import MimicLite
from motion_challenge.adapters.onnx import vector
from motion_challenge.sdk import BodyState, ReferenceFrame, StatePacket


def definitions(path, scope, classes, methods):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in classes:
            node.body = [item for item in node.body if isinstance(item, ast.Assign)
                         or isinstance(item, ast.FunctionDef) and item.name in methods]
            nodes.append(node)
    exec(compile(ast.Module(ast.parse('from __future__ import annotations').body + nodes, []), str(path), 'exec'), scope)


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'native-sources.json').read_text())
    for name, expected in pins['files'].items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == expected, name
    config_path = root / 'validation/author-policy.yaml'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == pins['author_config_sha256']
    cfg = yaml.safe_load(config_path.read_text())
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = MimicLite(root, runtime)
    for field, actual in [('default_joint_pos', policy.default), ('joint_kp', policy.kp),
                          ('joint_kd', policy.kd), ('action_scale', policy.scale)]:
        np.testing.assert_array_equal(actual, vector(cfg[field], policy.names))
    assert policy.names == cfg['policy_joint_names'] == cfg['joint_names_simulation']
    assert runtime['reference_steps'] == cfg['motion']['future_steps']
    np.testing.assert_allclose(runtime['reference_offsets_s'], np.array(runtime['reference_steps']) / 50, atol=1e-12)
    # Import original observation modules without starting hardware, transport or dataset loaders.
    for name in ('sim2real', 'sim2real.rl_policy', 'sim2real.rl_policy.observations',
                 'sim2real.rl_policy.utils', 'sim2real.utils'):
        module = ModuleType(name)
        module.__path__ = [str(args.source.resolve() / name.replace('.', '/'))]
        sys.modules[name] = module
    motion = ModuleType('sim2real.rl_policy.utils.motion')
    motion.MotionData = NS  # Only a typed array container is used by these observation modules.
    sys.modules[motion.__name__] = motion
    from sim2real.rl_policy.observations.base import Observation, ObsGroup, normalize_observation_array
    from sim2real.rl_policy.observations import common, mimic_lite
    scope = {'np': np, 'Observation': Observation, 'ObsGroup': ObsGroup,
             'normalize_observation_array': normalize_observation_array, 'logger': NS(info=lambda *_: None)}
    definitions(args.source / 'sim2real/rl_policy/inference/onnx_module.py', scope,
                ['ONNXModule'], {'__call__', '_get_input_value'})
    model = scope['ONNXModule'].__new__(scope['ONNXModule'])
    model.ort_session = policy.session
    model.in_keys = [node.name for node in policy.session.get_inputs()]
    model.out_keys = [node.name for node in policy.session.get_outputs()]
    model.input_shapes = [tuple(node.shape) for node in policy.session.get_inputs()]
    scope['build_inference_module'] = lambda *_: model
    definitions(args.source / 'sim2real/rl_policy/base_policy.py', scope, ['BasePolicy'],
                {'setup_policy', 'setup_observations', 'reset', 'update', 'prepare_obs_for_rl'})
    errors = {key: 0. for key in ('policy', 'command', 'network', 'action', 'motor_target', 'reset_history', 'synthetic_motor_target')}
    def compare(key, actual, expected):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
    rng = np.random.default_rng(241025)
    names = tuple(np.array(policy.names)[rng.permutation(29)])
    order = [names.index(name) for name in policy.names]
    max_action = 0.
    for reset in range(4):
        native = scope['BasePolicy'].__new__(scope['BasePolicy'])
        native.state_processor = NS(joint_names=list(names))
        native.num_actions = 29
        native.policy_joint_names = policy.names
        native.joint_names_simulation = cfg['joint_names_simulation']
        native.body_names_simulation = cfg['body_names_simulation']
        native.motion_joint_names = list(names)
        native.motion_body_names = cfg['motion']['body_names']
        native.motion_config = cfg['motion']
        native.motion_future_steps = cfg['motion']['future_steps']
        native.inference_backend = 'onnx-cpu'
        native.default_dof_angles = vector(cfg['default_joint_pos'], policy.names)
        native.action_scale = np.array(cfg['action_scale'])
        native.controlled_joint_indices = list(range(29))
        native.state_dict = {'action': np.zeros(29, np.float32)}
        with contextlib.redirect_stdout(io.StringIO()):
            native.setup_observations(cfg['observation'])
        native.setup_policy(str(args.weights))
        for step in range(40):
            actual = R.from_euler('xyz', [0.1, np.pi / 2 - .01, .3]) if step % 10 == 0 else R.random(random_state=rng)
            q, qd, omega = rng.normal(0, .35, 29), rng.normal(0, .5, 29), rng.normal(0, .3, 3)
            state = StatePacket(step * .02, names, tuple(q), tuple(qd), (0., 0., .8), tuple(actual.as_quat()),
                                (0., 0., 0.), tuple(actual.apply(omega)))
            frames = []
            for offset in runtime['reference_steps']:
                pos, rot = tuple(rng.normal(0, .3, 3)), tuple(R.random(random_state=rng).as_quat())
                bodies = tuple(BodyState(name, pos, rot, (0.,) * 3, (0.,) * 3) for name in native.motion_body_names)
                frames.append(ReferenceFrame((step + offset) * .02, max(0., (step + offset) * .02),
                              tuple(rng.normal(0, .35, 29)), pos, rot, (0.,) * 29, (0.,) * 3, (0.,) * 3, bodies))
            refs = tuple(frames)
            native.state_processor.joint_pos, native.state_processor.joint_vel = q, qd
            native.state_processor.root_ang_vel_b, native.state_processor.root_quat_w = omega, actual.as_quat()[[3, 0, 1, 2]]
            body_pos = np.array([[b.position_world_m for b in f.bodies] for f in refs])[None]
            native.motion_data = NS(joint_pos=np.array([f.joint_position_rad for f in refs])[None],
                joint_vel=np.zeros((1, 8, 29)), body_pos_w=body_pos,
                body_quat_w=np.array([[b.orientation_xyzw for b in f.bodies] for f in refs])[None][..., [3, 0, 1, 2]],
                body_lin_vel_w=np.zeros_like(body_pos), body_ang_vel_w=np.zeros_like(body_pos))
            if step == 0:
                native.reset()
                policy.reset(state, refs, reset)
                expected = native.observations['policy'].compute()
                history = np.concatenate([h[policy.history_steps].reshape(-1) for h in policy.buffers]
                                         + [policy.action_buffer.reshape(-1)])
                compare('reset_history', history, expected)
            native.update()
            expected, _ = native.prepare_obs_for_rl()
            inputs = policy.observations(state, refs)
            for key in ('policy', 'command'):
                compare(key, inputs[key], expected[key])
            raw = model(expected)['action']
            max_action = max(max_action, float(np.max(np.abs(raw))))
            compare('network', policy.infer(expected)['action'].reshape(-1), raw)
            action, target, native.state_dict = native.policy(expected)
            native.state_dict['action'] = action
            fn = policy.observations
            policy.observations = lambda *_: inputs
            result = policy.act(state, refs)
            policy.observations = fn
            compare('action', policy.last, action)
            compare('motor_target', np.array(result.joint_position_target_rad)[order], target)
        synthetic = np.linspace(-20, 20, 29, dtype=np.float32)
        scope['build_inference_module'] = lambda *_: lambda _: {'action': synthetic}
        native.setup_policy(str(args.weights))
        _, target, _ = native.policy(expected)
        fn = policy.infer
        policy.infer = lambda _: {'action': synthetic}
        result = policy.act(state, refs)
        policy.infer = fn
        scope['build_inference_module'] = lambda *_: model
        compare('synthetic_motor_target', np.array(result.joint_position_target_rad)[order], target)
    limits = {key: 2e-4 for key in errors}
    limits['network'] = 0.
    report = {'passed': all(errors[key] <= limits[key] for key in errors), 'cases': 160, 'resets': 4,
              'max_abs_error': errors, 'limits': limits, 'max_abs_native_action': max_action,
              'checkpoint_sha256': runtime['weights_sha256'], 'source_commit': pins['source_commit'],
              'scope': 'Original author deployment observation modules, history reset, projected heading including '
                       'near-vertical poses, ONNX call and raw action-to-target conversion; shuffled SDK joints. '
                       'Hardware/transport and dataset loaders omitted. Production adapter has no external deployment dependency.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
