#!/usr/bin/env python3
"""
Generates a Gazebo world (SDF + textures + JSON) from a scenario in nanouav.sim.scenarios.

  nanouav gazebo gen-world --scenario bosque --seed 3     → gazebo/worlds/gen/bosque_s3.sdf (+ .json)
  nanouav gazebo gen-world --all --seeds 0,1,2            → all scenarios

Textures are sampled with the SAME Texture.sample of our own renderer that the CNN was trained on
(value noise, stripes, checkerboard): a smaller renderer ↔ ogre2 domain gap than flat colours.
Everything is grayscale (the camera is L8).
"""
import os, sys, json, math, argparse
import numpy as np
import cv2

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # gazebo/
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))                          # nanouav package not installed
from nanouav.sim import scenarios as SC
from nanouav.scene.renderer import Texture

TEX_DIR = os.path.join(HERE, 'models', 'nano_textures', 'materials', 'textures')
OUT_DIR = os.path.join(HERE, 'worlds', 'gen')
PX_PER_M = 200


def _T(d):
    return Texture(d.get('kind', 'uniform'), albedo=d.get('albedo', 0.5), albedo2=d.get('albedo2', 0.3),
                   scale=d.get('scale', 1.0), seed=d.get('seed', 0))


def _img(tex, w_m, h_m, max_px=1024, min_px=64):
    """Grayscale uint8 image of w_m × h_m metres (u → horizontal, v → vertical)."""
    W = int(np.clip(w_m * PX_PER_M, min_px, max_px))
    H = int(np.clip(h_m * PX_PER_M, min_px, max_px))
    u = (np.arange(W) + 0.5) / W * w_m
    v = (np.arange(H) + 0.5) / H * h_m
    U, V = np.meshgrid(u, v[::-1])
    P = np.stack([U, np.zeros_like(U), V], -1)             # vertical face: (u, 0, z)
    a = _T(tex).sample(P)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _ground_img(tex, x0, x1, y0, y1, px=2048):
    xs = np.linspace(x0, x1, px)
    ys = np.linspace(y1, y0, px)
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, np.zeros_like(X)], -1)
    return (np.clip(_T(tex).sample(P), 0, 1) * 255).astype(np.uint8)


def _mat(uri, gray=1.0):
    g = f'{gray:.3f} {gray:.3f} {gray:.3f} 1'
    return (f'<material><ambient>{g}</ambient><diffuse>{g}</diffuse><specular>0.03 0.03 0.03 1</specular>'
            f'<pbr><metal><albedo_map>{uri}</albedo_map><metalness>0.0</metalness><roughness>0.95</roughness>'
            f'</metal></pbr></material>')


def _marker(name, x, y, rgb):
    c = ' '.join(f'{v:g}' for v in rgb)
    return f'''    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} 0.006 0 0 0</pose>
      <link name="l">
        <visual name="disk"><geometry><cylinder><radius>0.25</radius><length>0.004</length></cylinder></geometry>
          <material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><emissive>{c} 1</emissive></material></visual>
        <visual name="center"><pose>0 0 0.001 0 0 0</pose><geometry><cylinder><radius>0.08</radius><length>0.004</length></cylinder></geometry>
          <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>1 1 1 1</emissive></material></visual>
      </link></model>
'''


def _drone_block(A, yaw0):
    """X3 <include> block (motor plugins + velocity control) taken from the base world."""
    src = open(os.path.join(HERE, 'worlds', 'nano_uav_world.sdf')).read()
    i, j = src.index('    <include>'), src.index('</include>') + len('</include>')
    blk = src[i:j]
    k0 = blk.index('<pose>'); k1 = blk.index('</pose>') + len('</pose>')
    return blk[:k0] + f'<pose>{A[0]:.3f} {A[1]:.3f} 0.5 0 0 {yaw0:.4f}</pose>' + blk[k1:] + '\n'


def build(s, tag):
    os.makedirs(TEX_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    uri = lambda f: f'model://nano_textures/materials/textures/{f}'
    parts = []
    # ground: a single 40 × 40 m texture (2 cm/px) centred between A and B
    cx = (s['A'][0] + s['B'][0]) / 2
    cy = (s['A'][1] + s['B'][1]) / 2
    L = 40.0
    gname = f'{tag}_ground.png'
    cv2.imwrite(os.path.join(TEX_DIR, gname), _ground_img(s['ground'], cx - L / 2, cx + L / 2, cy - L / 2, cy + L / 2))
    parts.append(f'''    <model name="ground"><static>true</static><pose>{cx:.2f} {cy:.2f} -0.005 0 0 0</pose>
      <link name="l">
        <collision name="c"><geometry><box><size>{L} {L} 0.01</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{L} {L} 0.01</size></box></geometry>{_mat(uri(gname))}</visual>
      </link></model>
''')
    for n, o in enumerate(s['objects']):
        f = f'{tag}_o{n}.png'
        if o['type'] == 'box':
            hx, hy, hz = o['half']
            cv2.imwrite(os.path.join(TEX_DIR, f), _img(o['tex'], 2 * max(hx, hy), 2 * hz))
            geo = f'<box><size>{2 * hx:.3f} {2 * hy:.3f} {2 * hz:.3f}</size></box>'
            pose = f'{o["center"][0]:.3f} {o["center"][1]:.3f} {o["center"][2]:.3f} 0 0 {o.get("yaw", 0.0):.4f}'
        else:
            h = o['z1'] - o['z0']
            cv2.imwrite(os.path.join(TEX_DIR, f), _img(o['tex'], 2 * math.pi * o['r'], h))
            geo = f'<cylinder><radius>{o["r"]:.3f}</radius><length>{h:.3f}</length></cylinder>'
            pose = f'{o["c"][0]:.3f} {o["c"][1]:.3f} {(o["z0"] + o["z1"]) / 2:.3f} 0 0 0'
        parts.append(f'''    <model name="obs_{n}"><static>true</static><pose>{pose}</pose>
      <link name="l"><collision name="c"><geometry>{geo}</geometry></collision>
        <visual name="v"><geometry>{geo}</geometry>{_mat(uri(f))}</visual></link></model>
''')
    for d in s['dynamic']:
        g = int(255 * d.get('tex', {}).get('albedo', 0.8))
        f = f'{tag}_{d["name"]}.png'
        cv2.imwrite(os.path.join(TEX_DIR, f), np.full((64, 64), g, np.uint8))
        x, y, z = d['p0']
        parts.append(f'''    <model name="{d['name']}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name="l"><gravity>false</gravity>
        <inertial><mass>5.0</mass><inertia><ixx>10</ixx><iyy>10</iyy><izz>10</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>{d['r']}</radius></sphere></geometry></collision>
        <visual name="v"><geometry><sphere><radius>{d['r']}</radius></sphere></geometry>{_mat(uri(f))}</visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system" name="gz::sim::systems::VelocityControl">
        <topic>/model/{d['name']}/cmd_vel</topic></plugin>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_frame>world</odom_frame><robot_base_frame>{d['name']}</robot_base_frame>
        <odom_topic>/model/{d['name']}/odometry</odom_topic><odom_publish_frequency>30</odom_publish_frequency>
        <dimensions>3</dimensions></plugin>
    </model>
''')
    parts.append(_marker('punto_A', *s['A'], (0.1, 0.4, 1.0)))
    parts.append(_marker('meta_B', *s['B'], (0.1, 0.9, 0.1)))
    parts.append(_drone_block(s['A'], s['yaw0']))
    sdf = f'''<?xml version="1.0"?>
<!-- Generated by gazebo/tools/gen_world.py · scenario {s['name']} · seed {s['seed']} · do not edit by hand -->
<sdf version="1.9">
  <world name="nano_uav_world">
    <physics name="4ms" type="ignored"><max_step_size>0.004</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <scene><ambient>0.45 0.45 0.45 1</ambient><background>0.72 0.72 0.74 1</background><shadows>true</shadows><grid>false</grid></scene>
    <light type="directional" name="sun"><cast_shadows>true</cast_shadows><pose>0 0 10 0 0 0</pose>
      <diffuse>0.85 0.85 0.85 1</diffuse><specular>0.1 0.1 0.1 1</specular>
      <direction>{-0.5:.2f} {0.1:.2f} -0.9</direction></light>
{''.join(parts)}  </world>
</sdf>
'''
    path = os.path.join(OUT_DIR, f'{tag}.sdf')
    open(path, 'w').write(sdf)
    json.dump(s, open(os.path.join(OUT_DIR, f'{tag}.json'), 'w'), indent=1)
    return path


def ensure_model_config():
    d = os.path.join(HERE, 'models', 'nano_textures')
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, 'model.config')
    if not os.path.exists(p):
        open(p, 'w').write('<?xml version="1.0"?>\n<model><name>nano_textures</name><version>1.0</version>'
                           '<sdf version="1.9">model.sdf</sdf><description>Generated textures</description></model>\n')
        open(os.path.join(d, 'model.sdf'), 'w').write('<?xml version="1.0"?>\n<sdf version="1.9"><model name="nano_textures">'
                                                       '<static>true</static><link name="l"/></model></sdf>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='bosque', choices=list(SC.SCENARIOS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--kw', default='', help='extra parameters, e.g. "gap=1.0,density=0.5"')
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (x.split('=') for x in a.kw.split(',') if x)}
    ensure_model_config()
    names = list(SC.SCENARIOS) if a.all else [a.scenario]
    seeds = [int(x) for x in a.seeds.split(',')] if a.all else [a.seed]
    for n in names:
        for sd in seeds:
            s = SC.make(n, sd, **kw)
            tag = f'{n}_s{sd}' + (''.join(f'_{k}{v:g}' for k, v in kw.items()))
            p = build(s, tag)
            print(f'{p}  objects={len(s["objects"])} dynamic={len(s["dynamic"])} A={s["A"]} B={s["B"]}')


if __name__ == '__main__':
    main()
