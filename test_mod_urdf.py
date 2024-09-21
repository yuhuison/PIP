import xml.etree.ElementTree as et
from typing import List


tree = et.parse('models/physics.urdf')
root = tree.getroot()


def set_joint(name: str, xyz: List[float]) -> None:
    joint = root.find(f".//joint[@name='{name}']")
    if joint is not None:
        origin = joint.find('origin')
        if origin is not None:
            origin.set('xyz', f'{xyz[2]} {xyz[0]} {xyz[1]}')
    else:
        print(f'joint {name} is Not Found')

def set_limb(name: str, xyz: List[float]) -> None:
    limb = root.find(f".//link[@name='{name}']")
    if limb is not None:
        inertial = limb.find('inertial')
        collision = limb.find('collision')
        if inertial is not None:
            origin = inertial.find('origin')
            raw_xyz = origin.get('xyz').split(" ")
            origin.set('xyz', f'{raw_xyz[0]} {raw_xyz[1]} {xyz[1]/2}')
        if collision is not None:
            origin = collision.find('origin')
            raw_xyz = origin.get('xyz').split(" ")
            origin.set('xyz', f'{raw_xyz[0]} {raw_xyz[1]} {xyz[1]/2}')
            geo = collision.find('geometry')
            if geo is not None:
                capsule = geo.find('capsule')
                capsule_length = float(capsule.get('length'))
                geo.set('length', f'{capsule_length *(xyz[1]/float(raw_xyz[2]))}')

    else:
        print(f'joint {name} is Not Found')


joint_offsets = [
    [
        0,
        0,
        0
    ],
    [
        0.07715603,
        -0.03977417999999999,
        -0.00365623739
    ],
    [
        -0.07715603,
        -0.03977417999999999,
        -0.00365623739
    ],
    [
        -6.402521e-32,
        0.0520709157,
        0.012517427099999999
    ],
    [
        -1.4901161193847656e-8,
        -0.35290592900000006,
        -0.00738526229
    ],
    [
        1.4901161193847656e-8,
        -0.35290592900000006,
        -0.00738526229
    ],
    [
        2.16863662e-17,
        0.1130130290000001,
        0.0030498485999999984
    ],
    [
        0,
        -0.414744735,
        -0.024937644599999996
    ],
    [
        0,
        -0.414744735,
        -0.024937644599999996
    ],
    [
        -3.13020662e-17,
        0.10764396200000004,
        -0.014646641
    ],
    [
        0,
        -0.06305516,
        0.11068929
    ],
    [
        0,
        -0.06305516,
        0.11068929
    ],
    [
        -9.82368563e-17,
        0.1322883370000001,
        -0.0383563265
    ],
    [
        0.0223954022,
        0.10568630700000003,
        -0.029968924799999998
    ],
    [
        -0.0223954022,
        0.10568630700000003,
        -0.029968924799999998
    ],
    [
        1.57650851e-10,
        0.0731743600000001,
        0.009148589999999998
    ],
    [
        0.08616047999999998,
        -0.012260317800000031,
        7.450580999379675e-9
    ],
    [
        -0.08616048,
        -0.012260317800000031,
        7.450580999379675e-9
    ],
    [
        0.21985274599999996,
        0,
        0
    ],
    [
        -0.219852746,
        0,
        0
    ],
    [
        0.21468496299999995,
        0.000002384185789905402,
        0.00037596374799999874
    ],
    [
        -0.214684963,
        0.000002384185789905402,
        0.00037596374799999874
    ],
    [
        0.06637477999999997,
        0.007380246999999951,
        0.0020469874099999993
    ],
    [
        -0.06637477999999997,
        0.007380246999999951,
        0.0020469874099999993
    ]
]

joint_names = [['left_hip_rx', 1], ['right_hip_rx', 2], ['left_knee_rx', 4], ['right_knee_rx', 5],
               ['left_ankle_rx', 7], ['right_ankle_rx', 8]]

limb_names = [['left_knee_limb', 4], ['right_knee_limb', 5], ['left_ankle_limb', 7], ['right_ankle_limb', 8]]

for joint_name,index in joint_names:
    set_joint(joint_name, joint_offsets[index])


for limb_name,index in limb_names:
    set_limb(limb_name, joint_offsets[index])


