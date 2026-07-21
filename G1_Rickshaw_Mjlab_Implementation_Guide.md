# G1 Rickshaw Mjlab Implementation Guide

The task uses MuJoCo 3.5.0, Mjlab 1.2.0, MuJoCo-Warp 3.5.0, and RSL-RL
5.0.1. The registered tasks are:

- `Mjlab-G1-Rickshaw-Directional-Slope-Teacher`
- `Mjlab-G1-Rickshaw-Directional-Slope-Student`
- the corresponding `-H91` history variants

## Physical Contract

The rickshaw wheel diameter is 0.6 m and each wheel center remains 0.3 m above
the local terrain plane. The rickshaw center of mass is shifted rearward by
0.02 m.

The body-mesh tow points are `(0.276, -1.664929, 0.180746)` and
`(-0.276, -1.664929, 0.180746)` in the source STL frame. Fixed gripper sites
are connected to the corresponding rickshaw sites by two MuJoCo site-connect
equalities. The crossbar can rotate in the fixed claws, and all G1-rickshaw
collisions are disabled except contact between G1 and the two tow rods. The
fixed gripper bodies do not collide with the rods or the rest of the rickshaw.

The six G1 actuator groups use Unitree's open-source Mjlab defaults: MuJoCo
built-in position actuators, 10 Hz natural frequency, damping ratio 2.0,
motor-specific reflected armature, and the published effort limits. Waist
roll/pitch and both ankle axes use the official doubled-5020 approximation.

## Initialization

At startup, MuJoCo inverse dynamics solves the constrained pose for each of the
19 configured directional slopes. A bounded forward-dynamics solve then finds
the actuator torque and converts it to the built-in position target
`q_ctrl = q_static + tau / Kp`. Every reset selects the matching solution,
writes the robot/cart state, and installs its 29-joint `q_ctrl` reference into
the persistent Butterworth action term. The solve includes the FAT2 torso prior
and both fixed grasp positions.

Run the physical validations before training:

```bash
python scripts/validate_mjlab_assets.py
python scripts/validate_static_initialization.py
```

## Training And Playback

```bash
python scripts/train_teacher.py
python scripts/finetune_student.py --teacher <teacher.pt> --context <context.pt>
python scripts/play_student.py --checkpoint <student.pt>
```

The Mjlab runtime owns terrain assignment, startup-fixed nine-parameter domain
randomization, online FAT2/ZMP state, observations, rewards, curriculum, and
RSL-RL rollout state. There is no secondary simulator runtime path.
