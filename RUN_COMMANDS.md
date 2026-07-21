# G1 Rickshaw mjlab Commands

Install the project in the mjlab environment:

```bash
python -m pip install -e source/g1_rickshaw_lab
```

Validate the MuJoCo assets, fixed grippers, welds, collision masks, and hitch geometry:

```bash
python scripts/validate_mjlab_assets.py --output mjlab_asset_validation.json
```

Train with the same CLI convention as `unitree_rl_mjlab`:

```bash
python scripts/train.py Unitree-G1-Rickshaw-Flat --env.scene.num-envs 4096
```

Play a checkpoint:

```bash
python scripts/play.py Unitree-G1-Rickshaw-Flat \
  --checkpoint-file logs/rsl_rl/g1_rickshaw_velocity/<run>/model_<iteration>.pt
```

Initialization is produced by MuJoCo inverse-dynamics equilibrium solving on first reset and cached by gradient. There is no USD conversion, Kit process, reset-pose file, gain ramp, or settling controller.
