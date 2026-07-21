# G1 Rickshaw mjlab Commands

Install the project in the mjlab environment:

```bash
python -m pip install -e source/g1_rickshaw_lab
```

Validate the MuJoCo assets, fixed grippers, point connections, collision masks, and hitch geometry:

```bash
python scripts/validate_mjlab_assets.py --output mjlab_asset_validation.json
```

Validate all 19 MuJoCo static equilibria:

```bash
python scripts/validate_static_initialization.py
```

Train the teacher:

```bash
python scripts/train_teacher.py --num-envs 8192
```

Play or export the student:

```bash
python scripts/play_student.py --checkpoint <student-checkpoint.pt>
```

Render the solved initialization state:

```bash
python scripts/render_initialization.py --output outputs/initialization.png
```

Initialization precomputes all 19 MuJoCo inverse-dynamics equilibria at startup.
There is no asset conversion, reset-pose file, gain ramp, or settling controller.
