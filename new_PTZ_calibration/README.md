# PTZ Calibration Tool

This tool allows you to manually control your USB PTZ device in order to:

- Move to an **absolute pan/tilt position**
- Reset PTZ to **zero position**
- Fine-tune movement using keyboard nudging
- Perform automated **sweep tests**
- Determine how PTZ internal units correspond to real-world angles

The main goal is to determine the mapping between:

Acoustic Sensor (degrees)  →  PTZ Firmware Units

So later your PTZ can accurately rotate toward the direction detected by the acoustic sensor.

---

# 1. Requirements

- Python 3.9+
- `pyserial`

Install dependency:

```bash
pip install pyserial
```

---

# 2. Configure Serial Port

Inside `ptz_calibrate.py`, check:

```python
GIMBAL_USB_PORT = "/dev/ttyUSB0"
GIMBAL_BAUD = 115200
```

To verify correct device:

```bash
ls /dev/ttyUSB*
```

Unplug/replug PTZ to confirm which device appears.

---

# 3. Run the Tool

```bash
python3 ptz_calibrate.py
```

On successful start you should see:

```
[INFO] PTZ ID: ...
```

If not:
- Check port
- Check baud rate
- Check PTZ power

---

# 4. Available Commands

Type:

```
help
```

Available commands:

| Command | Description |
|----------|-------------|
| `set <pan> <tilt>` | Move to absolute pan & tilt |
| `pan <value>` | Move pan only |
| `tilt <value>` | Move tilt only |
| `zero` | Reset PTZ firmware + software state |
| `stop` | Stop motion |
| `sweep_pan a b step` | Sweep pan range |
| `sweep_tilt a b step` | Sweep tilt range |
| `nudge` | Keyboard interactive mode |
| `limits` | Show configured limits |
| `q` | Quit |

---

# 5. Basic Usage

## Reset to Zero

```
ptz> zero
```

This:
- Sends firmware `Z`
- Sets internal software pan/tilt = 0
- Moves PTZ to mechanical zero

Always start calibration from zero.

---

## Move to Absolute Position

```
ptz> set 500 0
ptz> set -500 0
ptz> set 0 200
```

Moves PTZ to specified internal units.

⚠ These are **firmware units**, not necessarily degrees.

---

## Move Only Pan

```
ptz> pan 800
```

## Move Only Tilt

```
ptz> tilt -300
```

---

## Stop Motion

```
ptz> stop
```

Sends firmware `S` command.

---

# 6. Nudge Mode (Recommended for Calibration)

Enter:

```
ptz> nudge
```

Controls:

| Key | Action |
|------|--------|
| a | pan left |
| d | pan right |
| w | tilt up |
| s | tilt down |
| z | reset zero |
| x | stop |
| q | exit nudge mode |

This mode is ideal for finding precise PTZ values for specific directions.

---

# 7. Sweep Mode

## Sweep Pan

```
ptz> sweep_pan -1000 1000 100
```

Moves pan from -1000 to 1000 in steps of 100.

Tilt remains unchanged.

---

## Sweep Tilt

```
ptz> sweep_tilt -500 500 50
```

---

# 8. Calibration Procedure for Acoustic Synchronization

### Step 1 — Reset

```
ptz> zero
```

### Step 2 — Face Forward

Ensure PTZ is physically facing forward at pan=0.

If not:
- Adjust mount
- Or record offset

---

### Step 3 — Measure Known Angles

Place reference objects at known angles:

- 0° (front)
- 45° left
- 90° left
- 45° right
- 90° right

Record PTZ pan value required to point at each.

Example:

| Real Angle | PTZ Pan |
|------------|---------|
| 0°         | 0       |
| 45° left   | -450    |
| 90° left   | -900    |
| 45° right  | 450     |
| 90° right  | 900     |

---

# 9. Compute Scale Factor

Use:

```
PTZ_units_per_degree = PTZ_value / real_angle
```

Example:

```
900 / 90 = 10 units per degree
```

So:

```
ptz_pan = acoustic_azimuth_deg * 10
```

---

# 10. Determine Offset (If Needed)

If acoustic 0° does not correspond to PTZ 0:

Example:

- Acoustic 0° → PTZ = 120 units

Then:

```
ptz_pan = (acoustic_azimuth - offset_deg) * scale
```

Where:

```
offset_deg = mechanical_front_alignment
```

---

# 11. After Calibration

Update your main detection script:

```python
PTZ_SCALE = 10.0
PTZ_OFFSET = 0.0

ptz_pan = (acoustic_azimuth - PTZ_OFFSET) * PTZ_SCALE
```

Clamp to limits:

```python
ptz_pan = clamp(ptz_pan, -PAN_LIMIT, PAN_LIMIT)
```

---

# 12. Safety Notes

- Start with small movement values
- Reduce limits if unsure
- Watch cables during sweep tests
- Avoid hitting mechanical stops

To reduce limits:

```python
PAN_LIMIT = 1500
TILT_LIMIT_UP = 500
```

---

# 13. Troubleshooting

## Permission denied

```bash
sudo usermod -a -G dialout $USER
reboot
```

## No movement

- Verify serial port
- Verify baud rate
- Check PTZ power
- Try `zero` first

---

# 14. Why This Tool Matters

Your acoustic sensor outputs:

```
azimuth (degrees)
elevation (degrees)
```

Your PTZ firmware expects:

```
absolute pan units
absolute tilt units
```

This calibration tool allows you to precisely determine the mathematical conversion between the two systems.

Once calibrated, your PTZ will reliably rotate toward the direction detected by the acoustic sensor.