***How to tune the new smoothing parameters***
The new parameters are intentionally separated so you can tune smoothness without reworking the rest of the system.

The most important knobs:

- ```PID_MAX_SPEED_DPS```: lowers the maximum slope of the commanded angle (a position rate limit). Rate limiting is a standard way to prevent step-like control outputs.
Suggested start: 8–15 deg/s.
- ```PID_MAX_ACCEL_DPS2```: limits how quickly the command speed can change (reduces abrupt starts/stops). Trapezoidal profiles constrain speed and acceleration and are widely used in servo motion.
Suggested start: 80–200 deg/s².

- ```PID_ERR_LP_TAU_S```: filters pixel error to reduce vibration from tracker jitter; first-order low-pass filtering with α = dt/(τ + dt) is a standard discrete-time method.
Suggested start: 0.05–0.12 s (smaller = more responsive, larger = smoother).

- ```PIX_DEADZONE```: you already have this; deadband is commonly used to reduce hunting/chatter near setpoint. 

If you want smoothness beyond trapezoidal shaping: motion-control literature and industrial practice show that jerk discontinuities (in trapezoids) can excite vibration, and jerk-limited “S-curve” profiles reduce that by smoothing acceleration transitions.

This patch does not add a full jerk-limited planner (to keep it minimal), but it is compatible with adding one later only in the same PID block if you decide you need it.


***Optional improvement that is still “PID motor motion only”: use non-zero SPD/ACC in tracking mode***
Your Waveshare gimbal docs state clearly: for T=133, SPD and ACC set to 0 mean fastest.

If you want the gimbal firmware itself to ramp more gently, you can change only this line in the tracking branch:

```
base.gimbal_ctrl(pan_angle, tilt_angle, 0, 0)
```
to (example):
```
base.gimbal_ctrl(pan_angle, tilt_angle, 300, 50)
```
Because the documentation does not specify the numeric range/units for SPD/ACC for the gimbal in the same detail it does for the arm, you should treat these as “tune by experiment,” but the qualitative behavior (0 = fastest) is documented. 


***How to verify that the fix worked using your existing CSV log***

You already log: ```offset_x```, ```offset_y```, ```pan_angle```, ```tilt_angle```, ```fps```. You can validate smoothness by checking:

- Step size reduction: consecutive pan_angle differences become smaller and more continuous (rate limited).
- Reduced high-frequency oscillation: near center, pan_angle/tilt_angle should stop dithering when the target is mostly centered (deadband + filtering).
- Consistency across FPS drift: because the low-pass filter uses dt and the motion profile caps dt, occasional longer frames should not produce big jumps. (Discrete-time behavior depends on sampling time; bounding dt is a standard practical safeguard.)
If you later enable Ki or Kd, this patch is also safer: dt-aware I/D plus basic anti-windup reduces the risk of windup-induced “kicks” when outputs saturate. 