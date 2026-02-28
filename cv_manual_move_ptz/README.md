# Detection + Tracking (PTZ) + Manual Aim

This version contains:
- Vision detection
- CSRT tracking
- USB PTZ control
- Manual PTZ aiming when not tracking
- No acoustic logic
- No hold mode

---

# Modes

The bottom-left overlay shows the current mode:

- idle — no tracking, waiting (manual PTZ works)
- tracking — CSRT tracker active (manual PTZ disabled)
- lost — tracker failed this frame (goes back to idle quickly)
- detected — detection found an object and initialized tracker

---

# Keyboard Controls

## Quit

- q or x — quit program

---

## Reset Tracking (works anytime)

- r — stop tracking, reset tracker state, send PTZ stop

---

## Manual PTZ Aim (ONLY when NOT tracking)

- w — tilt up
- s — tilt down
- a — pan left
- d — pan right

Manual step sizes:

MANUAL_PAN_STEP = 40.0  
MANUAL_TILT_STEP = 20.0

---

## Force Detection (ONLY when NOT tracking)

- SPACE — forces an immediate detection attempt on the next loop

Useful after you manually aim the camera at the target.

---

## PTZ Control (ONLY when NOT tracking)

- z — zero
- k — stop PTZ motion

---

# What does “Zero” (z) do?

When you press z (only when not tracking), the script:

1. Sends firmware command Z
2. Resets internal variables:  
   pan = 0  
   tilt = 0
3. Sends absolute commands:  
   P0  
   T0

This keeps software state and PTZ firmware aligned at "home".

Note: Unless your firmware provides position feedback, the script cannot verify the real physical position.

For best results, always start from z before calibrating or testing.