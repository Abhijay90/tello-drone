## Phase 3: Gesture Classifier Improvements

### Known Issues from Live Testing

- [ ] **Palm Up not detected** — Palm Up gesture consistently classified as Palm Down or another palm variant. Needs feature-level review of up/down normal computation and hand orientation logic.
- [ ] **Closed Fist confused with Thumbs Down** — Both have curled fingers; distinguishing thumb position from palm curvature is unreliable with heuristic approach. Requires additional hand posture features (e.g., thumb tip-to-palm-center distance relative to wrist).

### Potential Fixes
- Use wrist-to-palm-normal vector for accurate palm-up/palm-down discrimination
- Add thumb tip to index tip distance as an auxiliary feature for close_fist vs thumbs_down separation
- Re-evaluate classify() angle-based heuristics in `vision/gestures_v2.py` with more synthetic test cases
