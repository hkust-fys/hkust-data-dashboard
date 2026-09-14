# Map label placement

`dashboard.maps.label_layout` places label boxes independently of projection
and drawing. Single and stacked bus labels share the same collision checks and
road-priority policy; directional arrows remain at their marker anchors.

The renderer supplies separate masks for Clear Water Bay Road / New Clear
Water Bay Road geometry and traffic-colored pixels in the Google Maps base.
Keeping the named roads clear takes priority over avoiding other traffic roads.
Ordinary traffic avoidance is balanced against label displacement.

When preferred rows cover a protected road or collide with another label or
arrow, the solver searches in two dimensions within 96 logical pixels of the
usual side positions. A canvas-wide fallback is reserved for label/arrow
congestion. If no local clear slot fits, road avoidance remains best-effort so
labels stay readable and nearby. Missing road geometry disables the extra
named-road priority while ordinary traffic-pixel avoidance remains available.

`tests/test_label_layout.py` covers loops, wider protected corridors, native map
scales, and locality when overlap is unavoidable. The map renderer tests also
cover ordinary traffic avoidance and the priority of the two named roads.
