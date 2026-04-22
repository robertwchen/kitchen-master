# Labeling Specification (v1)

## Class Definitions

- **behind_line**: The relevant foot is fully on the legal side of the kitchen line, with no visible contact crossing or touching the line.
- **on_line**: Any visible part of the relevant foot appears to touch the painted kitchen line.
- **over_line**: Any visible part of the relevant foot is clearly on the kitchen side beyond the line.
- **uncertain**: The available view is insufficient to assign the above labels with confidence.

## Ambiguity Rules

Assign `uncertain` when:
- Foot-line contact is visually indeterminate due to low resolution, compression, or motion blur.
- Perspective distortion prevents a reliable geometric interpretation.
- The line edge is partially missing or heavily degraded at the contact region.

## Occlusion Handling

Use `uncertain` if the decisive contact area (foot edge and local line segment) is occluded by:
- net posts
- player body segments
- equipment or bystanders
- frame cropping

## Which Foot to Label

For v1, label the **foot nearest to the kitchen line** in the frame.

If both feet appear equally near and contact status differs, prefer the more violation-relevant interpretation and document tie-breaking in experiment metadata. If still unresolved, use `uncertain`.
