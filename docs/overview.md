# patcolour Overview

`patcolour` is a selective-color tool for visual-novel style art workflows.

## Goal

Keep emotionally important regions in color and push everything else into monochrome without
forcing a full manual paint workflow.

This is not a fully automatic segmentation problem. The tool should let a human give guidance
when similar colors appear in different semantic roles.

## Intended use cases

- keeping flowers, ribbons, or props in color
- preparing mood-heavy stills for `skirts-colour`
- quickly testing whether a scene works better as "memory color" than as full color
- serving as a reusable processing step from other creative tools

## Non-goals

- full semantic segmentation
- a GUI editor
- perfect general-purpose photo masking

## Position in the broader pipeline

`patcolour` is the "regional emphasis" tool.

- `oniazusa` changes the whole scene into a Kizuato-like background
- `patcolour` decides what color should survive inside that scene
- `skirts-colour` is the final destination where these experiments become production assets
- `name-name` and other tools may call this logic as part of larger asset-generation flows
