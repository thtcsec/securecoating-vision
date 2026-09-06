# Third-Party Notices

This file documents licenses and attributions for SecureCoating-Vision and
materials it redistributes or builds upon.

## Project source license (AGPL-3.0)

SecureCoating-Vision project source is licensed under the
[GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0).

This repository ships Ultralytics YOLO26 weights (`outputs/best.pt`,
`outputs/model.onnx`). Those model artifacts inherit Ultralytics AGPL-3.0
terms when redistributed without an Ultralytics Enterprise license. Because
this distribution includes those weights under AGPL (not Enterprise), the
project source is released under AGPL-3.0 for license compliance.

## Ultralytics

- Project: [Ultralytics](https://ultralytics.com)
- License: GNU Affero General Public License v3.0 (AGPL-3.0)
- Notes: YOLO training/inference stack and YOLO26-derived weights used by
  this repository are subject to Ultralytics AGPL-3.0 unless covered by a
  separate Ultralytics Enterprise agreement.

## CoatingVision dataset

- Dataset: CoatingVision (Argonne / Figshare release)
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- DOI: [10.6084/m9.figshare.29260121.v1](https://doi.org/10.6084/m9.figshare.29260121.v1)
- Notes: Real optical coating images and published labels used for RGB
  detection experiments. Attribution is required under CC BY 4.0.

## LIBAD / DA-Core

- Dataset / method: LIBAD (aligned visible-light and inline-compatible
  X-ray validation inputs) and DA-Core memory-bank baseline
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
  for the official LIBAD release materials
- Attribution: Sui, Lichau, Phelippeau, and Liu (LIBAD / DA-Core; see
  [arXiv:2608.07958](https://arxiv.org/abs/2608.07958))
- Notes: SecureCoating-Vision does not claim DA-Core as an original
  algorithm. The local contribution is the evidence-gated industrial
  decision layer on top of modality scores.
