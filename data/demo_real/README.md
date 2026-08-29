# Attributed real optical demo subset

This directory contains six unmodified JPEG optical surface frames selected from
the public CoatingVision detection release. They are real optical inspection
frames, not synthetic renders and not ordinary overview photographs of the
manufacturing line.

- Source: CoatingVision, Figshare DOI `10.6084/m9.figshare.29260121.v1`
- License: CC BY 4.0
- Intended use: bounded dashboard/inference demonstration only
- Annotation status: source labels are deliberately not embedded in this demo
  directory; model overlays are predictions, not ground truth

`manifest.json` records the source identity and SHA-256 of every checked-in
sample. The API returns this provenance as metadata without embedding image
payloads in its operations snapshot.
