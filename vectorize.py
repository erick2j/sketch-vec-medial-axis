from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from image_processing import normalize_to_measure, process_image
from junction_types import StrokeGraph


def _build_stroke_graph(
    image_path: Path,
    stroke_width: float,
    iso_scale: float,
    pruning_object_angle: float,
    junction_object_angle: float,
) -> StrokeGraph:
    """Load an image, normalize it to a measure, and build the stroke graph."""

    base_image = 255 - process_image(str(image_path), padding=0)
    measure = normalize_to_measure(base_image.astype(np.float32))

    return StrokeGraph(
        measure=measure,
        stroke_width=stroke_width,
        iso_scale=iso_scale,
        pruning_object_angle=pruning_object_angle,
        junction_object_angle=junction_object_angle,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize sketches and export stroke-graph SVGs",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to an image file or a directory containing images",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=3.0,
        help="Stroke width used during vectorization (default: 2.0)",
    )
    parser.add_argument(
        "--iso-scale",
        type=float,
        default=0.3,
        help="Iso-value scale between distance min/max (default: 0.2)",
    )
    parser.add_argument(
        "--pruning-object-angle",
        type=float,
        default=7 * np.pi / 16.0,
        help="Object-angle pruning threshold in radians (default: 5π/6)",
    )
    parser.add_argument(
        "--junction-object-angle",
        type=float,
        default=6 * np.pi / 16.0,
        help="Object-angle threshold for junction detection in radians (default: 2π/6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output SVG path for a single image, or destination directory when processing a folder."
        ),
    )
    return parser.parse_args()


def _iter_image_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    raise FileNotFoundError(f"Input path does not exist: {path}")


def main() -> None:
    args = _parse_args()
    image_paths = _iter_image_paths(args.input)
    if not image_paths:
        print(f"No images found for input: {args.input}")
        return

    output = args.output
    if len(image_paths) > 1:
        if output is None:
            output = args.input / "stroke_graphs"
        output.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        try:
            graph = _build_stroke_graph(
                image_path=image_path,
                stroke_width=args.stroke_width,
                iso_scale=args.iso_scale,
                pruning_object_angle=args.pruning_object_angle,
                junction_object_angle=args.junction_object_angle,
            )
        except Exception as exc:
            print(f"Failed to process {image_path}: {exc}")
            continue

        stroke_graph = graph.stroke_graph
        node_count = stroke_graph.number_of_nodes() if stroke_graph is not None else 0
        edge_count = stroke_graph.number_of_edges() if stroke_graph is not None else 0
        print(f"Stroke graph constructed for '{image_path}': {node_count} nodes, {edge_count} edges")
        print(f"Junctions: {len(graph.junctions)}, Junction trees: {len(graph.junction_trees)}")

        if len(image_paths) == 1:
            output_path = output or image_path.with_name(f"{image_path.stem}_stroke_graph.svg")
        else:
            output_dir = output
            if output_dir is None:
                output_dir = args.input
            output_path = (output_dir / image_path.stem).with_suffix(".svg")

        strokes, cycles = graph.export_stroke_graph_svg(output_path)
        print(f"Exported SVG to {output_path}")
        print(f"Strokes: {len(strokes)}, Cycles: {len(cycles)}")


if __name__ == "__main__":
    main()
