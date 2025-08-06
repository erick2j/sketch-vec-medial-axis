import svgwrite
import numpy as np

def export_contours_to_svg(contours, filename, stroke='black', stroke_width=1.0, height=None, width=None):
    """
    Export a list of (N, 2) contour arrays as an SVG file.

    Parameters:
        contours: list of Nx2 numpy arrays (from find_contours)
        filename: output SVG file path
        stroke: stroke color (default: black)
        stroke_width: stroke thickness
        height, width: optional dimensions of canvas
    """
    # Determine canvas size if not given
    all_points = np.vstack(contours)
    max_y, max_x = np.max(all_points, axis=0)
    min_y, min_x = np.min(all_points, axis=0)
    svg_height = height if height else int(np.ceil(max_y + 10))
    svg_width  = width  if width  else int(np.ceil(max_x + 10))

    dwg = svgwrite.Drawing(filename, size=(svg_width, svg_height))

    for contour in contours:
        # Convert (row, col) → (x, y)
        points = contour[:, [1, 0]]
        # Convert to SVG path string
        path_data = ["M {:.3f},{:.3f}".format(*points[0])]
        for pt in points[1:]:
            path_data.append("L {:.3f},{:.3f}".format(*pt))
        path_str = " ".join(path_data)

        dwg.add(dwg.path(d=path_str, fill='none', stroke=stroke, stroke_width=stroke_width))

    dwg.save()
    print(f"[SVG] Exported {len(contours)} contours to {filename}")

