import os
import numpy as np
import subprocess
import open3d as o3d
from PIL import Image
import Imath
import OpenEXR
from concurrent.futures import ThreadPoolExecutor
import argparse


# ============================================================
#            Mitsuba Scene Template Definitions
# ============================================================

XML_HEAD = """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="{origin_x},{origin_y},{origin_z}" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="{fov}"/>
        <sampler type="ldsampler">
            <integer name="sampleCount" value="{sample_count}"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="{width}"/>
            <integer name="height" value="{height}"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>

    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/>
    </bsdf>
"""

XML_POINT = """
    <shape type="sphere">
        <float name="radius" value="{radius}"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

XML_TAIL = """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>

    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""


# ============================================================
#                   Utility Functions
# ============================================================

def convert_exr_to_jpg(exr_path, jpg_path):
    """Convert Mitsuba EXR render output to JPG."""
    file = OpenEXR.InputFile(exr_path)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Extract RGB channels
    rgb = [np.frombuffer(file.channel(c, pixel_type), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)
    rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgb]
    Image.merge("RGB", rgb8).save(jpg_path, "JPEG", quality=95)


def render_single_frame(xml_file):
    """Render a single Mitsuba frame (xml → exr → jpg)."""
    exr_output = xml_file.replace(".xml", ".exr")
    jpg_output = xml_file.replace(".xml", ".jpg")

    subprocess.run(["mitsuba", "-o", exr_output, xml_file], check=True)
    convert_exr_to_jpg(exr_output, jpg_output)

    return jpg_output


def build_scene_xml(points, colors, camera_pos, params, frame_idx, out_dir):
    """Construct and save a Mitsuba XML scene for one frame."""
    xml_segments = [XML_HEAD.format(origin_x=camera_pos[0], origin_y=camera_pos[1],
                                    origin_z=camera_pos[2], **params)]
    for p, c in zip(points, colors):
        xml_segments.append(XML_POINT.format(
            radius=params["point_radius"],
            x=p[2], y=p[0], z=p[1],
            r=c[0], g=c[1], b=c[2]
        ))
    xml_segments.append(XML_TAIL)

    xml_path = os.path.join(out_dir, f"frame_{frame_idx:03d}.xml")
    with open(xml_path, "w") as f:
        f.write("".join(xml_segments))
    return xml_path


# ============================================================
#                Rendering Pipeline Entry
# ============================================================

def render_video(
    ply_file,
    output_dir="renders",
    frames=200,
    radius=3.5,
    fps=24,
    width=800,
    height=800,
    point_radius=0.025,
    fov=25,
    sample_count=256,
    workers=16
):
    """
    Render a rotating GIF of a colored point cloud using Mitsuba.

    Args:
        ply_file (str): Path to the .ply point cloud.
        output_dir (str): Output directory for intermediate and final results.
        frames (int): Number of frames to render.
        radius (float): Rotation radius of the camera.
        fps (int): Frames per second for the GIF.
        width, height (int): Image resolution.
        point_radius (float): Radius of each rendered point (sphere).
        fov (float): Camera field of view.
        sample_count (int): Mitsuba sample count per pixel.
        workers (int): Number of parallel threads for rendering.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(ply_file).replace(".ply", ".gif")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)

    # Parameters for XML header
    xml_params = dict(width=width, height=height, fov=fov,
                      sample_count=sample_count, point_radius=point_radius)

    # Generate all XML frames
    xml_files = []
    for i in range(frames):
        theta = 2 * np.pi * i / frames
        cam_pos = [radius * np.cos(theta), radius * np.sin(theta), 3.0]
        xml_path = build_scene_xml(points, colors, cam_pos, xml_params, i, output_dir)
        xml_files.append(xml_path)
        print(f"[Frame {i+1}/{frames}] Scene saved → {xml_path}")

    # Parallel rendering
    print(f"[INFO] Start rendering {frames} frames using {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jpg_files = list(executor.map(render_single_frame, xml_files))

    # Assemble GIF
    jpg_files.sort()
    imgs = [Image.open(p) for p in jpg_files]
    duration = 1000 // fps
    gif_path = os.path.join(output_dir, file_name)
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"[DONE] GIF saved → {gif_path}")


# ============================================================
#                     Command-line Interface
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render rotating GIFs from .ply point clouds using Mitsuba.")
    parser.add_argument("--input", type=str, required=True, help="Path to a text file listing .ply files.")
    parser.add_argument("--out_dir", type=str, default="runs/video", help="Output directory for renders.")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames per rotation.")
    parser.add_argument("--radius", type=float, default=3.5, help="Camera rotation radius.")
    parser.add_argument("--fps", type=int, default=24, help="Frame rate for output GIF.")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel rendering threads.")
    args = parser.parse_args()

    # Read all .ply file paths from input text file
    with open(args.input, "r") as f:
        ply_files = [line.strip() for line in f.readlines() if line.strip()]

    # Batch render all point clouds
    for ply_path in ply_files:
        render_video(
            ply_file=ply_path,
            output_dir=args.out_dir,
            frames=args.frames,
            radius=args.radius,
            fps=args.fps,
            workers=args.workers
        )
