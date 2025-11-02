#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mitsuba Rendering Pipeline for Point Clouds
-------------------------------------------
This script provides a full pipeline for:
  1. Generating Mitsuba .xml scene files from .ply point clouds
  2. Rendering them into .exr images using Mitsuba
  3. Converting .exr images into .jpg format

Usage Examples:
---------------
# 1. Generate .xml files
python render_pipeline.py --mode xml \
    --input_txt visualization/piad/path2all.text \
    --xml_dir visualization/piad/xml_file

# 2. Render .xml to .exr using Mitsuba
python render_pipeline.py --mode render \
    --xml_dir visualization/piad/xml_file \
    --exr_dir visualization/piad/exr_file

# 3. Convert .exr to .jpg
python render_pipeline.py --mode exr2jpg \
    --exr_dir visualization/piad/exr_file \
    --jpg_dir visualization/piad/jpg_file

# 4. Run full pipeline (generate XML → render → convert)
python render_pipeline.py --mode full \
    --input_txt visualization/piad/path2all.text \
    --xml_dir visualization/piad/xml_file \
    --exr_dir visualization/piad/exr_file \
    --jpg_dir visualization/piad/jpg_file
"""

import os
import subprocess
import numpy as np
import open3d as o3d
from PIL import Image
import OpenEXR, Imath
import argparse


# ============================================================
#                Mitsuba Scene Template
# ============================================================

XML_HEAD = """<scene version="0.6.0">
    <integrator type="path"><integer name="maxDepth" value="-1"/></integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        <sampler type="ldsampler"><integer name="sampleCount" value="256"/></sampler>
        <film type="hdrfilm">
            <integer name="width" value="800"/>
            <integer name="height" value="800"/>
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

XML_POINT = """    <shape type="sphere">
        <float name="radius" value="0.025"/>
        <transform name="toWorld">
            <translate x="{x}" y="{y}" z="{z}"/>
        </transform>
        <bsdf type="diffuse">
            <rgb name="reflectance" value="{r},{g},{b}"/>
        </bsdf>
    </shape>
"""

XML_TAIL = """    <shape type="rectangle">
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
        <emitter type="area"><rgb name="radiance" value="6,6,6"/></emitter>
    </shape>
</scene>
"""


# ============================================================
#                Step 1: Generate XML Files
# ============================================================

def generate_xml_from_ply(txt_file, xml_dir):
    """Convert all .ply listed in txt_file → Mitsuba .xml scenes."""
    os.makedirs(xml_dir, exist_ok=True)
    with open(txt_file, "r") as f:
        files = [line.strip() for line in f.readlines() if line.strip()]
    for ply_file in files:
        xml_segments = [XML_HEAD]
        pcd = o3d.io.read_point_cloud(ply_file)
        pts, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
        for p, c in zip(pts, colors):
            xml_segments.append(XML_POINT.format(x=p[2], y=p[0], z=p[1],
                                                 r=c[0], g=c[1], b=c[2]))
        xml_segments.append(XML_TAIL)
        xml_path = os.path.join(xml_dir, os.path.basename(ply_file).replace(".ply", ".xml"))
        with open(xml_path, "w") as fxml:
            fxml.write("".join(xml_segments))
        print(f"[XML] Generated: {xml_path}")
    print(f"All XML files saved to: {xml_dir}")


# ============================================================
#                Step 2: Mitsuba Rendering
# ============================================================

def render_with_mitsuba(xml_dir, exr_dir):
    """Render all .xml scenes to .exr using Mitsuba."""
    os.makedirs(exr_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    for xml_file in xml_files:
        base = os.path.splitext(xml_file)[0]
        xml_path = os.path.join(xml_dir, xml_file)
        exr_path = os.path.join(exr_dir, f"{base}.exr")
        if os.path.exists(exr_path):
            print(f"[Skip] {exr_path} already exists.")
            continue
        cmd = ["mitsuba", "-o", exr_path, xml_path]
        subprocess.run(cmd, check=True)
        print(f"[Render] {xml_file} → {exr_path}")
    print(f"All EXR files saved to: {exr_dir}")


# ============================================================
#                Step 3: EXR → JPG Conversion
# ============================================================

def convert_exr_to_jpg(exr_path, jpg_path):
    """Convert a single Mitsuba EXR output to JPG."""
    file = OpenEXR.InputFile(exr_path)
    pix_type = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    rgb = [np.frombuffer(file.channel(c, pix_type), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)
    rgb8 = [Image.frombytes("F", size, c.tobytes()).convert("L") for c in rgb]
    Image.merge("RGB", rgb8).save(jpg_path, "JPEG", quality=95)


def batch_exr_to_jpg(exr_dir, jpg_dir):
    """Batch convert all EXR files in folder."""
    os.makedirs(jpg_dir, exist_ok=True)
    for file in os.listdir(exr_dir):
        if not file.endswith(".exr"):
            continue
        exr_file = os.path.join(exr_dir, file)
        jpg_file = os.path.join(jpg_dir, file.replace(".exr", ".jpg"))
        convert_exr_to_jpg(exr_file, jpg_file)
        print(f"[EXR→JPG] {jpg_file}")
    print(f"All JPGs saved to: {jpg_dir}")


# ============================================================
#                Unified Pipeline Entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full Mitsuba Rendering Pipeline for Point Clouds.")
    parser.add_argument("--mode", choices=["xml", "render", "exr2jpg", "full"], required=True,
                        help="Pipeline step to execute.")
    parser.add_argument("--input_txt", type=str, default="runs/ply/ply_paths.txt")
    parser.add_argument("--xml_dir", type=str, default="runs/xml_new", help="Directory for generated .xml files.")
    parser.add_argument("--exr_dir", type=str, default="runs/exr_file", help="Directory for Mitsuba .exr outputs.")
    parser.add_argument("--jpg_dir", type=str, default="runs/jpg_file", help="Directory for converted .jpg images.")
    args = parser.parse_args()

    if args.mode in ["xml", "full"]:
        if not args.input_txt:
            raise ValueError("--input_txt is required for XML generation.")
        generate_xml_from_ply(args.input_txt, args.xml_dir)
    if args.mode in ["render", "full"]:
        render_with_mitsuba(args.xml_dir, args.exr_dir)
    if args.mode in ["exr2jpg", "full"]:
        batch_exr_to_jpg(args.exr_dir, args.jpg_dir)


if __name__ == "__main__":
    main()
