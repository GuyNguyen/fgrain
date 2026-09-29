# fgrain

Physically based 3D film grain simulation engine modeling silver halide photographic emulsions.

Instead of 2D noise overlays or post-processing filters, `fgrain` simulates the physical formation of photographic grain: stochastic 3D crystal synthesis, Monte Carlo photon scattering through gelatin, latent image chemical development, and transmittance scanning.

## Features

- **3D Emulsion Synthesis:** Generates crystal lattices using Poisson-disk sampling with support for cubic, tabular (T-Grain), and octahedral morphologies.
- **Monte Carlo Photon Transport:** Models incident photon paths, Beer-Lambert crystal absorption, Henyey-Greenstein bulk gelatin scattering, and anti-halation layer reflections.
- **Chemical Development:** Boolean latent-image reduction with filament growth and organic crystal clumping in dense exposure areas.
- **Optical Scanning:** Evaluates transmittance through developed silver, incorporating optical transfer function (OTF) modulation and aperture supersampling.
- **Hardware Acceleration:** Multi-core parallelization via Rayon on CPU, with optional GPU compute acceleration via `wgpu`.
- **Film Stock Presets:** Built-in profiles for Kodak Tri-X 400, Kodak T-Max 100, Ilford HP5 Plus, and push/stand development recipes.

## Installation

```bash
cargo build --release
```

## Usage

### Basic Usage

Apply film grain to an input image:

```bash
fgrain apply input.jpg output.png
```

You can also pass the input image path directly:

```bash
fgrain input.jpg output.png
```

### Common Options

```text
Options:
  -o, --output <path>           Output image path
  -p, --preset <stock>          Emulsion stock: tri-x, tri-x-1600, caffenol, t-max, hp5 (default: tri-x)
  -f, --format <format>         Negative format scaling: 35mm, 120, 4x5 (default: 35mm)
      --push <stops>            Push processing stops: 0.0 to 3.0 (default: 0.0)
      --eberhard <float>        Eberhard / Mackie line acutance halo strength (default: 0.0)
      --no-scanner-otf          Disable optical scanner lens MTF aperture simulation
  -c, --color                   Simulate color film dye-clouds (RGB)
  -e, --exposure <float>        Exposure multiplier (default: 1.0)
  -s, --grain-size <pixels>     Crystal grain diameter in pixels (default: 2.4 for 35mm Tri-X)
      --grain-strength <float>  Grain contrast/intensity multiplier (default: 1.0)
      --scale <float>           Global grain scale multiplier (default: 1.0)
      --auto-scale              Auto-scale grain parameters relative to reference resolution
      --reference-res <pixels>  Reference resolution long edge for auto-scale (default: 6000)
      --supersample <1-4>       Optical scanner aperture supersampling factor (default: 1)
      --tile-size <pixels>      Tile dimension for processing (default: 320)
  -w, --width <pixels>          Target output width (default: match input)
      --height <pixels>         Target output height (default: match input)
      --gpu                     Accelerate optical scanning with GPU compute shaders
  -h, --help                    Print help information
```

### Examples

Apply Kodak T-Max 100 grain with GPU acceleration:

```bash
fgrain apply photo.jpg -o output.png --preset t-max --gpu
```

Simulate pushed 35mm Tri-X (+2 stops) with color dye-clouds:

```bash
fgrain apply photo.jpg -o output.png --preset tri-x --push 2.0 --color
```

Auto-scale grain relative to a 24MP scan:

```bash
fgrain apply photo.jpg -o output.png --auto-scale --reference-res 6000
```

### Demonstrations

Run physical simulation tests and benchmark routines:

```bash
# Continuous exposure ramp (shadow to highlight transition)
fgrain gradient [output.png]

# Kodak Tri-X 400 cubic grain demo
fgrain tri-x [output.png]

# Kodak T-Max 100 tabular grain demo
fgrain t-max [output.png]

# Halation point-source reflection demo
fgrain halation [output.png]
```

### Help

View available commands:

```bash
fgrain --help
```

View options for a specific command:

```bash
fgrain apply --help
fgrain demo --help
```
