use image::{Rgb, RgbImage};

fn main() {
    let mut img = RgbImage::new(256, 256);

    for y in 0..256 {
        for x in 0..256 {
            let dx = (x as f64) - 128.0;
            let dy = (y as f64) - 128.0;
            let dist = (dx * dx + dy * dy).sqrt();

            // Gradient vignette base
            let base = (190.0 - dist * 0.9 + (x as f64 / 256.0) * 60.0).clamp(15.0, 240.0);

            // Highlight specular sphere
            let hx = (x as f64) - 165.0;
            let hy = (y as f64) - 85.0;
            let h_dist = (hx * hx + hy * hy).sqrt();
            let highlight = if h_dist < 35.0 {
                (35.0 - h_dist) * 2.0
            } else {
                0.0
            };

            // Deep shadow block
            let in_shadow = (25..=100).contains(&x) && (155..=230).contains(&y);

            let val = if in_shadow {
                18.0
            } else {
                (base + highlight).clamp(0.0, 255.0)
            } as u8;

            img.put_pixel(
                x,
                y,
                Rgb([val, (val as f32 * 0.95) as u8, (val as f32 * 0.88) as u8]),
            );
        }
    }

    img.save("sample_scene.png").unwrap();
    println!("Created sample_scene.png");
}
