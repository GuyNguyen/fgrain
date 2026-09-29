use image::GenericImageView;

fn main() {
    let input_path = if std::path::Path::new("test_assets/DSCF4259.JPG").exists() {
        "test_assets/DSCF4259.JPG"
    } else {
        "DSCF4259.JPG"
    };
    let img = image::open(input_path).expect("Failed to open DSCF4259.JPG");
    println!("Loaded {}: {:?}", input_path, img.dimensions());

    // Crop a 1024x1024 region around the lamp and swing set
    let crop = img.crop_imm(2000, 1500, 1024, 1024);
    let out_path = if std::path::Path::new("test_assets").exists() {
        "test_assets/crop_test_input.png"
    } else {
        "crop_test_input.png"
    };
    crop.save(out_path).unwrap();
    println!("Saved {} (1024x1024)", out_path);
}
