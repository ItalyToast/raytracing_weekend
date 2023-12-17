use std::io::Write;
use crate::vec3::Vec3;

pub fn write(path: &str, width: usize, height: usize, pixels: &Vec<Vec3>) -> std::io::Result<()>{
    let mut file = std::fs::File::create(path).expect("Could not create file");
    
    assert_eq!(width * height, pixels.len());

    // Image
    writeln!(&mut file, "P3")?;
    writeln!(&mut file, "{} {}", width, height)?;
    writeln!(&mut file, "255")?;

    // Render
    for pixel in pixels {
        // Apply the linear to gamma transform.
        let color = Vec3 {
            x: pixel.x.sqrt(),
            y: pixel.y.sqrt(),
            z: pixel.z.sqrt(),
        };

        let color = color.scale(255.999);

        writeln!(&mut file, "{} {} {}", color.x as i32, color.y as i32, color.z as i32)?;
    }
    Ok(())
}