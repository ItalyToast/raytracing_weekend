#![allow(dead_code)]

use rand::prelude::*;
use serde::{Serialize, Deserialize};

use crate::{vec3::Vec3, Shape, RenderMode, Material, rand_unit_vec3};

#[derive(Serialize, Deserialize, Debug)]
pub struct Scene {
    pub world             : Vec<Shape>,
    pub aspect_ratio      : f32,
    pub image_width       : u32,
    pub samples_per_pixel : u32,
    pub max_depth         : u32,
    pub vertical_fov      : f32,  // Vertical view angle (field of view)
    pub lookfrom          : Vec3, // Point camera is looking from
    pub lookat            : Vec3, // Point camera is looking at
    pub vup               : Vec3,     // Camera-relative "up" direction
    pub defocus_angle     : f32,  // Variation angle of rays through each pixel
    pub focus_dist        : f32,
    pub threads           : Option<u32>,
    pub render_mode       : RenderMode,
}

pub fn load(file: &str) -> std::io::Result<Scene> {
    let yaml = std::fs::read_to_string(file)?;
    let scene : Scene = serde_yaml::from_str(&yaml).expect("Failed to parse yaml");

    Ok(scene)
}

pub fn scene_book() -> Scene {
    let mut world : Vec<Shape> = vec![];
    //ground
    world.push(Shape::Sphere { 
        center: Vec3::new(0.0,-1000.0,0.0), 
        radius: 1000.0, 
        material: Material::Lambertian { albedo: Vec3::new(0.5, 0.5, 0.5) } 
    });

    let mut rng =  thread_rng();
    for a in -11..11 {
        for b in -11..11 {
            let choose_mat = rng.gen_range(0.0..1.0);
            let center: Vec3 = Vec3::new(a as f32 + rng.gen_range(0.0..0.9), 0.2, b as f32 + rng.gen_range(0.0..0.9));

            if center.sub(Vec3::new(4.0, 0.2, 0.0)).len() > 0.9 {
                if choose_mat < 0.8 {
                    // diffuse
                    world.push(Shape::Sphere { 
                        center: center, 
                        radius: 0.2, 
                        material: Material::Lambertian { albedo: rand_unit_vec3(&mut rng).scale_by_vec(rand_unit_vec3(&mut rng)) }
                    });
                } else if choose_mat < 0.95 {
                    // metal
                    world.push(Shape::Sphere { 
                        center: center, 
                        radius: 0.2, 
                        material: Material::Metal { 
                            albedo: Vec3 { 
                                x: rng.gen_range(0.5..1.0), 
                                y: rng.gen_range(0.5..1.0), 
                                z: rng.gen_range(0.5..1.0) 
                            },
                            fuzz: rng.gen_range(0.0..0.5), 
                        }
                    });
                } else {
                    // glass
                    world.push(Shape::Sphere { 
                        center: center, 
                        radius: 0.2, 
                        material: Material::Dielectric { ir: 1.5 }
                    });
                }
            }
        }
    }

    world.push(Shape::Sphere { 
        center: Vec3::new(0.0, 1.0, 0.0), 
        radius: 1.0, 
        material: Material::Dielectric { ir: 1.5 }
    });

    world.push(Shape::Sphere { 
        center: Vec3::new(0.0, 1.0, 0.0), 
        radius: 1.0, 
        material: Material::Dielectric { ir: 1.5 }
    });

    world.push(Shape::Sphere { 
        center: Vec3::new(-4.0, 1.0, 0.0), 
        radius: 1.0, 
        material: Material::Lambertian { albedo: Vec3::new(0.4, 0.2, 0.1) }
    });

    world.push(Shape::Sphere { 
        center: Vec3::new(4.0, 1.0, 0.0), 
        radius: 1.0, 
        material: Material::Metal { albedo: Vec3::new(0.4, 0.2, 0.1), fuzz: 0.0 }
    });

    Scene { 
        world, 
        aspect_ratio: 16.0 / 9.0, 
        image_width: 400, 
        samples_per_pixel: 500, 
        max_depth: 50, 
        vertical_fov: 20.0, 
        lookfrom: Vec3::new(13.0,2.0, 3.0), 
        lookat: Vec3::new(0.0,0.0, 0.0), 
        vup: Vec3::new(0.0,1.0, 0.0), 
        defocus_angle: 0.6, 
        focus_dist: 10.0, 
        threads: None,
        render_mode: RenderMode::Raytracing, 
    }
}

pub fn scene1() -> Scene {
    let material_ground = Material::Lambertian { albedo: Vec3{x: 0.8, y: 0.8, z: 0.0} };
    let material_center = Material::Lambertian { albedo: Vec3{x: 0.7, y: 0.3, z: 0.3} };
    let material_left   = Material::Metal { albedo: Vec3{x: 0.8, y: 0.8, z: 0.8}, fuzz: 0.1 };
    let material_right  = Material::Metal { albedo: Vec3{x: 0.8, y: 0.6, z: 0.2}, fuzz: 0.9 };
    //let material_glass  = Material::Dielectric { ir: 1.5 };

    Scene { 
        world: vec![
            Shape::Sphere { center: Vec3{ x: 0.0, y: -100.5, z: -1.0 }, radius: 100.0, material: material_ground },
            Shape::Sphere { center: Vec3{ x: 0.0, y: 0.0, z: -1.0 }, radius: 0.5, material: material_center },
            Shape::Sphere { center: Vec3{ x: 1.0, y: 0.0, z: -1.0 }, radius: 0.5, material: material_right },
            Shape::Sphere { center: Vec3{ x: -1.0, y: 0.0, z: -1.0 }, radius: 0.5, material: material_left },
        ], 
        aspect_ratio: 16.0 / 9.0, 
        image_width: 400, 
        samples_per_pixel: 500, 
        max_depth: 50, 
        vertical_fov: 20.0, 
        lookfrom: Vec3::new(2.0,2.0, 3.0), 
        lookat: Vec3::new(0.0,0.0, 0.0), 
        vup: Vec3::new(0.0,1.0, 0.0), 
        defocus_angle: 0.6, 
        focus_dist: 10.0, 
        threads: None,
        render_mode: RenderMode::Raytracing, 
    }
}