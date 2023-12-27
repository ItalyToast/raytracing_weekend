mod vec3;
mod ppm;
mod scene;
mod bvh;

use pk_stl::geometry::Triangle;
use rand::thread_rng;
use vec3::Vec3;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub enum RenderMode {
    Raytracing,
    Normals,
    Depth,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Shape {
    Sphere{
        center: Vec3, 
        radius: f32,
        material: Material,
    },
    Aabb{
        center: Vec3,
        size: Vec3,
        material: Material,
    },
    Plane{
        normal: Vec3,
        distance: f32,
        material: Material,
    },
    Triangle{
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        material: Material,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Material {
    Lambertian {
        albedo: Vec3,
    },
    Metal {
        albedo: Vec3,
        fuzz: f32,
    },
    Dielectric {
        //albedo: Vec3,
        ir: f32,
    },
    Light {
        albedo: Vec3,
    } 
}

struct HitRecord {
    t: f32,
    p: Vec3,
    face_normal: bool,
    normal: Vec3,
    material: Material,
}

fn main() {
    let mut scene = scene::load("scene_triangle.yaml").expect("Failed to load scene");

    let mut model = load_stl("model.stl", &Material::Lambertian { albedo: Vec3 { x: 1.0, y: 1.0, z: 1.0 } }).unwrap();
    let offset = Vec3{ x: 0.0, y: 0.0, z: 100.0 };
    let mut model = model.into_iter()
    .map(|t| match t {
        Shape::Triangle { v0, v1, v2, material } => Shape::Triangle { v0: v0.add(offset), v1: v1.add(offset), v2: v2.add(offset), material },
        _ => todo!(),
    }).collect();
    scene.world.append(&mut model);

    println!("primitives: {}", scene.world.len());
    //let world = scene1();

    // let aspect_ratio = 16.0 / 9.0;
    // let image_width = 400;
    // let samples_per_pixel = 50;
    // let max_depth         = 10;
    // let vertical_fov = 20.0;  // Vertical view angle (field of view)
    // let lookfrom = Vec3::new(13.0,2.0, 3.0);  // Point camera is looking from
    // let lookat   = Vec3::new(0.0,0.0, 0.0);   // Point camera is looking at
    // let vup      = Vec3::new(0.0,1.0, 0.0);     // Camera-relative "up" direction
    // let defocus_angle = 0.6;  // Variation angle of rays through each pixel
    // let focus_dist = 10.0;    // Distance from camera lookfrom point to plane of perfect focus

    let aspect_ratio = scene.aspect_ratio;
    let image_width = scene.image_width;
    let samples_per_pixel = scene.samples_per_pixel; 
    let max_depth         = scene.max_depth; 
    let vertical_fov = scene.vertical_fov; // Vertical view angle (field of view)
    let lookfrom = scene.lookfrom; // Point camera is looking from
    let lookat   = scene.lookat; // Point camera is looking at
    let vup      = scene.vup; // Camera-relative "up" direction
    let defocus_angle = scene.defocus_angle; // Variation angle of rays through each pixel
    let focus_dist = scene.focus_dist; // Distance from camera lookfrom point to plane of perfect focus

    // Calculate the image height, and ensure that it's at least 1.
    let image_height = (image_width as f32 / aspect_ratio) as u32;
    let image_height = if image_height < 1 { 1 } else { image_height }; 

    // Viewport widths less than one are ok since they are real valued.
    //let focal_length = lookfrom.sub(lookat).len();
    let theta = degrees_to_radians(vertical_fov);
    let h = f32::tan(theta/2.0);
    let viewport_height = 2.0 * h * focus_dist;
    let viewport_width = viewport_height * (image_width as f32 / image_height as f32);
    let camera_center = lookfrom;

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    let w = lookfrom.sub(lookat).unit_vector();
    let u = vup.cross(w).unit_vector();
    let v = w.cross(u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    let viewport_u = u.scale(viewport_width);
    let viewport_v = v.reverse().scale(viewport_height);

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    let pixel_delta_u = viewport_u.div_scale(image_width  as f32);
    let pixel_delta_v = viewport_v.div_scale(image_height as f32);


    // Calculate the location of the upper left pixel.
    let upper_left_focal_point = w.scale(focus_dist).add(viewport_u.div_scale(2.0)).add(viewport_v.div_scale(2.0));
    let viewport_upper_left = camera_center.sub(upper_left_focal_point);
    let pixel00_loc = viewport_upper_left.add(pixel_delta_u.add(pixel_delta_v).scale(0.5));

    let defocus_radius = focus_dist * degrees_to_radians(defocus_angle / 2.0).tan();
    let defocus_disk_u = u.scale(defocus_radius);  // Defocus disk horizontal radius
    let defocus_disk_v = v.scale(defocus_radius);  // Defocus disk vertical radius

    rayon::ThreadPoolBuilder::new()
        .num_threads(scene.threads.unwrap_or(16) as usize)
        .build_global()
        .unwrap();

    let mut pixels :Vec<Vec3>= vec![];
    for h in 0..image_height {
        if h % 10 == 0 { println!("Scanlines remaining: {}", image_height - h); }

        let mut row_pixels :Vec<Vec3>= (0..image_width).into_par_iter().map(|w|{
            let mut rng = thread_rng();
            let pixel_h_offset = pixel_delta_v.scale(h as f32);
            let pixel_w_offset = pixel_delta_u.scale(w as f32);
            let pixel_center = pixel00_loc.add(pixel_h_offset).add(pixel_w_offset);

            let mut pixel = Vec3{ x: 0.0, y: 0.0, z: 0.0 };
            for _ in 0..samples_per_pixel {
                // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
                // the camera defocus disk.
                let u_offset : f32 = rng.gen_range(-0.5..0.5);
                let v_offset : f32 = rng.gen_range(-0.5..0.5);
                let pixel_sample_square = pixel_delta_u.scale(u_offset).add(pixel_delta_v.scale(v_offset));

                let defocus = match defocus_angle <= 0.0 { 
                    true => camera_center,
                    false => camera_center.add(defocus_disk_sample(defocus_disk_u, defocus_disk_v, &mut rng)),
                };

                let sample_ray = Ray {
                    origin: defocus,
                    direction: pixel_center.sub(defocus).add(pixel_sample_square).unit_vector(),
                };

                let sample = match scene.render_mode {
                    RenderMode::Raytracing => ray_color(&sample_ray, &scene.world, max_depth, &mut rng),
                    RenderMode::Normals => ray_normal(&sample_ray, &scene.world, &mut rng),
                    RenderMode::Depth => ray_depth(&sample_ray, &scene.world, &mut rng),
                };
                pixel = pixel.add(sample);
            }
            pixel.div_scale(samples_per_pixel as f32)
       }).collect(); 

       pixels.append(&mut row_pixels);
    }

    ppm::write("output.ppm", image_width as usize, image_height as usize, &pixels).expect("Could not write PPM");
    println!("Done");
}

fn defocus_disk_sample(defocus_disk_u: Vec3, defocus_disk_v: Vec3, rng: &mut ThreadRng) -> Vec3 {
    // Returns a random point in the camera defocus disk.
    let p = random_in_unit_disk(rng);
    defocus_disk_u.scale(p.x).add(defocus_disk_v.scale(p.y))
}

fn hit_sphere(center: Vec3, radius: f32, r: &Ray, interval : Interval, material: &Material) -> Option<HitRecord> {
    let oc = r.origin.sub(center);
    let a = r.direction.len_squared();
    let half_b = oc.dot(r.direction);
    let c = oc.len_squared() - radius*radius;
    let discriminant = half_b * half_b - a*c;
    if discriminant < 0.0 {
        return None;
    }

    let sqrtd = discriminant.sqrt();
    // Find the nearest root that lies in the acceptable range.
    let mut root = (-half_b - sqrtd) / a;
    if !interval.surrounds(root) {
        root = (-half_b + sqrtd) / a;
        if !interval.surrounds(root) {
            return None;
        }
    }

    let t = root;
    let p = r.at(root);
    let normal = p.sub(center).div_scale(radius);
    let (face_normal, normal) = face_normal(&r, normal);

    Some(HitRecord{t, p, face_normal, normal, material: material.clone()})
}

fn hit_aabb(center: Vec3, size: Vec3, r: &Ray, interval : Interval, material: &Material) -> Option<HitRecord> {
    //bool RayAABBSimple(Ray ray, AABB aabb, float tmin, float tmax) {
    let aabb_min = center.sub(size.scale(0.5));
    let aabb_max = center.add(size.scale(0.5));

    let mut tmin = interval.min;
    let mut tmax = interval.max;

    let mut hit = Vec3::new(0.0,0.0,0.0);

    for a in 0..3 {
        let inv_d = 1.0 / r.direction.get(a);
        let mut t0 = (aabb_min.get(a) - r.origin.get(a)) * inv_d;
        let mut t1 = (aabb_max.get(a) - r.origin.get(a)) * inv_d;
        if inv_d < 0.0 {
            let temp = t1;
            t1 = t0;
            t0 = temp;
        }
        
        if t0 > tmin {
            tmin = t0;
        }
        if t1 < tmax {
            tmax = t1;
        }
        if tmax <= tmin {
            return None;
        }
    }

    Some(HitRecord { 
        t: hit.len(), 
        p: r.at(tmin), 
        face_normal: true, 
        normal: Vec3 { x: 1.0, y: 1.0, z: 1.0 }.unit_vector(), 
        material: material.clone()
    })
}

fn hit_plane(normal: Vec3, distance: f32, r: &Ray, interval : Interval, material: &Material) -> Option<HitRecord> {
    //https://gdbooks.gitbooks.io/3dcollisions/content/Chapter3/raycast_plane.html
    //t = (plane.distance - DOT(ray.position, plane.normal) / DOT(ray.normal, plane.normal)
    let normal = normal.unit_vector();
    let t = distance - r.origin.dot(normal) / r.direction.dot(normal);

    if interval.contains(t) {
        let (face_normal, normal) = face_normal(&r, normal);

        Some(HitRecord { 
            t: t, 
            p: r.at(t), 
            face_normal, 
            normal, 
            material: material.clone()
        })
    }
    else {
        None
    }
}

//https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution.html
fn hit_triangle(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3, material: &Material) -> Option<HitRecord>
{
    // compute the plane's normal
    let v0v1 = v1.sub(v0);
    let v0v2 = v2.sub(v0);
    // no need to normalize
    let N = v0v1.cross(v0v2); // N
    let area2 = N.len();
 
    // Step 1: finding P
    
    // check if the ray and plane are parallel.
    let NdotRayDirection = N.dot(dir);
    if NdotRayDirection.abs() < f32::EPSILON {// almost 0
        return None; // they are parallel, so they don't intersect! 
    } 

    // compute d parameter using equation 2
    let d = -N.dot(v0);
    
    // compute t (equation 3)
    let t = -(N.dot(orig) + d) / NdotRayDirection;
    
    // check if the triangle is behind the ray
    if t < 0.0 { return None; } // the triangle is behind
 
    // compute the intersection point using equation 1
    let P = orig.add(dir.scale(t));
 
    // Step 2: inside-outside test
    //Vec3f C; // vector perpendicular to triangle's plane
 
    // edge 0
    let edge0 = v1.sub(v0); 
    let vp0 = P.sub(v0);
    let C = edge0.cross(vp0);
    if N.dot(C) < 0.0 {
        return None; // P is on the right side
    }

    // edge 1
    let edge1 = v2.sub(v1); 
    let vp1 = P.sub(v1);
    let C = edge1.cross(vp1);
    if N.dot(C) < 0.0 {
        return None; // P is on the right side
    }  
 
    // edge 2
    let edge2 = v0.sub(v2); 
    let vp2 = P.sub(v2);
    let C = edge2.cross(vp2);
    if N.dot(C) < 0.0 {
        return None; // P is on the right side;
    } 

    let r = Ray{origin: orig, direction: dir };
    let (face_normal, normal) = face_normal(&r, N);
    Some(HitRecord { 
        t: t, 
        p: r.at(t), 
        face_normal, 
        normal, 
        material: material.clone()
    }) // this ray hits the triangle
}

fn ray_color(ray : &Ray, world: &Vec<Shape>, depth: u32, rng: &mut ThreadRng) -> Vec3 {
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if depth <= 0 { return Vec3{ x: 0.0, y: 0.0, z: 0.0 } }

    let mut closest_hit = f32::INFINITY;
    let mut t : Option<HitRecord> = None;
    for shape in world {
        let interval = Interval { min: 0.001, max: 100.0};
        let hit = match shape {
            Shape::Sphere { center, radius, material } => hit_sphere(center.clone(), radius.clone(), &ray, interval, material),
            Shape::Aabb { center, size, material } => hit_aabb(center.clone(), size.clone(), &ray, interval, material),
            Shape::Plane { normal, distance, material } => hit_plane(*normal, *distance, &ray, interval, material),
            Shape::Triangle { v0, v1, v2, material } => hit_triangle(ray.origin, ray.direction, *v0, *v1, *v2, material),
        };

        if let Some(hr) = &hit{
            if hr.t < closest_hit {
                closest_hit = hr.t;
                t = hit;
            }
        }
    };

    if let Some(hit) = t {
        if let Material::Light { albedo } = hit.material {
            return albedo;
        }

        return match scatter(ray, &hit, rng) {
            Some((ray, attenuation)) => {
                return ray_color(&ray, &world, depth-1,  rng).scale_by_vec(attenuation)
            },
            None => Vec3 {
                x: 0.0, 
                y: 0.0, 
                z: 0.0, 
            },
        };
    }

    let white = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    let blue = Vec3 {
        x: 0.5,
        y: 0.7,
        z: 1.0,
    };

    let unit_direction  = ray.direction.unit_vector();
    let a = 0.5 * unit_direction.y + 1.0; 
    let white_weight = white.scale(1.0 - a);
    let blue_weight  = blue.scale(a);
    white_weight.add(blue_weight)
}

fn ray_normal(ray : &Ray, world: &Vec<Shape>, _rng: &mut ThreadRng) -> Vec3 {
    let mut closest_hit = f32::INFINITY;
    let mut t : Option<HitRecord> = None;
    for shape in world {
        let interval = Interval { min: 0.001, max: 100.0};
        let hit = match shape {
            Shape::Sphere { center, radius, material } => hit_sphere(center.clone(), radius.clone(), &ray, interval, material),
            Shape::Aabb { center, size, material } => hit_aabb(center.clone(), size.clone(), &ray, interval, material),
            Shape::Plane { normal, distance, material } => hit_plane(*normal, *distance, &ray, interval, material),
            Shape::Triangle { v0, v1, v2, material } => hit_triangle(ray.origin, ray.direction, *v0, *v1, *v2, material),
        };

        if let Some(hr) = &hit{
            if hr.t < closest_hit {
                closest_hit = hr.t;
                t = hit;
            }
        }
    };

    match t {
        Some(hit) => hit.normal,
        None => Vec3 {
            x: 0.0, 
            y: 0.0, 
            z: 0.0, 
        },
    }
}

fn ray_depth(ray : &Ray, world: &Vec<Shape>, rng: &mut ThreadRng) -> Vec3 {
    let mut closest_hit = f32::INFINITY;
    let mut t : Option<HitRecord> = None;
    for shape in world {
        let interval = Interval { min: 0.001, max: 100.0};
        let hit = match shape {
            Shape::Sphere { center, radius, material } => hit_sphere(center.clone(), radius.clone(), &ray, interval, material),
            Shape::Aabb { center, size, material } => hit_aabb(center.clone(), size.clone(), &ray, interval, material),
            Shape::Plane { normal, distance, material } => hit_plane(*normal, *distance, &ray, interval, material),
            Shape::Triangle { v0, v1, v2, material } => hit_triangle(ray.origin, ray.direction, *v0, *v1, *v2, material),
        };

        if let Some(hr) = &hit{
            if hr.t < closest_hit {
                closest_hit = hr.t;
                t = hit;
            }
        }
    };

    match t {
        Some(hit) => Vec3::new(1.0, 1.0, 1.0).scale(1.0 / hit.t),
        None => Vec3 {
            x: 0.0, 
            y: 0.0, 
            z: 0.0, 
        },
    }
}

fn scatter(ray: &Ray, hit: &HitRecord, rng: &mut ThreadRng) -> Option<(Ray, Vec3)> {
    match hit.material {
        Material::Lambertian { albedo } => {
            let mut direction = hit.normal.add(rand_unit_vec3_sphere(rng));
            
            // Catch degenerate scatter direction
            if direction.near_zero() {
                direction = hit.normal;
            }

            let next_ray = Ray { 
                origin: hit.p, 
                direction: direction,
            };
            Some((next_ray, albedo.clone()))
        },
        Material::Metal{ albedo, fuzz } => {
            let direction = reflect(ray.direction.unit_vector(), hit.normal);
            let scattered =  direction.add(rand_unit_vec3(rng).scale(fuzz));
            
            let next_ray = Ray { 
                origin: hit.p, 
                direction: scattered,
            };
            
            match scattered.dot(hit.normal) > 0.0 {
                true => Some((next_ray, albedo.clone())),
                false => None,
            }
        },
        Material::Dielectric { ir } => {
            let refraction_ratio = match hit.face_normal {
                true => 1.0/ir,
                false => ir,
            };
    
            let unit_direction = ray.direction.unit_vector();
            //let refracted = refract(unit_direction, hit.normal, refraction_ratio);
    
            let cos_theta = unit_direction.reverse().dot(hit.normal).min(1.0);
            let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
    
            let cannot_refract = refraction_ratio * sin_theta > 1.0;
            let direction = match cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.gen_range(0.0..1.0) {
                true  => reflect(unit_direction, hit.normal),
                false => refract(unit_direction, hit.normal, refraction_ratio),
            };

            Some((Ray{
                origin: hit.p,
                direction: direction,
            }, Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            })) 
        },
        Material::Light { albedo } => None,
    }
}

fn reflect(v: Vec3, n: Vec3) -> Vec3 {
    let b = n.scale(v.dot(n));
    v.sub(b.scale(2.0))
}

fn refract(uv: Vec3, n: Vec3, etai_over_etat: f32) -> Vec3{
    let cos_theta = uv.reverse().dot(n).min(1.0);
    let r_out_perp = uv.add_scalar(cos_theta).scale_by_vec(n).scale(etai_over_etat);
    let r_out_parallel_scale = -(1.0 - r_out_perp.len_squared()).abs().sqrt();
    let r_out_parallel = n.scale(r_out_parallel_scale);
    return r_out_perp.add(r_out_parallel);
}

 fn reflectance(cosine: f32, ref_idx: f32) -> f32 {
    // Use Schlick's approximation for reflectance.
    let r0 = (1.0-ref_idx) / (1.0+ref_idx);
    let r0 = r0*r0;
    r0 + (1.0-r0)*(1.0 - cosine).powi(5)
}

fn face_normal(r: &Ray, outward_normal: Vec3) -> (bool, Vec3) {
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    let front_face = r.direction.dot(outward_normal) < 0.0;
    let normal = match front_face {
        true => outward_normal,
        false => outward_normal.reverse(),
    };
    (front_face, normal)
}

#[derive(Clone, Debug, Default)]
pub struct Ray {
    pub origin:    Vec3,
    pub direction: Vec3
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin.add(self.direction.scale(t))
    } 
}

fn degrees_to_radians(degrees: f32) -> f32 {
    return degrees * std::f32::consts::PI / 180.0;
}

struct Interval {
    min: f32,
    max: f32,
}

impl Interval {
    fn empty() -> Self { 
        Self { min: f32::INFINITY, max: f32::NEG_INFINITY } 
    }

    fn universe() -> Self { 
        Self { min: f32::NEG_INFINITY, max: f32::INFINITY } 
    }

    fn contains(&self, x: f32) -> bool {
        return self.min <= x && x <= self.max;
    }

    fn surrounds(&self, x: f32) -> bool {
        return self.min < x && x < self.max;
    }

    fn clamp(&self, x: f32) -> f32 {
        if x < self.min { return self.min; }
        if x > self.max { return self.max; }
        return x;
    }
}


fn rand_unit_vec3(rng: &mut ThreadRng) -> Vec3 {
    Vec3{
        x: rng.gen_range(0.0..1.0),
        y: rng.gen_range(0.0..1.0),
        z: rng.gen_range(0.0..1.0),
    }
}

fn rand_vec3(min: f32, max: f32, rng: &mut ThreadRng) -> Vec3 {
    Vec3{
        x: rng.gen_range(min..max),
        y: rng.gen_range(min..max),
        z: rng.gen_range(min..max),
    }
}

fn rand_unit_vec3_sphere(rng: &mut ThreadRng) -> Vec3 {
    loop {
        let unit = rand_vec3(-1.0, 1.0, rng);
        if unit.len_squared() < 1.0 {
            return unit.unit_vector();
        }
    }
}

fn random_on_hemisphere(normal: Vec3, rng: &mut ThreadRng) -> Vec3 {
    let on_unit_sphere = rand_unit_vec3_sphere(rng);
    match on_unit_sphere.dot(normal) > 0.0 {
        true => on_unit_sphere,
        false => on_unit_sphere.reverse(),// In the same hemisphere as the normal
    } 
}

fn random_in_unit_disk(rng: &mut ThreadRng) -> Vec3 {
    loop {
        let p = Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);
        if p.len_squared() < 1.0 {
            return p;
        }
    }
}

fn load_stl(file: &str, material: &Material) -> Option<Vec<Shape>> {
    let content = std::fs::read(file).ok()?;
    let stl = pk_stl::parse_stl(&content).ok()?;
    Some(stl.triangles.iter()
        .map(|t|Shape::Triangle { v0: t.vertices[0].into(), v1: t.vertices[1].into(), v2: t.vertices[2].into(), material: material.clone() })
        .collect())
}
