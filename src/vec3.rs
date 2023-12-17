

#[derive(Copy, Clone, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn origin() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn reverse(self) -> Vec3 {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }

    pub fn get(&self, i: u32) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Out of bounds of vec"),
        }
    }

    pub fn add_scalar(self, other : f32) -> Vec3 {
        Vec3 { 
            x: self.x + other, 
            y: self.y + other, 
            z: self.z + other 
        }
    }

    pub fn add(self, other : Vec3) -> Vec3 {
        Vec3 { 
            x: self.x + other.x, 
            y: self.y + other.y, 
            z: self.z + other.z 
        }
    }

    pub fn sub(self, other : Vec3) -> Vec3 {
        Vec3 { 
            x: self.x - other.x, 
            y: self.y - other.y, 
            z: self.z - other.z 
        }
    }

    pub fn scale(self, scalar : f32) -> Vec3 {
        Vec3 { 
            x: self.x * scalar, 
            y: self.y * scalar, 
            z: self.z * scalar 
        }
    }

    pub fn scale_by_vec(self, other : Vec3) -> Vec3 {
        Vec3 { 
            x: self.x * other.x, 
            y: self.y * other.y, 
            z: self.z * other.z 
        }
    }

    pub fn div_scale(self, scalar : f32) -> Vec3 {
        Vec3 { 
            x: self.x / scalar, 
            y: self.y / scalar, 
            z: self.z / scalar 
        }
    }

    pub fn len(self) -> f32 {
        self.len_squared().sqrt()
    }

    pub fn len_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn dot(self, other : Vec3) -> f32 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }

    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3 { 
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn unit_vector(self) -> Vec3 {
        self.div_scale(self.len())
    }

    pub fn near_zero(self) -> bool {
        let s = 1e-8;
        self.x.abs() < s && 
        self.y.abs() < s && 
        self.z.abs() < s
    }
}

/*
// Vector Utility Functions

inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline vec3 operator*(const vec3 &v, double t) {
    return t * v;
}

inline vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}

inline double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
    + u.e[1] * v.e[1]
    + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
    }
    
    inline vec3 unit_vector(vec3 v) {
        return v / v.length();
    }
    
    #endif
*/