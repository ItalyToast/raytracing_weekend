use crate::{vec3::Vec3, Shape, Ray};

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
pub struct Bvh {
    nodes: Vec<BvhNode>,
    tri_idx: Vec<u32>,
    nodes_used: u32,
}

#[derive(Clone, Default)]
struct BvhNode {
    aabb:  Aabb,
    left_first:  u32,
    count: u32,
}

#[derive(Clone, Default)]
struct Aabb {
    min: Vec3,
    max: Vec3,
}

struct Triangle {
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
}

impl Triangle {
    fn centroid(&self) -> Vec3 {
        self.v0.add(self.v1).add(self.v2).scale(0.3333)
    }

    fn intersect(&self, ray: &Ray) -> Option<f32> {
        let edge1 = self.v1.sub(self.v0);
        let edge2 = self.v2.sub(self.v0);
        let h = ray.direction.cross(edge2);
        let a = edge1.dot(h);
        if a > -0.0001 && a < 0.0001 {
            return None; // ray parallel to triangle
        }
        let f = 1.0 / a;
        let s = ray.origin.sub(self.v0);
        let u = f * s.dot(h);
        if u < 0.0 || u > 1.0 { return None; }
        let q = s.cross(edge1);
        let v = f * ray.direction.dot(q);
        if v < 0.0 || u + v > 1.0 { return None; }
        let t = f * edge2.dot(q);
        if t > 0.0001 {
            return Some(t);
        }
        None
    }
}

impl Bvh {
    pub fn new(shapes: Vec<Shape>) -> Bvh {
        let tris : Vec<Triangle> = shapes.iter()
            .map(|s| match s {
                Shape::Triangle { v0, v1, v2, material: _ } => Triangle { v0: v0.clone(), v1: v1.clone(), v2: v2.clone() },
                _ => todo!(),
            }).collect();

        let mut nodes : Vec<BvhNode> = vec![BvhNode::default(); tris.len() * 2];

        let mut root = BvhNode::default();
        root.count = tris.len() as u32;
        nodes[0] = root;

        let mut tri_idx :Vec<u32>= tris.iter()
            .enumerate()
            .map(|(i, _)|i as u32)
            .collect();

        let mut bvh = Bvh { nodes: nodes, tri_idx: vec![], nodes_used: 1 };
        bvh.update_node_bounds(&tris, &tri_idx, 0);
        bvh.subdivide(&tris, &mut tri_idx, 0);
        bvh.tri_idx = tri_idx;
        bvh
    } 

    
// BVHNode bvhNode[N * 2 - 1];
// uint rootNodeIdx = 0, nodesUsed = 1;
 
// void BuildBVH()
// {
//     for (int i = 0; i < N; i++) tri[i].centroid = 
//         (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
//     // assign all triangles to root node
//     BVHNode& root = bvhNode[rootNodeIdx];
//     root.leftChild = root.rightChild = 0;
//     root.firstPrim = 0, root.primCount = N;
//     UpdateNodeBounds( rootNodeIdx );
//     // subdivide recursively
//     Subdivide( rootNodeIdx );
// }

    fn update_node_bounds(&mut self, tri: &Vec<Triangle>, tri_idx: &Vec<u32>, nodeid: u32) {
        let node = &mut self.nodes[nodeid as usize];
        node.aabb.min = Vec3{ x: 1e30, y: 1e30, z: 1e30 };
        node.aabb.max = Vec3{ x: -1e30, y: -1e30, z: -1e30 };
        for i in node.left_first..(node.left_first + node.count) {
            let leaf_tri_idx = tri_idx[i as usize];
            let leaf_tri = &tri[leaf_tri_idx as usize];
            node.aabb.min = node.aabb.min.min(&leaf_tri.v0);
            node.aabb.min = node.aabb.min.min(&leaf_tri.v1);
            node.aabb.min = node.aabb.min.min(&leaf_tri.v2);
            node.aabb.max = node.aabb.max.max(&leaf_tri.v0);
            node.aabb.max = node.aabb.max.max(&leaf_tri.v1);
            node.aabb.max = node.aabb.max.max(&leaf_tri.v2);
        }
    }

    fn subdivide(&mut self, tri: &Vec<Triangle>, tri_idx: &mut Vec<u32>, node_idx: u32)
    {
        // terminate recursion
        let mut node = self.nodes[node_idx as usize].clone();
        if node.count <= 2 {
            return;
        }
        // determine split axis and position
        let extent = node.aabb.max.sub(node.aabb.min);
        let mut axis = 0;
        if extent.y > extent.x { axis = 1 };
        if extent.z > extent.get(axis) { axis = 2 };
        let split_pos = node.aabb.min.get(axis) + extent.get(axis) * 0.5;
        // in-place partition
        let mut i = node.left_first;
        let mut j = i + node.count - 1;
        while i <= j
        {
            if tri[tri_idx[i as usize] as usize].centroid().get(axis) < split_pos {
                i += 1;
            } else {
                tri_idx.swap(i as usize, j as usize);
                j -= 1;
            }
        }
        // abort split if one of the sides is empty
        let left_count = i - node.left_first;
        if left_count == 0 || left_count == node.count {
            return;
        }
        // create child nodes
        let left_child_idx = self.nodes_used;
        self.nodes_used += 1;
        let right_child_idx = self.nodes_used;
        self.nodes_used += 1;
        self.nodes[left_child_idx as usize].left_first = node.left_first;
        self.nodes[left_child_idx as usize].count = left_count;
        self.nodes[right_child_idx as usize].left_first = i;
        self.nodes[right_child_idx as usize].count = node.count - left_count;
        node.left_first = left_child_idx;
        node.count = 0;
        self.nodes[node_idx as usize] = node;
        self.update_node_bounds(tri, &tri_idx, left_child_idx);
        self.update_node_bounds(tri, &tri_idx, right_child_idx);
        // recurse
        self.subdivide(tri, tri_idx, left_child_idx);
        self.subdivide(tri, tri_idx, right_child_idx);
    }

    pub fn traverse(&self, tri: &Vec<Triangle>, tri_idx: &Vec<u32>, ray: &Ray) -> Vec<u32> {
        let mut result :Vec<u32>= vec![];
        self.traverse_inner(tri, tri_idx, ray, 0, &mut result);
        result
    }

    fn traverse_inner(&self, tri: &Vec<Triangle>, tri_idx: &Vec<u32>, ray: &Ray, index: u32, result: &mut Vec<u32>) {
        let node = &self.nodes[index as usize];
        if !node.aabb.intersect(&ray) {
            return;
        }
        if node.is_leaf()
        {
            for i in node.left_first..(node.left_first + node.count) {
                if let Some(_) = tri[tri_idx[i as usize] as usize].intersect(&ray) {
                    result.push(i);
                };
            }
        }
        else
        {
            self.traverse_inner(tri, tri_idx, &ray, node.left_first, result);
            self.traverse_inner(tri, tri_idx, &ray, node.left_first + 1, result);
        }
    }
}

impl BvhNode {
    fn is_leaf(&self) -> bool {
        self.count > 0
    }
}

impl Aabb {
    fn intersect(&self, ray: &Ray) -> bool {
        //bool IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
        let tx1 = (self.min.x - ray.origin.x) / ray.direction.x;
        let tx2 = (self.max.x - ray.origin.x) / ray.direction.x;
        let tmin = tx1.min(tx2);
        let tmax = tx1.max(tx2);
        let ty1 = (self.min.y - ray.origin.y) / ray.direction.y;
        let ty2 = (self.max.y - ray.origin.y) / ray.direction.y;
        let tmin = tmin.max(ty1.min(ty2 ));
        let tmax = tmax.min(ty1.max(ty2 ));
        let tz1 = (self.min.z - ray.origin.z) / ray.direction.z;
        let tz2 = (self.max.z - ray.origin.z) / ray.direction.z;
        let tmin = tmin.max(tz1.min(tz2));
        let tmax = tmax.min(tz1.max(tz2));
        //return tmax >= tmin && tmin < ray.t && tmax > 0.0;
        return tmax >= tmin && tmax > 0.0; //TODO ray.t
    }
}

impl From<Shape> for Aabb {
    fn from(value: Shape) -> Self {
        match value {
            Shape::Triangle { v0, v1, v2, material } => {
                Aabb{
                    min: v0.min(&v1).min(&v2),
                    max: v0.max(&v1).max(&v2),
                }
            },
            _ => todo!(),
        }
    }
}