use crate::{vec3::Vec3, Shape, Ray};

// https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/
struct Bvh {
    nodes: Vec<BvhNode>,
}

#[derive(Clone)]
struct BvhNode {
    aabb:  Aabb,
    index: u32,
    left:  u32,
    right:  u32,
    count: u32,
}

#[derive(Clone)]
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

    fn intersect(&self, ray: &Ray) -> bool {
        todo!()
    }
}

impl Bvh {
    fn new(shapes: Vec<Shape>) -> Bvh {
        let tris : Vec<Triangle> = shapes.iter()
            .map(|s| match s {
                Shape::Triangle { v0, v1, v2, material: _ } => Triangle { v0: v0.clone(), v1: v1.clone(), v2: v2.clone() },
                _ => todo!(),
            }).collect();

        let mut nodes : Vec<BvhNode> = vec![BvhNode{ 
            aabb:Aabb { min: Vec3 { x: 0.0, y: 0.0, z: 0.0 }, max: Vec3 { x: 0.0, y: 0.0, z: 0.0 } }, 
            index: 0, 
            left: 0, 
            right: 0, 
            count: 0 
        };tris.len() * 2];

        
        let root = BvhNode{
            aabb: Aabb { min: Vec3 { x: 0.0, y: 0.0, z: 0.0 }, max: Vec3 { x: 0.0, y: 0.0, z: 0.0 } },
            index: 0,
            left: 0,
            right: 0,
            count: tris.len() as u32,
        };

        Bvh { nodes: nodes }
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

    fn update_node_bounds(&mut self, tri: Vec<Triangle>, tri_idx: Vec<u32>, nodeid: u32) {
        let node = &mut self.nodes[nodeid as usize];
        node.aabb.min = Vec3{ x: 1e30, y: 1e30, z: 1e30 };
        node.aabb.max = Vec3{ x: -1e30, y: -1e30, z: -1e30 };
        for i in node.index..(node.index + node.count) {
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

    fn subdivide(&self, tri: Vec<Triangle>, tri_idx: Vec<u32>, nodeIdx: u32)
    {
        // terminate recursion
        let node = self.nodes[nodeIdx as usize];
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
        let mut i = node.left;
        let mut j = i + node.count - 1;
        while i <= j
        {
            if tri[tri_idx[i]].centroid[axis] < split_pos {
                i += 1;
            } else {
                tri_idx.swap(i as usize, j as usize);
                j -= 1;
                //swap( triIdx[i], triIdx[j--] );
            }
        }
        // abort split if one of the sides is empty
        let left_count = i - node.firstTriIdx;
        if left_count == 0 || left_count == node.count {
            return;
        }
        // create child nodes
        let left_child_idx = nodesUsed + 1;
        let right_child_idx = nodesUsed + 1;
        self.nodes[left_child_idx as usize].firstTriIdx = node.firstTriIdx;
        self.nodes[left_child_idx as usize].count = left_count;
        self.nodes[right_child_idx as usize].firstTriIdx = i;
        self.nodes[right_child_idx as usize].count = node.count - left_count;
        node.left = left_child_idx;
        node.count = 0;
        self.update_node_bounds(tri, tri_idx, left_child_idx);
        self.update_node_bounds(tri, tri_idx, right_child_idx);
        // recurse
        self.subdivide(tri, tri_idx, left_child_idx);
        self.subdivide(tri, tri_idx, right_child_idx);
    }

    fn traverse(&self, tri: &Vec<Triangle>, tri_idx: &Vec<u32>, ray: &Ray, index: u32) -> Vec<u32> {
        let node = &self.nodes[index as usize];
        if !node.aabb.intersect(&ray) {
            return vec![];
        }
        if node.is_leaf()
        {
            for i in node.index..(node.index + node.count) {
                tri[tri_idx[i as usize] as usize].intersect(&ray);
            }
        }
        else
        {
            self.traverse(tri, tri_idx, &ray, node.left);
            self.traverse(tri, tri_idx, &ray, node.right);
        }
        todo!()
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