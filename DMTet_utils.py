import numpy as np
import pymeshlab as pml
import torch
import kaolin
import numpy as np
from dmtet_network import Decoder
import open3d as o3d
import os


device = 'cuda'
lr = 1e-3
laplacian_weight = 0.1
iterations = 5000
save_every = 100
multires = 2
grid_res = 128


#prepare loss and regularizer
# Laplacian regularization using umbrella operator (Fujiwara / Desbrun).
# https://mgarland.org/class/geom04/material/smoothing.pdf
def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1,3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1,3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1,3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)

def loss_f(mesh_verts, mesh_faces, points, it):
    #######################################
    w=0.7
    ########################################
    pred_points = kaolin.ops.mesh.sample_points(mesh_verts.unsqueeze(0), mesh_faces, 50000)[0][0]
    chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), points.unsqueeze(0)).mean()
    if it > iterations//2:
        lap = laplace_regularizer_const(mesh_verts, mesh_faces)
        return w*chamfer + lap * laplacian_weight
    return w*chamfer


#save in obj
def save_obj(vertices, faces, filename):
    """
    Save mesh to .obj file.
    
    vertices: (N, 3) tensor or numpy array containing vertex coordinates
    faces: (M, 3) tensor or numpy array containing triangle indices
    filename: str, path to save the .obj file
    """
    with open(filename, 'w') as f:
        # Write vertices
        for vert in vertices:
            f.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
        
        # Write faces
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # 1-based indexing for faces



def return_mesh(points, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    scaling_factor = 1
    points_ = np.asarray(pcd.points) * scaling_factor
    pcd.points = o3d.utility.Vector3dVector(points_)

    if normals is None:
        pcd.estimate_normals()


    points = torch.tensor(points_, dtype=torch.float32, device='cuda')

    if points.shape[0] > 100000:
        idx = list(range(points.shape[0]))
        np.random.shuffle(idx)
        idx = torch.tensor(idx[:100000], device=points.device, dtype=torch.long)    
        points = points[idx]

    # The reconstructed object needs to be slightly smaller than the grid to get watertight surface after MT.
    points = kaolin.ops.pointcloud.center_points(points.unsqueeze(0), normalize=True).squeeze(0) * 0.9

    #load grid
    tet_verts = torch.tensor(np.load('./{}_verts.npz'.format(grid_res))['data'], dtype=torch.float, device=device)


    tets = torch.tensor(([np.load('./{}_tets_{}.npz'.format(grid_res, i))['data'] for i in range(4)]), dtype=torch.long, device=device).permute(1,0)
    
    points_min = points.min(0)[0]
    points_max = points.max(0)[0]
    tet_verts_min = tet_verts.min(0)[0]
    tet_verts_max = tet_verts.max(0)[0]

    # points를 tet_verts의 범위에 맞춰 정규화
    points = (points - points_min) / (points_max - points_min)
    points = points * (tet_verts_max - tet_verts_min) + tet_verts_min

    # Initialize model and create optimizer
    model = Decoder(multires=multires).to(device)
    model.pre_train_sphere(1000)

    #set optimizer
    vars = [p for _, p in model.named_parameters()]
    optimizer = torch.optim.Adam(vars, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(0.0, 10**(-x*0.0002))) # LR decay over time

    #train
    for it in range(iterations):
        pred = model(tet_verts) # predict SDF and per-vertex deformation
        sdf, deform = pred[:,0], pred[:,1:]


        #print(f"Min SDF: {sdf.min().item()}, Max SDF: {sdf.max().item()}")
        if(sdf.min().item()>0):
            print("<")
            sdf = sdf - sdf.mean() #중심화
        if(sdf.max().item()<0):
            print(">")
            sdf = sdf + sdf.mean()

        verts_deformed = tet_verts + torch.tanh(deform) / grid_res # constraint deformation to avoid flipping tets
        mesh_verts, mesh_faces = kaolin.ops.conversions.marching_tetrahedra(verts_deformed.unsqueeze(0), tets, sdf.unsqueeze(0)) # running MT (batched) to extract surface mesh
        mesh_verts, mesh_faces = mesh_verts[0], mesh_faces[0]

        loss = loss_f(mesh_verts, mesh_faces, points, it)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (it) % save_every == 0 or it == (iterations - 1): 
            print ('Iteration {} - loss: {}, # of mesh vertices: {}, # of mesh faces: {}'.format(it, loss, mesh_verts.shape[0], mesh_faces.shape[0]))


    #save_obj(mesh_verts.detach().cpu().numpy(), mesh_faces.detach().cpu().numpy(), f"mesh_{it+1}.obj")

    return mesh_verts, mesh_faces



