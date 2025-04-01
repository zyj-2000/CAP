import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# Step 1: Load the OBJ file and display the 3D model
def load_obj(file_path):
    mesh = trimesh.load(file_path)
    return mesh


# Step 2: Calculate the center of the 3D model based on the mesh vertices
def calculate_center(mesh):
    # Calculate the center by averaging the vertices' coordinates
    center = np.mean(mesh.vertices, axis=0)
    return center


# Step 3: Visualize the 3D model with slicing planes centered at the center of the model
def visualize_3d_model_with_slices(mesh, z_range, center, slices_shown=True):
    # Create an interactive 3D plot
    scene = mesh.scene()

    # Visualize slices (CT-like projections) as planes in the 3D model
    for z in z_range:
        # Create a slicing plane at each Z level, centered at the center of the model
        slice_plane = trimesh.creation.box(extents=[mesh.bounds[1][0] - mesh.bounds[0][0],
                                                    mesh.bounds[1][1] - mesh.bounds[0][1],
                                                    0.01])  # Thin plane
        slice_plane.apply_translation([center[0], center[1], z])

        # Add the slice plane to the scene
        scene.add_geometry(slice_plane, node_name=f"Slice at Z={z:.2f}")

    if slices_shown:
        # Display the scene with the slices in place
        scene.show()


# Step 4: Slice the mesh at different Z values and create a 2D projection (CT slice)
def slice_mesh_at_z(mesh, z):
    # Project the mesh to the X-Y plane by slicing along the Z axis
    slice_mesh = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if slice_mesh is None:
        return None

    # Get the 2D polygon of the slice
    polygon = slice_mesh.vertices
    return polygon


# Step 5: Visualize CT projections (slices) in 2D, while showing slicing track in 3D
def visualize_ct_projections_with_track(mesh, save_path,num_slices=10, step_size=0.1):
    # Get the bounds of the mesh to determine the Z range
    z_min = mesh.bounds[0][2]  # Min Z value (bottom of the mesh)
    z_max = mesh.bounds[1][2]# Max Z value (top of the mesh)

    # Calculate the center of the 3D model
    center = calculate_center(mesh)

    # Create slices at different Z levels from top to bottom
    z_range = np.linspace(z_max, z_min, num_slices)

    # Visualize the 3D model with slicing planes
    print("Visualizing the 3D model with slicing planes centered at the center of the model...")
    # visualize_3d_model_with_slices(mesh, z_range, center)

    # Loop through the Z values and plot each slice
    fig, axes = plt.subplots(1, num_slices, figsize=(num_slices * 2, 2))

    for i, z in enumerate(z_range):
        # Slice the mesh at the current Z level
        polygon = slice_mesh_at_z(mesh, z)

        if polygon is None:
            continue

        # Create an image (blank canvas) for each slice
        img = np.zeros((100, 100), dtype=np.uint8)

        # Convert 2D polygon to 2D image (by scaling the coordinates to fit in the canvas)
        x_vals = polygon[:, 0]
        y_vals = polygon[:, 1]

        # Normalize the polygon's coordinates to the range [0, 99] (for a 100x100 image)
        x_min, x_max = np.min(x_vals), np.max(x_vals)
        y_min, y_max = np.min(y_vals), np.max(y_vals)

        x_scaled = np.interp(x_vals, (x_min, x_max), (0, 99))
        y_scaled = np.interp(y_vals, (y_min, y_max), (0, 99))

        # Fill the polygon in the image
        for j in range(len(x_scaled)):
            img[int(x_scaled[j]), int(y_scaled[j])] = 255

        # Display the slice on the subplot
        axes[i].imshow(img, cmap=cm.gray)
        axes[i].axis('off')
        axes[i].set_title(f"Z = {z:.2f}")

        # Save the image with appropriate name
        image_name = os.path.join(save_path, f"{i}.png")
        plt.imsave(image_name, img, cmap=cm.gray)

    # plt.tight_layout()
    # plt.show()


# Example usage
def main(obj_file_path,tarr_path):
    mesh = load_obj(obj_file_path)

    # Visualize CT-like projections (2D slices) with track in 3D
    print("Visualizing CT slices with track...")
    visualize_ct_projections_with_track(mesh,tarr_path)


# Run the code (replace 'your_model.obj' with the path to your OBJ file)
if __name__ == "__main__":
    obj_path = "/DATA/zyj/dataset/3DGCQA_master/obj/"
    target_path = "/DATA/zyj/dataset/3DGCQA_master/ct/slice10"
    
    objs_dir = os.listdir(obj_path)
    for obj_dir in objs_dir:
        new_dir = os.path.join(obj_path,obj_dir)
        objs_file = os.listdir(new_dir)
        for obj_file in objs_file:
            obj_file_path = os.path.join(new_dir,obj_file,'model.obj')
            tarr_path = os.path.join(target_path,obj_dir,obj_file)
            os.makedirs(tarr_path, exist_ok=True)
            print(obj_file_path,tarr_path)
    # obj_file_path = "D:\Project\CTPCQA\objmodel\objmodel\model.obj"  # Replace with your actual file path
            main(obj_file_path,tarr_path)
