## Output Format

Each captured frame uses a zero-padded frame index (`xxxxxx`) and produces the following files:

1. `rgb/xxxxxx_rgb.png`  
   RGB image of the scene.

2. `depth/xxxxxx_depth.png`  
   16-bit depth image. Depth values are stored in millimeters.

3. `mask/xxxxxx_<gear>.png`  
   Binary object mask for each visible gear in the frame.

   - `0` = background
   - `255` = object

   If a gear is not visible in the frame, no mask file is generated for that object.

4. `gt/xxxxxx.json`  
   Ground-truth 6D pose of each visible gear in the camera coordinate frame using the OpenCV convention.

   ```json
   {
     "<gear_label>": {
       "cam_R_m2c": [3x3 rotation matrix, flattened row-major],
       "cam_t_m2c": [x, y, z]
     }
   }
cam_R_m2c: model-to-camera rotation matrix
cam_t_m2c: model-to-camera translation in millimeters

These values are read directly from Isaac Sim and can be used as training labels for 6D pose estimation models such as FFB6D.

cam_K.json
Camera intrinsic parameters. This file is written once because the camera resolution and field of view remain fixed during the capture run.

{
  "fx": ...,
  "fy": ...,
  "cx": ...,
  "cy": ...,
  "K": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ]
}

The intrinsic matrix is used for projecting 3D points into image coordinates and back-projecting depth pixels into 3D during training and validation.
