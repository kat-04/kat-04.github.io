# Rendering

## Installation/Run Process

You should be able to run `make` within the `render` folder then run the executable `./GOL3D` that is created after.

If issues occur, then you may need to run one of the following commands:

1) `brew install glfw3`
2) `brew install glew`
3) `brew install pkg-config`
4) Not a command, but you may need XCode?

## How to navigate the window

So after you run `./GOL3D`, a window will pop up that has a rotating voxel cube in the middle. The red wireframe border is the "bounding box" and the cubes inside are the cells.

The initial state of the scene is with an orbital camera and automatic frame increments (set to every 5 seconds).

### Toggle Options

To change the camera mode between free/orbiting, press `C` for camera.

To change the automatic frames to manual frames (mainly for debugging), press `F` for frame.

### Navigation

For the free camera (3rd person), there are many different keys that manage the camera rotation and panning:
- `WASD` pans the camera (`A` = camera to the left, `D` = to the right, `W` = forward, `S` = backward
- `Up, Down, Left, Right` arrow keys rotate the camera in that direction
- Using 2 fingers to slide up on the trackpad zooms in, slide down zooms out
- Moving the cursor within the window rotates the object with respect to your cursor location

For the frames:
- To go one frame ahead, press `.`
- To move back a frame, press `,`

## Other resources

You can find the GitHub for raylib [here](https://github.com/raysan5/raylib) but examples may be more helpful and can be found [here](https://www.raylib.com/examples.html).

## Later Additions

- Add ambient occlusion to easily distinguish depth of voxels
