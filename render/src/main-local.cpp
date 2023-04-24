#include "raylib.h"
#include "mesh.h"
#include "raymath.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <time.h>

#define RLIGHTS_IMPLEMENTATION
#include "../libs/raylib/raylib/include/rlights.h"

#if defined(PLATFORM_DESKTOP)
    #define GLSL_VERSION            330
#else   // PLATFORM_RPI, PLATFORM_ANDROID, PLATFORM_WEB
    #define GLSL_VERSION            100
#endif

#define SECONDS_PER_FRAME 0.1


//------------------------------------------------------------------------------------
// Logging
//------------------------------------------------------------------------------------
void log(int msgType, const char *text, va_list args)
{
    char timeStr[64] = { 0 };
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);

    strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("[%s] ", timeStr);

    switch (msgType)
    {
        case LOG_INFO: printf("[INFO] : "); break;
        case LOG_ERROR: printf("[ERROR]: "); break;
        case LOG_WARNING: printf("[WARN] : "); break;
        case LOG_DEBUG: printf("[DEBUG]: "); break;
        default: break;
    }

    vprintf(text, args);
    printf("\n");
}


//------------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------------
int main()
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1600;
    const int screenHeight = 900;

    // Set logging method and configuration
    SetTraceLogCallback(log);
    /* SetConfigFlags(FLAG_MSAA_4X_HINT); // if available */
    InitWindow(screenWidth, screenHeight, "3D Conway's Game of Life");

    // Initialize the camera
    /* Vector3 starting_pos = (Vector3){ 30.0f, 10.0f, 10.0f }; */
    Vector3 starting_pos = (Vector3){ 0.0f, 0.0f, 0.0f };
    Camera3D camera = { 0 };
    camera.position = starting_pos; // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type

    // Cube mesh properties (width, height, length)
    Mesh cube = GenMeshCube(1.0f, 1.0f, 1.0f);


    //--------------------------------------------------------------------------------------
    // SHADERS
    Shader shader = LoadShader(TextFormat("libs/raylib/examples/shaders/resources/shaders/glsl%i/lighting_instancing.vs", GLSL_VERSION),
                               TextFormat("libs/raylib/examples/shaders/resources/shaders/glsl%i/lighting.fs", GLSL_VERSION));

    // Get shader locations
    shader.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shader, "mvp");
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");
    shader.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocationAttrib(shader, "instanceTransform");

    // Set shader value: ambient light level
    int ambientLoc = GetShaderLocation(shader, "ambient");
    float shader_attr[4] = { 0.2f, 0.2f, 0.2f, 1.0f };
    SetShaderValue(shader, ambientLoc, shader_attr, SHADER_UNIFORM_VEC4);
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // LIGHT(S)

    // Color of the light: can either be preset or custom: (Color){ r, g, b, alpha }
    // Where 0 <= r, g, b, alpha <= 255 and alpha is opacity
    Color light_color1 = (Color) { 255, 87, 115, 255 };
    Light light1 = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color1, shader);
    Color light_color2 = (Color) { 208, 106, 252, 255 };
    Light light2 = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color2, shader);
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // MATERIALS

    Color mat_color = WHITE;
    Material matInstances = LoadMaterialDefault();
    matInstances.shader = shader;
    matInstances.maps[MATERIAL_MAP_DIFFUSE].color = mat_color;
    //--------------------------------------------------------------------------------------


    // Limit cursor to only within the window
    DisableCursor();

    SetTargetFPS(60);


    //--------------------------------------------------------------------------------------
    // INITIALIZING VARIABLES

    // Frame number -- used to get vertices of a certain frame
    int frame = 0;

    // Check if first frame
    bool first = true;

    // Time of previous frame (used when letting frames run)
    double lastFrameTime = 0;

    // OS Stream for filename
    std::ostringstream oss;
    std::string filename;

    // parse_data return format
    std::tuple<int, std::vector<Matrix> > info;

    // Variables for generating mesh instance
    Matrix *transforms;
    int size;
    int num_blocks;
    Vector3 origin = Vector3Zero();

    // Toggles for different modes
    // Manual mode: use arrow keys to move frame by frame
    // If false, will increment frame every 5 seconds
    // Press "F" to toggle frame mode
    bool manual = false;

    // Free mode: freely zoom in/out and pan around the scene
    // If false, automatically default to orbit mode from current position
    // Press "C" to toggle camera into FREE/ORBITAL mode
    bool free = false;

    bool updated = true;
    //--------------------------------------------------------------------------------------



    // MAIN LOOP
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        //----------------------------------------------------------------------------------
        // UPDATE 
        
        // Move camera around the scene
        if (IsKeyPressed(KEY_C)) free = !free;
        if (!free) {
            UpdateCamera(&camera, CAMERA_ORBITAL);
        } else {
            UpdateCamera(&camera, CAMERA_THIRD_PERSON);
        }

        // Center camera
        if (IsKeyPressed(KEY_Z)) {
            camera.position = Vector3Add(origin, starting_pos);
            camera.target = origin;
            camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
            camera.fovy = 45.0f;
        }

        // Update shader and light accordingly to follow camera movement
        float cameraPos[3] = { camera.position.x, camera.position.y, camera.position.z };
        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
        light1.position = (Vector3){camera.position.x, camera.position.y, -camera.position.z};
        light2.position = (Vector3){-camera.position.x, camera.position.y, camera.position.z};
        UpdateLightValues(shader, light1);
        UpdateLightValues(shader, light2);
        //----------------------------------------------------------------------------------


        //----------------------------------------------------------------------------------
        // DRAW

        if (first) lastFrameTime = GetTime();

        BeginDrawing();

            ClearBackground(BLACK);

            BeginMode3D(camera);

            // Toggle frame increment method depending on if SPACE is pressed
            if (IsKeyPressed(KEY_F)) manual = !manual;

            if (!manual) {
                // Increment frame every SECONDS_PER_FRAME seconds if not at last frame
                if (GetTime() - lastFrameTime > SECONDS_PER_FRAME) {
                    frame++;
                    updated = true;
                    lastFrameTime = GetTime();
                }
            } else {
                // Increment frame using WASD keys
                if (IsKeyPressed(KEY_PERIOD)) {
                    frame++;
                    updated = true;
                }
                if (IsKeyPressed(KEY_COMMA)) {
                    frame--;
                    updated = true;
                }
            }

            // clear OS Stream
            oss.str("");
            oss.clear();

            // Get filename
            oss << "../output-files/frame" << frame << ".txt";
            filename = oss.str();

            // Check if file exists
            std::ifstream file (filename);

            // If file does not exist, then decrement frame and leave at frame (done = true)
            if (!file.is_open()) {
                /* std::cout << "frame " << frame << " does not exist" << std::endl; */
                if (frame < 0) {
                    frame++;
                    updated = false;
                } else {
                    frame--;
                    updated = false;
                }
                oss.str("");
                oss.clear();
                oss << "../output-files/frame" << frame << ".txt";
                filename = oss.str();
            } else {
                /* std::cout << "currently reading frame " << frame << std::endl; */
            }

            if (updated) {
                // Get transforms and other data from the file
                info = parse_data(filename);
                transforms = &(std::get<1>(info))[0];
                size = std::get<0>(info);
                num_blocks = (std::get<1>(info)).size();
                updated = false;
                /* std::cout << "Number of blocks: " << num_blocks << std::endl; */
            }

            // Draw outline (bounding box)
            if (!first) {
                /* std::cout << "Origin: (" << origin.x << ", " << origin.y << ", " << origin.z << ")" << std::endl; */
                /* std::cout << "Size: " << size << std::endl; */
                DrawCubeWires(origin, size, size, size, RED);
                // Draw mesh instances
                DrawMeshInstanced(cube, matInstances, transforms, num_blocks);
            }

            if (first) {
                // sets camera position in accordance to the size of the cube during first frame
                camera.position = (Vector3) { (float)size, size / 4.f, 2.f * size };
                starting_pos = camera.position;
                /* std::cout << "Camera Position: (" << starting_pos.x << ", " << starting_pos.y << ", " << starting_pos.z << ")" << std::endl; */
            }

            first = false;

            EndMode3D();

            // Uncomment if you want to see FPS
            DrawFPS(10, 10);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    CloseWindow();        // Close window and OpenGL context

    return 0;
}
