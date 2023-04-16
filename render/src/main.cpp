#include "raylib.h"
#include "mesh.h"
#include "raymath.h"
#include <sstream>
#include <iostream>
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


// Logging
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
// Program main entry point
//------------------------------------------------------------------------------------
int main()
{
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 1600;
    const int screenHeight = 900;

    // Set logging method and configuration
    SetTraceLogCallback(log);
    SetConfigFlags(FLAG_MSAA_4X_HINT); // if available
    InitWindow(screenWidth, screenHeight, "3D Conway's Game of Life");

    // Initialize the camera
    Vector3 starting_pos = (Vector3){ 30.0f, 10.0f, 10.0f };
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
    SetShaderValue(shader, ambientLoc, (float[4]){ 0.2f, 0.2f, 0.2f, 1.0f }, SHADER_UNIFORM_VEC4);
    //--------------------------------------------------------------------------------------


    // LIGHT(S)
    //--------------------------------------------------------------------------------------
    // Color of the light: can either be preset or custom: (Color){ r, g, b, alpha }
    // Where 0 <= r, g, b, alpha <= 255 and alpha is opacity
    Color light_color = SKYBLUE;
    Light light = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color, shader);
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // MATERIALS
    Color mat_color = (Color){ 135, 60, 190, 150 };
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
    Vector3 origin;

    // Toggles for different modes
    // Manual mode: use arrow keys to move frame by frame
    // If false, will increment frame every 5 seconds
    // Press "F" to toggle frame mode
    bool manual = false;

    // Free mode: freely zoom in/out and pan around the scene
    // If false, automatically default to orbit mode from current position
    // Press "C" to toggle camera into FREE/ORBITAL mode
    bool free = false;
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
        }

        // Update shader and light accordingly to follow camera movement
        float cameraPos[3] = { camera.position.x, camera.position.y, camera.position.z };
        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
        light.position = camera.position;
        UpdateLightValues(shader, light);
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(BLACK);

            BeginMode3D(camera);

            // Toggle frame increment method depending on if SPACE is pressed
            if (IsKeyPressed(KEY_F)) manual = !manual;

            if (!manual) {
                // Increment frame every 5 seconds if not at last frame
                if (GetTime() - lastFrameTime > 5) {
                    frame++;
                    lastFrameTime = GetTime();
                }
            } else {
                // Increment frame using WASD keys
                if (IsKeyPressed(KEY_PERIOD)) frame++;
                if (IsKeyPressed(KEY_COMMA)) frame--;
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
                if (frame < 0) {
                    frame++;
                } else {
                    frame--;
                }
                oss.str("");
                oss.clear();
                oss << "../output-files/frame" << frame << ".txt";
                filename = oss.str();
            }

            // Get transforms and other data from the file
            info = parse_data(filename);
            transforms = &(std::get<1>(info))[0];
            size = std::get<0>(info);
            num_blocks = (std::get<1>(info)).size();
            origin = { (float)(size / 2), (float)(size / 2), (float)(size / 2) };

            // Draw outline (bounding box)
            DrawCubeWires(origin, size, size, size, RED);
            // Draw mesh instances
            DrawMeshInstanced(cube, matInstances, transforms, num_blocks);

            EndMode3D();

            // Uncomment if you want to see FPS
            // DrawFPS(10, 10);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}
