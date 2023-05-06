#include "raylib.h"
#include "mesh.h"
#include "raymath.h"
#include <sstream>
#include <tuple>
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
    const int screenWidth = 2880;
    const int screenHeight = 1160;

    //--------------------------------------------------------------------------------------
    // LOAD VALUES
    std::tuple<int, int> data;
    data = parse_init("../output-files/frame_init.txt");
    int size = std::get<0>(data);
    int numStates = std::get<1>(data);
    //--------------------------------------------------------------------------------------

    // Set logging method and configuration
    SetTraceLogCallback(log);
    /* SetConfigFlags(FLAG_MSAA_4X_HINT); // if available */
    InitWindow(screenWidth, screenHeight, "3D Conway's Game of Life");

    // Initialize the camera
    Camera3D camera = { 0 };
    camera.position = (Vector3) { 0, size / 4.f, 2.2f * size };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 45.0f;                                // Camera field-of-view Y
    camera.projection = CAMERA_PERSPECTIVE;             // Camera projection type
    Vector3 starting_pos = camera.position;

    // Cube mesh properties (width, height, length)
    Mesh cube = GenMeshCube(1.01f, 1.01f, 1.01f);
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
    float shader_attr[4] = { 4.5f, 4.5f, 4.4f, 4.f };
    SetShaderValue(shader, ambientLoc, shader_attr, SHADER_UNIFORM_VEC4);
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // LIGHT(S)

    // Color of the light: can either be preset or custom: (Color){ r, g, b, alpha }
    // Where 0 <= r, g, b, alpha <= 255 and alpha is opacity
    Color light_color = (Color) { 255, 255, 255, 255 };
    Light light = CreateLight(LIGHT_POINT, camera.position, Vector3Zero(), light_color, shader);
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // MATERIALS

    // Colors in backwards order of states (1 -> n - 1)
    // Can custom set these values using RGBA
    // For n states, set n - 1 colors, as one state is always the dead state
    std::vector<Color> mat_color;

    mat_color.push_back((Color) { 255, 255, 218, 255 });


    Material matInstances[numStates - 1];
    for (int i = 0; i < numStates - 1; i++) {
        matInstances[i] = LoadMaterialDefault();
        matInstances[i].shader = shader;
        matInstances[i].maps[MATERIAL_MAP_DIFFUSE].color = mat_color[i];
    }
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
    std::map<int, std::vector<Matrix> > info;

    // Variables for generating mesh instance
    std::vector<Matrix*> transforms;
    std::vector<int> num_blocks;
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

    // Toggles the bounding box on or off
    // Press "B" to turn the wireframe on or off
    bool box = false;

    // Checks whether or not the frame has been updated (incremented or reset)
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
        
        light.position = (Vector3){sqrt(2.f) * camera.position.x / 2.f, 4 * camera.position.y,  -2 * camera.position.z / 2.f};
        UpdateLightValues(shader, light);
        //----------------------------------------------------------------------------------


        //----------------------------------------------------------------------------------
        // DRAW

        if (first) lastFrameTime = GetTime();

        BeginDrawing();

            ClearBackground(BLACK);

            BeginMode3D(camera);

            // Toggle frame increment method depending on if SPACE is pressed
            if (IsKeyPressed(KEY_F)) manual = !manual;

            // Toggle bounding box on or off
            if (IsKeyPressed(KEY_B)) box = !box;

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

            // Resets render to frame 0
            if (IsKeyPressed(KEY_R)) {
                frame = 0;
                updated = true;
            }

            // Clear OS Stream
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
                    updated = false;
                } else {
                    frame--;
                    updated = false;
                }
                oss.str("");
                oss.clear();
                oss << "../output-files/frame" << frame << ".txt";
                filename = oss.str();
            }

            if (updated) {
                transforms.clear();
                num_blocks.clear();
                // Get transforms and other data from the file
                info = parse_data(filename, numStates, size);
                for (int i = 1; i < numStates; i++) {
                    transforms.push_back(&(info[i])[0]);
                    num_blocks.push_back((info[i]).size());
                }
                updated = false;
            }

            // Draw outline (bounding box)
            if (box) {
                DrawCubeWires(origin, size, size, size, DARKBLUE);
            }
            // Draw mesh instances
            for (int i = 0; i < numStates - 1; i++) {
                DrawMeshInstanced(cube, matInstances[i], transforms.at(i), num_blocks.at(i));
            }

            first = false;

            EndMode3D();

            // Uncomment if you want to see FPS
            DrawFPS(10, 10);
            DrawText(TextFormat("Frame: %i", frame), 10, 40, 25, SKYBLUE);


        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    UnloadShader(shader);
    CloseWindow();        // Close window and OpenGL context

    return 0;
}
