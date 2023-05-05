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
    /* Vector3 starting_pos = (Vector3){ 30.0f, 10.0f, 10.0f }; */
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

    /* Shader shader = LoadShader(0, TextFormat("resources/shaders/glsl%i/raymarching.fs", GLSL_VERSION)); */

    // Get shader locations for required uniforms
    /* int viewEyeLoc = GetShaderLocation(shader, "viewEye"); */
    /* int viewCenterLoc = GetShaderLocation(shader, "viewCenter"); */
    /* int runTimeLoc = GetShaderLocation(shader, "runTime"); */
    /* int resolutionLoc = GetShaderLocation(shader, "resolution"); */

    /* // Get shader locations */
    shader.locs[SHADER_LOC_MATRIX_MVP] = GetShaderLocation(shader, "mvp");
    shader.locs[SHADER_LOC_VECTOR_VIEW] = GetShaderLocation(shader, "viewPos");
    shader.locs[SHADER_LOC_MATRIX_MODEL] = GetShaderLocationAttrib(shader, "instanceTransform");

    /* float resolution[2] = { (float)screenWidth, (float)screenHeight }; */
    /* SetShaderValue(shader, resolutionLoc, resolution, SHADER_UNIFORM_VEC2); */

    /* float runTime = 0.0f; */

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
    Light light1 = CreateLight(LIGHT_POINT, camera.position, Vector3Zero(), light_color, shader);
    /* Light light2 = CreateLight(LIGHT_POINT, Vector3Negate(camera.position), camera.up, light_color, shader); */
    /* /1* Color light_color2 = (Color) { 208, 106, 252, 255 }; *1/ */
    /* Light light2 = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color, shader); */
    /* Light light3 = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color, shader); */
    /* Light light4 = CreateLight(LIGHT_DIRECTIONAL, camera.position, Vector3Zero(), light_color, shader); */
    //--------------------------------------------------------------------------------------


    //--------------------------------------------------------------------------------------
    // MATERIALS

    std::vector<Color> mat_color;
    /* for (int i = 0; i < numStates - 1; i++) { */
    /*     mat_color.push_back((Color) { static_cast<unsigned char>(i * (205 / (numStates - 1)) + 50), 0, 0, 255 }); */
    /* } */
    // KEEP IN MIND THESE ARE COLORS IN BACKWARDS ORDER
    /* mat_color.push_back((Color) { 0, 200, 255, 255 }); */

    /* mat_color.push_back((Color) { 64, 63,  }); */ 
    /* mat_color.push_back((Color) { 87, 20, 15, 87 }); */
    /* mat_color.push_back((Color) { 255, 8, 0, 255 }); */
    /* mat_color.push_back((Color) { 255, 168, 38, 255 }); */
    /* mat_color.push_back((Color) { 255, 237, 166, 255 }); */
    /* mat_color.push_back((Color) { 102, 13, 184, 183 }); */
    /* mat_color.push_back((Color) { 189, 6, 40, 188 }); */
    /* mat_color.push_back((Color) { 255, 0, 0, 255 }); */
    /* mat_color.push_back((Color) { 255, 94, 0, 255 }); */
    /* mat_color.push_back((Color) { 255, 158, 3, 255 }); */

    mat_color.push_back((Color) { 255, 255, 218, 255 });
    mat_color.push_back((Color) { 255, 244, 140, 255 });
    mat_color.push_back((Color) { 250, 199, 47, 250 });
    mat_color.push_back((Color) { 250, 115, 1, 250 });
    mat_color.push_back((Color) { 214, 32, 32, 214 });
    /* mat_color.push_back((Color) { 132, 15, 138, 138 }); */
    /* mat_color.push_back((Color) { 66, 0, 148, 148 }); */
    mat_color.push_back((Color) { 87, 0, 1, 87 });
    mat_color.push_back((Color) { 87, 0, 1, 87 });
    mat_color.push_back((Color) { 25, 33, 48, 48 });
    /* mat_color.push_back((Color) { 9, 12, 14, 14 }); */
    mat_color.push_back((Color) { 9, 12, 14, 14 });


    Material matInstances[numStates - 1];
    for (int i = 0; i < numStates - 1; i++) {
        matInstances[i] = LoadMaterialDefault();
        matInstances[i].shader = shader;
        matInstances[i].maps[MATERIAL_MAP_DIFFUSE].color = mat_color[i];
    }

    /* Material matDefault = LoadMaterialDefault(); */
    /* matDefault.maps[MATERIAL_MAP_DIFFUSE].color = BLUE; */
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
        /* float cameraTarget[3] = { camera.target.x, camera.target.y, camera.target.z }; */
        SetShaderValue(shader, shader.locs[SHADER_LOC_VECTOR_VIEW], cameraPos, SHADER_UNIFORM_VEC3);
        
        /* float deltaTime = GetFrameTime(); */
        /* runTime += deltaTime; */

        // Set shader required uniform values
        /* SetShaderValue(shader, viewEyeLoc, cameraPos, SHADER_UNIFORM_VEC3); */
        /* SetShaderValue(shader, viewCenterLoc, cameraTarget, SHADER_UNIFORM_VEC3); */
        /* SetShaderValue(shader, runTimeLoc, &runTime, SHADER_UNIFORM_FLOAT); */

        /* light1.position = Vector3RotateByAxisAngle(camera.position, (Vector3) {sqrt(2.f) / 2.f, 0, sqrt(2.f) / 2.f}, 2 * PI/3.f); */
        /* light2.position = Vector3RotateByAxisAngle(light1.position, (Vector3) {0, -1, 0},  PI); */
        light1.position = (Vector3){sqrt(2.f) * camera.position.x / 2.f, 4 * camera.position.y,  -2 * camera.position.z / 2.f};
        /* light3.position = (Vector3){-camera.position.x, camera.position.y, -camera.position.z}; */
        /* light4.position = (Vector3){camera.position.x, camera.position.y, -camera.position.z}; */
        UpdateLightValues(shader, light1);
        /* UpdateLightValues(shader, light2); */
        /* UpdateLightValues(shader, light3); */
        /* UpdateLightValues(shader, light4); */
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
                transforms.clear();
                num_blocks.clear();
                // Get transforms and other data from the file
                info = parse_data(filename, numStates, size);
                for (int i = 1; i < numStates; i++) {
                    transforms.push_back(&(info[i])[0]);
                    /* std::cout << "Transforms: " << transforms[0].m5 << std::endl; */
                    num_blocks.push_back((info[i]).size());
                }
                updated = false;
                /* std::cout << "Number of blocks: " << num_blocks << std::endl; */
            }

            // Draw outline (bounding box)
            /* std::cout << "Origin: (" << origin.x << ", " << origin.y << ", " << origin.z << ")" << std::endl; */
            /* std::cout << "Size: " << size << std::endl; */
            /* DrawCubeWires(origin, size, size, size, DARKBLUE); */
            /* DrawMesh(cube, matDefault, MatrixTranslate(-camera.position.x, camera.position.y, camera.position.z)); */
            // Draw mesh instances
            for (int i = 0; i < numStates - 1; i++) {
                /* BeginShaderMode(shader); */
                DrawMeshInstanced(cube, matInstances[i], transforms.at(i), num_blocks.at(i));
                /* EndShaderMode(); */
            }

            first = false;

            EndMode3D();

            // Uncomment if you want to see FPS
            /* DrawFPS(10, 10); */
            /* std::string frame_num = "Frame " + std::to_string(frame + 1); */
            /* DrawText(TextFormat("Frame: %i", frame), 10, 40, 25, SKYBLUE); */


        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    UnloadShader(shader);
    CloseWindow();        // Close window and OpenGL context

    return 0;
}
