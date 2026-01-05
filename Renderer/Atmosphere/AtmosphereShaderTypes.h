/**
 * Atmospheric scattering shader types shared between Metal shaders and CPU code.
 * Based on Eric Bruneton's Precomputed Atmospheric Scattering.
 */

#ifndef AtmosphereShaderTypes_h
#define AtmosphereShaderTypes_h

#include <simd/simd.h>

// =============================================================================
// Buffer and Texture Indices
// =============================================================================

// Buffer indices
typedef enum BufferIndex {
    BufferIndexUniforms             = 0,
    BufferIndexAtmosphere           = 1,
    BufferIndexVertices             = 2,
} BufferIndex;

// Texture indices
typedef enum TextureIndex {
    TextureIndexTransmittance       = 0,
    TextureIndexScattering          = 1,
    TextureIndexIrradiance          = 2,
    TextureIndexSingleMieScattering = 3,
    TextureIndexDeltaIrradiance     = 4,
    TextureIndexDeltaRayleigh       = 5,
    TextureIndexDeltaMie            = 6,
    TextureIndexDeltaScatteringDensity = 7,
    TextureIndexDeltaMultipleScattering = 8,
} TextureIndex;

// =============================================================================
// Density Profile Layer
// =============================================================================

/**
 * An atmosphere layer of width 'width', and whose density is defined as
 *   'exp_term' * exp('exp_scale' * h) + 'linear_term' * h + 'constant_term',
 * clamped to [0,1], and where h is the altitude (in meters).
 */
typedef struct {
    float width;           // Layer width in meters
    float exp_term;        // Exponential term coefficient
    float exp_scale;       // Exponential scale (1/m)
    float linear_term;     // Linear term coefficient (1/m)
    float constant_term;   // Constant term
    float _padding[3];     // Padding to 32 bytes for Metal alignment
} DensityProfileLayer;

/**
 * An atmosphere density profile made of two layers on top of each other.
 * The width of the last layer is ignored (extends to atmosphere top).
 */
typedef struct {
    DensityProfileLayer layers[2];
} DensityProfile;

// =============================================================================
// Atmosphere Parameters
// =============================================================================

/**
 * Complete atmosphere parameters for scattering computation.
 * All lengths are in meters, angles in radians.
 */
typedef struct {
    // Solar irradiance at top of atmosphere (W/m^2/nm for each RGB wavelength)
    vector_float3 solar_irradiance;
    // Sun's angular radius in radians (should be < 0.1 rad)
    float sun_angular_radius;

    // Distance from planet center to bottom of atmosphere (m)
    float bottom_radius;
    // Distance from planet center to top of atmosphere (m)
    float top_radius;
    // Padding for alignment
    float _padding1[2];

    // Rayleigh scattering
    DensityProfile rayleigh_density;
    vector_float3 rayleigh_scattering;  // Scattering coefficient at sea level (1/m)
    float _padding2;

    // Mie scattering (aerosols)
    DensityProfile mie_density;
    vector_float3 mie_scattering;       // Scattering coefficient at sea level (1/m)
    float _padding3;
    vector_float3 mie_extinction;       // Extinction coefficient at sea level (1/m)
    float mie_phase_function_g;         // Asymmetry parameter for Cornette-Shanks phase function

    // Absorption (ozone)
    DensityProfile absorption_density;
    vector_float3 absorption_extinction; // Extinction coefficient (1/m)
    float _padding4;

    // Ground
    vector_float3 ground_albedo;        // Average ground albedo
    // Cosine of maximum sun zenith angle for precomputation
    float mu_s_min;
} AtmosphereParameters;

// =============================================================================
// Rendering Uniforms
// =============================================================================

/**
 * Per-frame uniforms for atmosphere rendering.
 */
typedef struct {
    // Transformation matrices
    matrix_float4x4 view_from_clip;     // Inverse projection matrix
    matrix_float4x4 model_from_view;    // Inverse view matrix (world from camera)

    // Camera and scene
    vector_float3 camera;               // Camera position in world space (m)
    float exposure;                     // Exposure value for tone mapping

    vector_float3 white_point;          // White point for color correction
    float _padding1;

    vector_float3 earth_center;         // Earth center position in world space (m)
    float _padding2;

    vector_float3 sun_direction;        // Direction to sun (normalized)
    float _padding3;

    vector_float2 sun_size;             // Sun angular size (cos(angle), sin(angle))
    vector_float2 _padding4;
} AtmosphereUniforms;

// =============================================================================
// Luminance Mode
// =============================================================================

typedef enum LuminanceMode {
    // Render in radiance mode (no luminance conversion)
    LUMINANCE_MODE_NONE = 0,
    // Use approximate 3-wavelength luminance
    LUMINANCE_MODE_APPROXIMATE = 1,
    // Use full precomputed luminance
    LUMINANCE_MODE_PRECOMPUTED = 2,
} LuminanceMode;

// =============================================================================
// Demo Configuration
// =============================================================================

typedef struct {
    float view_distance_meters;         // Camera distance from earth center
    float view_zenith_angle_radians;    // Camera zenith angle
    float view_azimuth_angle_radians;   // Camera azimuth angle
    float sun_zenith_angle_radians;     // Sun zenith angle
    float sun_azimuth_angle_radians;    // Sun azimuth angle
    float exposure;                     // Exposure value
    LuminanceMode luminance_mode;       // Luminance conversion mode
    int use_ozone;                      // Whether to include ozone absorption
    int use_combined_textures;          // Whether Mie is combined with Rayleigh
    int _padding[3];
} DemoConfiguration;

// =============================================================================
// Vertex Types for Full-Screen Quad
// =============================================================================

typedef struct {
    vector_float2 position;             // Clip-space position
    vector_float2 texcoord;             // Texture coordinates
} AtmosphereVertex;

#endif /* AtmosphereShaderTypes_h */
