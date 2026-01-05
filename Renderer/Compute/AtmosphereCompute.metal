/**
 * Compute shaders for precomputing atmospheric scattering textures.
 * Based on Eric Bruneton's Precomputed Atmospheric Scattering.
 */

#include <metal_stdlib>
using namespace metal;

#import "../Atmosphere/AtmosphereShaderTypes.h"
#import "../Atmosphere/AtmosphereConstants.h"

// =============================================================================
// Type Aliases
// =============================================================================

typedef float Length;
typedef float Number;
typedef float Angle;
typedef float SolidAngle;
typedef float Area;
typedef float3 DimensionlessSpectrum;
typedef float3 IrradianceSpectrum;
typedef float3 RadianceSpectrum;
typedef float3 RadianceDensitySpectrum;
typedef float3 ScatteringSpectrum;
typedef float3 Position;
typedef float3 Direction;

// =============================================================================
// Utility Functions (duplicated for compute shaders - Metal doesn't support includes well)
// =============================================================================

inline Number ClampCosine(Number mu) {
    return clamp(mu, Number(-1.0), Number(1.0));
}

inline Length ClampDistance(Length d) {
    return max(d, Length(0.0));
}

inline Length ClampRadius(constant AtmosphereParameters& atmosphere, Length r) {
    return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

inline Length SafeSqrt(Area a) {
    return sqrt(max(a, Area(0.0)));
}

inline Length DistanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    Area discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.top_radius * atmosphere.top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

inline Length DistanceToBottomAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    Area discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

inline bool RayIntersectsGround(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}

inline Number GetLayerDensity(DensityProfileLayer layer, Length altitude) {
    Number density = layer.exp_term * exp(layer.exp_scale * altitude) +
        layer.linear_term * altitude + layer.constant_term;
    return clamp(density, Number(0.0), Number(1.0));
}

inline Number GetProfileDensity(DensityProfile profile, Length altitude) {
    return altitude < profile.layers[0].width ?
        GetLayerDensity(profile.layers[0], altitude) :
        GetLayerDensity(profile.layers[1], altitude);
}

inline Number GetTextureCoordFromUnitRange(Number x, int texture_size) {
    return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
}

inline Number GetUnitRangeFromTextureCoord(Number u, int texture_size) {
    return (u - 0.5 / Number(texture_size)) / (1.0 - 1.0 / Number(texture_size));
}

inline float RayleighPhaseFunction(Number nu) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + nu * nu);
}

inline float MiePhaseFunction(Number g, Number nu) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

// =============================================================================
// Transmittance Computation
// =============================================================================

inline Length ComputeOpticalLengthToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    DensityProfile profile,
    Length r, Number mu) {
    const int SAMPLE_COUNT = 500;
    Length dx = DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);
    Length result = 0.0;
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        Length d_i = Number(i) * dx;
        Length r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        Number y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        result += y_i * weight_i * dx;
    }
    return result;
}

inline DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    return exp(-(
        atmosphere.rayleigh_scattering *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.rayleigh_density, r, mu) +
        atmosphere.mie_extinction *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.mie_density, r, mu) +
        atmosphere.absorption_extinction *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.absorption_density, r, mu)));
}

inline void GetRMuFromTransmittanceTextureUv(
    constant AtmosphereParameters& atmosphere,
    float2 uv, thread Length& r, thread Number& mu) {
    Number x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    Number x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = H * x_r;
    r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length d_min = atmosphere.top_radius - r;
    Length d_max = rho + H;
    Length d = d_min + x_mu * (d_max - d_min);
    mu = d == 0.0 ? Number(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = ClampCosine(mu);
}

// =============================================================================
// Transmittance Compute Kernel
// =============================================================================

kernel void ComputeTransmittance(
    texture2d<float, access::write> transmittance_texture [[texture(0)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= TRANSMITTANCE_TEXTURE_WIDTH || gid.y >= TRANSMITTANCE_TEXTURE_HEIGHT) {
        return;
    }

    float2 frag_coord = float2(gid) + 0.5;
    const float2 TRANSMITTANCE_TEXTURE_SIZE =
        float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);

    Length r;
    Number mu;
    GetRMuFromTransmittanceTextureUv(
        atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);

    float3 transmittance = ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
    transmittance_texture.write(float4(transmittance, 1.0), gid);
}

// =============================================================================
// Direct Irradiance Computation
// =============================================================================

inline float2 GetTransmittanceTextureUvFromRMu(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    Length d_min = atmosphere.top_radius - r;
    Length d_max = rho + H;
    Number x_mu = (d - d_min) / (d_max - d_min);
    Number x_r = rho / H;
    return float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
                  GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

inline DimensionlessSpectrum GetTransmittanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu) {
    float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
    return DimensionlessSpectrum(transmittance_texture.sample(s, uv).rgb);
}

inline void GetRMuSFromIrradianceTextureUv(
    constant AtmosphereParameters& atmosphere,
    float2 uv, thread Length& r, thread Number& mu_s) {
    Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    r = atmosphere.bottom_radius +
        x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
    mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

inline IrradianceSpectrum ComputeDirectIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu_s) {
    Number alpha_s = atmosphere.sun_angular_radius;
    Number average_cosine_factor =
        mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :
            (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

    return atmosphere.solar_irradiance *
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, s, r, mu_s) * average_cosine_factor;
}

// =============================================================================
// Direct Irradiance Compute Kernel
// =============================================================================

kernel void ComputeDirectIrradiance(
    texture2d<float, access::write> delta_irradiance_texture [[texture(0)]],
    texture2d<float, access::write> irradiance_texture [[texture(1)]],
    texture2d<float> transmittance_texture [[texture(2)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= IRRADIANCE_TEXTURE_WIDTH || gid.y >= IRRADIANCE_TEXTURE_HEIGHT) {
        return;
    }

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float2 frag_coord = float2(gid) + 0.5;
    const float2 IRRADIANCE_TEXTURE_SIZE =
        float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

    Length r;
    Number mu_s;
    GetRMuSFromIrradianceTextureUv(
        atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);

    float3 delta_irradiance = ComputeDirectIrradiance(
        atmosphere, transmittance_texture, s, r, mu_s);

    delta_irradiance_texture.write(float4(delta_irradiance, 1.0), gid);
    irradiance_texture.write(float4(0.0, 0.0, 0.0, 1.0), gid);
}

// =============================================================================
// Single Scattering Computation
// =============================================================================

inline Length DistanceToNearestAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu, bool ray_r_mu_intersects_ground) {
    if (ray_r_mu_intersects_ground) {
        return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
    } else {
        return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    }
}

inline DimensionlessSpectrum GetTransmittance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {

    Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    Number mu_d = ClampCosine((r * mu + d) / r_d);

    if (ray_r_mu_intersects_ground) {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r, -mu),
            DimensionlessSpectrum(1.0));
    } else {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r, mu) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r_d, mu_d),
            DimensionlessSpectrum(1.0));
    }
}

inline DimensionlessSpectrum GetTransmittanceToSun(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu_s) {
    Number sin_theta_h = atmosphere.bottom_radius / r;
    Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, s, r, mu_s) *
        smoothstep(-sin_theta_h * atmosphere.sun_angular_radius,
                   sin_theta_h * atmosphere.sun_angular_radius,
                   mu_s - cos_theta_h);
}

inline void ComputeSingleScatteringIntegrand(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu, Length d,
    bool ray_r_mu_intersects_ground,
    thread DimensionlessSpectrum& rayleigh, thread DimensionlessSpectrum& mie) {

    Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    Number mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
    DimensionlessSpectrum transmittance =
        GetTransmittance(atmosphere, transmittance_texture, s, r, mu, d,
            ray_r_mu_intersects_ground) *
        GetTransmittanceToSun(atmosphere, transmittance_texture, s, r_d, mu_s_d);
    rayleigh = transmittance * GetProfileDensity(
        atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
    mie = transmittance * GetProfileDensity(
        atmosphere.mie_density, r_d - atmosphere.bottom_radius);
}

inline void ComputeSingleScattering(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground,
    thread IrradianceSpectrum& rayleigh, thread IrradianceSpectrum& mie) {

    const int SAMPLE_COUNT = 50;
    Length dx = DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,
        ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);

    DimensionlessSpectrum rayleigh_sum = DimensionlessSpectrum(0.0);
    DimensionlessSpectrum mie_sum = DimensionlessSpectrum(0.0);

    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        Length d_i = Number(i) * dx;
        DimensionlessSpectrum rayleigh_i;
        DimensionlessSpectrum mie_i;
        ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture, s,
            r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_sum += rayleigh_i * weight_i;
        mie_sum += mie_i * weight_i;
    }
    rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *
        atmosphere.rayleigh_scattering;
    mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
}

inline void GetRMuMuSNuFromScatteringTextureFragCoord(
    constant AtmosphereParameters& atmosphere, float3 frag_coord,
    thread Length& r, thread Number& mu, thread Number& mu_s, thread Number& nu,
    thread bool& ray_r_mu_intersects_ground) {

    const float4 SCATTERING_TEXTURE_SIZE = float4(
        SCATTERING_TEXTURE_NU_SIZE - 1,
        SCATTERING_TEXTURE_MU_S_SIZE,
        SCATTERING_TEXTURE_MU_SIZE,
        SCATTERING_TEXTURE_R_SIZE);

    Number frag_coord_nu = floor(frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));
    Number frag_coord_mu_s = fmod(frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));
    float4 uvwz = float4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
        SCATTERING_TEXTURE_SIZE;

    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
    r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

    if (uvwz.z < 0.5) {
        Length d_min = r - atmosphere.bottom_radius;
        Length d_max = rho;
        Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
        mu = d == 0.0 ? Number(-1.0) :
            ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = true;
    } else {
        Length d_min = atmosphere.top_radius - r;
        Length d_max = rho + H;
        Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
        mu = d == 0.0 ? Number(1.0) :
            ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = false;
    }

    Number x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
    Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    Length d_max = H;
    Length D = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
    Number A = (D - d_min) / (d_max - d_min);
    Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    Length d = d_min + min(a, A) * (d_max - d_min);
    mu_s = d == 0.0 ? Number(1.0) :
        ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

    nu = ClampCosine(uvwz.x * 2.0 - 1.0);
    nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
        mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

// =============================================================================
// Single Scattering Compute Kernel
// =============================================================================

kernel void ComputeSingleScattering(
    texture3d<float, access::write> delta_rayleigh_texture [[texture(0)]],
    texture3d<float, access::write> delta_mie_texture [[texture(1)]],
    texture3d<float, access::write> scattering_texture [[texture(2)]],
    texture3d<float, access::write> single_mie_scattering_texture [[texture(3)]],
    texture2d<float> transmittance_texture [[texture(4)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    constant float3x3& luminance_from_radiance [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {

    if (gid.x >= SCATTERING_TEXTURE_WIDTH ||
        gid.y >= SCATTERING_TEXTURE_HEIGHT ||
        gid.z >= SCATTERING_TEXTURE_DEPTH) {
        return;
    }

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float3 frag_coord = float3(gid) + 0.5;

    Length r;
    Number mu;
    Number mu_s;
    Number nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    IrradianceSpectrum delta_rayleigh;
    IrradianceSpectrum delta_mie;
    ComputeSingleScattering(atmosphere, transmittance_texture, s,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground, delta_rayleigh, delta_mie);

    delta_rayleigh_texture.write(float4(delta_rayleigh, 1.0), gid);
    delta_mie_texture.write(float4(delta_mie, 1.0), gid);

    float3 scattering_rgb = luminance_from_radiance * delta_rayleigh;
    float mie_r = (luminance_from_radiance * delta_mie).r;
    scattering_texture.write(float4(scattering_rgb, mie_r), gid);
    single_mie_scattering_texture.write(float4(luminance_from_radiance * delta_mie, 1.0), gid);
}

// =============================================================================
// Scattering Density Computation
// =============================================================================

inline float4 GetScatteringTextureUvwzFromRMuMuSNu(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {

    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

    Length r_mu = r * mu;
    Area discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
    Number u_mu;

    if (ray_r_mu_intersects_ground) {
        Length d = -r_mu - SafeSqrt(discriminant);
        Length d_min = r - atmosphere.bottom_radius;
        Length d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    } else {
        Length d = -r_mu + SafeSqrt(discriminant + H * H);
        Length d_min = atmosphere.top_radius - r;
        Length d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    }

    Length d = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, mu_s);
    Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    Length d_max = H;
    Number a = (d - d_min) / (d_max - d_min);
    Length D = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
    Number A = (D - d_min) / (d_max - d_min);
    Number u_mu_s = GetTextureCoordFromUnitRange(
        max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

    Number u_nu = (nu + 1.0) / 2.0;
    return float4(u_nu, u_mu_s, u_mu, u_r);
}

inline float3 GetScattering(
    constant AtmosphereParameters& atmosphere,
    texture3d<float> scattering_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {

    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
    Number tex_x = floor(tex_coord_x);
    Number lerp = tex_coord_x - tex_x;
    float3 uvw0 = float3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    return float3(scattering_texture.sample(s, uvw0).rgb * (1.0 - lerp) +
        scattering_texture.sample(s, uvw1).rgb * lerp);
}

inline float2 GetIrradianceTextureUvFromRMuS(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu_s) {
    Number x_r = (r - atmosphere.bottom_radius) /
        (atmosphere.top_radius - atmosphere.bottom_radius);
    Number x_mu_s = mu_s * 0.5 + 0.5;
    return float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
                  GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

inline IrradianceSpectrum GetIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> irradiance_texture,
    sampler s,
    Length r, Number mu_s) {
    float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
    return IrradianceSpectrum(irradiance_texture.sample(s, uv).rgb);
}

inline RadianceSpectrum GetScatteringWithPhase(
    constant AtmosphereParameters& atmosphere,
    texture3d<float> single_rayleigh_scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    texture3d<float> multiple_scattering_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground,
    int scattering_order) {

    if (scattering_order == 1) {
        IrradianceSpectrum rayleigh = GetScattering(
            atmosphere, single_rayleigh_scattering_texture, s, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
        IrradianceSpectrum mie = GetScattering(
            atmosphere, single_mie_scattering_texture, s, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
        return rayleigh * RayleighPhaseFunction(nu) +
            mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
    } else {
        return GetScattering(
            atmosphere, multiple_scattering_texture, s, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
    }
}

inline RadianceDensitySpectrum ComputeScatteringDensity(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    texture3d<float> single_rayleigh_scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    texture3d<float> multiple_scattering_texture,
    texture2d<float> irradiance_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu, int scattering_order) {

    float3 zenith_direction = float3(0.0, 0.0, 1.0);
    float3 omega = float3(sqrt(1.0 - mu * mu), 0.0, mu);
    Number sun_dir_x = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
    Number sun_dir_y = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
    float3 omega_s = float3(sun_dir_x, sun_dir_y, mu_s);

    const int SAMPLE_COUNT = 16;
    const Angle dphi = PI / Number(SAMPLE_COUNT);
    const Angle dtheta = PI / Number(SAMPLE_COUNT);
    RadianceDensitySpectrum rayleigh_mie = RadianceDensitySpectrum(0.0);

    for (int l = 0; l < SAMPLE_COUNT; ++l) {
        Angle theta = (Number(l) + 0.5) * dtheta;
        Number cos_theta = cos(theta);
        Number sin_theta = sin(theta);
        bool ray_r_theta_intersects_ground = RayIntersectsGround(atmosphere, r, cos_theta);

        Length distance_to_ground = 0.0;
        DimensionlessSpectrum transmittance_to_ground = DimensionlessSpectrum(0.0);
        DimensionlessSpectrum ground_albedo = DimensionlessSpectrum(0.0);
        if (ray_r_theta_intersects_ground) {
            distance_to_ground = DistanceToBottomAtmosphereBoundary(atmosphere, r, cos_theta);
            transmittance_to_ground = GetTransmittance(atmosphere, transmittance_texture, s,
                r, cos_theta, distance_to_ground, true);
            ground_albedo = atmosphere.ground_albedo;
        }

        for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {
            Angle phi = (Number(m) + 0.5) * dphi;
            float3 omega_i = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            SolidAngle domega_i = dtheta * dphi * sin(theta);

            Number nu1 = dot(omega_s, omega_i);
            RadianceSpectrum incident_radiance = GetScatteringWithPhase(atmosphere,
                single_rayleigh_scattering_texture, single_mie_scattering_texture,
                multiple_scattering_texture, s, r, omega_i.z, mu_s, nu1,
                ray_r_theta_intersects_ground, scattering_order - 1);

            float3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);
            IrradianceSpectrum ground_irradiance = GetIrradiance(
                atmosphere, irradiance_texture, s, atmosphere.bottom_radius,
                dot(ground_normal, omega_s));
            incident_radiance += transmittance_to_ground *
                ground_albedo * (1.0 / PI) * ground_irradiance;

            Number nu2 = dot(omega, omega_i);
            Number rayleigh_density = GetProfileDensity(
                atmosphere.rayleigh_density, r - atmosphere.bottom_radius);
            Number mie_density = GetProfileDensity(
                atmosphere.mie_density, r - atmosphere.bottom_radius);
            rayleigh_mie += incident_radiance * (
                atmosphere.rayleigh_scattering * rayleigh_density *
                    RayleighPhaseFunction(nu2) +
                atmosphere.mie_scattering * mie_density *
                    MiePhaseFunction(atmosphere.mie_phase_function_g, nu2)) *
                domega_i;
        }
    }
    return rayleigh_mie;
}

// =============================================================================
// Scattering Density Compute Kernel
// =============================================================================

kernel void ComputeScatteringDensity(
    texture3d<float, access::write> scattering_density_texture [[texture(0)]],
    texture2d<float> transmittance_texture [[texture(1)]],
    texture3d<float> single_rayleigh_scattering_texture [[texture(2)]],
    texture3d<float> single_mie_scattering_texture [[texture(3)]],
    texture3d<float> multiple_scattering_texture [[texture(4)]],
    texture2d<float> irradiance_texture [[texture(5)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    constant int& scattering_order [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {

    if (gid.x >= SCATTERING_TEXTURE_WIDTH ||
        gid.y >= SCATTERING_TEXTURE_HEIGHT ||
        gid.z >= SCATTERING_TEXTURE_DEPTH) {
        return;
    }

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float3 frag_coord = float3(gid) + 0.5;

    Length r;
    Number mu;
    Number mu_s;
    Number nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    float3 density = ComputeScatteringDensity(atmosphere,
        transmittance_texture, single_rayleigh_scattering_texture,
        single_mie_scattering_texture, multiple_scattering_texture,
        irradiance_texture, s, r, mu, mu_s, nu, scattering_order);

    scattering_density_texture.write(float4(density, 1.0), gid);
}

// =============================================================================
// Indirect Irradiance Computation
// =============================================================================

inline IrradianceSpectrum ComputeIndirectIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture3d<float> single_rayleigh_scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    texture3d<float> multiple_scattering_texture,
    sampler s,
    Length r, Number mu_s, int scattering_order) {

    const int SAMPLE_COUNT = 32;
    const Angle dphi = PI / Number(SAMPLE_COUNT);
    const Angle dtheta = PI / Number(SAMPLE_COUNT);

    IrradianceSpectrum result = IrradianceSpectrum(0.0);
    float3 omega_s = float3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);

    for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
        Angle theta = (Number(j) + 0.5) * dtheta;
        for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
            Angle phi = (Number(i) + 0.5) * dphi;
            float3 omega = float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
            SolidAngle domega = dtheta * dphi * sin(theta);

            Number nu = dot(omega, omega_s);
            result += GetScatteringWithPhase(atmosphere,
                single_rayleigh_scattering_texture, single_mie_scattering_texture,
                multiple_scattering_texture, s, r, omega.z, mu_s, nu, false,
                scattering_order) * omega.z * domega;
        }
    }
    return result;
}

// =============================================================================
// Indirect Irradiance Compute Kernel
// =============================================================================

kernel void ComputeIndirectIrradiance(
    texture2d<float, access::write> delta_irradiance_texture [[texture(0)]],
    texture2d<float, access::read_write> irradiance_texture [[texture(1)]],
    texture3d<float> single_rayleigh_scattering_texture [[texture(2)]],
    texture3d<float> single_mie_scattering_texture [[texture(3)]],
    texture3d<float> multiple_scattering_texture [[texture(4)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    constant float3x3& luminance_from_radiance [[buffer(1)]],
    constant int& scattering_order [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {

    if (gid.x >= IRRADIANCE_TEXTURE_WIDTH || gid.y >= IRRADIANCE_TEXTURE_HEIGHT) {
        return;
    }

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float2 frag_coord = float2(gid) + 0.5;
    const float2 IRRADIANCE_TEXTURE_SIZE =
        float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

    Length r;
    Number mu_s;
    GetRMuSFromIrradianceTextureUv(
        atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);

    float3 delta_irradiance = ComputeIndirectIrradiance(atmosphere,
        single_rayleigh_scattering_texture, single_mie_scattering_texture,
        multiple_scattering_texture, s, r, mu_s, scattering_order);

    delta_irradiance_texture.write(float4(delta_irradiance, 1.0), gid);

    float4 existing = irradiance_texture.read(gid);
    float3 accumulated = existing.rgb + luminance_from_radiance * delta_irradiance;
    irradiance_texture.write(float4(accumulated, 1.0), gid);
}

// =============================================================================
// Multiple Scattering Computation
// =============================================================================

inline RadianceSpectrum ComputeMultipleScattering(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    texture3d<float> scattering_density_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {

    const int SAMPLE_COUNT = 50;
    Length dx = DistanceToNearestAtmosphereBoundary(
        atmosphere, r, mu, ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);

    RadianceSpectrum rayleigh_mie_sum = RadianceSpectrum(0.0);
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        Length d_i = Number(i) * dx;
        Length r_i = ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
        Number mu_i = ClampCosine((r * mu + d_i) / r_i);
        Number mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

        RadianceSpectrum rayleigh_mie_i =
            GetScattering(atmosphere, scattering_density_texture, s,
                r_i, mu_i, mu_s_i, nu, ray_r_mu_intersects_ground) *
            GetTransmittance(atmosphere, transmittance_texture, s,
                r, mu, d_i, ray_r_mu_intersects_ground) * dx;
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_mie_sum += rayleigh_mie_i * weight_i;
    }
    return rayleigh_mie_sum;
}

// =============================================================================
// Multiple Scattering Compute Kernel
// =============================================================================

kernel void ComputeMultipleScattering(
    texture3d<float, access::write> delta_multiple_scattering_texture [[texture(0)]],
    texture3d<float, access::read_write> scattering_texture [[texture(1)]],
    texture2d<float> transmittance_texture [[texture(2)]],
    texture3d<float> scattering_density_texture [[texture(3)]],
    constant AtmosphereParameters& atmosphere [[buffer(0)]],
    constant float3x3& luminance_from_radiance [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]) {

    if (gid.x >= SCATTERING_TEXTURE_WIDTH ||
        gid.y >= SCATTERING_TEXTURE_HEIGHT ||
        gid.z >= SCATTERING_TEXTURE_DEPTH) {
        return;
    }

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float3 frag_coord = float3(gid) + 0.5;

    Length r;
    Number mu;
    Number mu_s;
    Number nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    float3 delta_multiple_scattering = ComputeMultipleScattering(atmosphere,
        transmittance_texture, scattering_density_texture, s,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    delta_multiple_scattering_texture.write(float4(delta_multiple_scattering, 1.0), gid);

    float4 existing = scattering_texture.read(gid);
    float3 accumulated = existing.rgb + luminance_from_radiance *
        delta_multiple_scattering / RayleighPhaseFunction(nu);
    scattering_texture.write(float4(accumulated, existing.a), gid);
}
