/**
 * Bruneton & Neyret (2008) "Precomputed Atmospheric Scattering" 논문의 핵심 알고리즘을 구현합니다.
 *
 * 논문의 주요 아이디어:
 * 1. 투과율(Transmittance), 단일 산란(Single Scattering), 복수 산란(Multiple Scattering)을
 *    4차원 함수로 표현
 * 2. 이 함수들을 사전 계산하여 텍스처(LUT)에 저장
 * 3. 렌더링 시 텍스처 룩업으로 빠르게 계산
 *
 * 핵심 파라미터화 (Section 4):
 * - r: 관측자의 지구 중심으로부터의 거리 (고도 + 지구 반지름)
 * - μ (mu): 시선 방향과 천정 방향의 코사인값, cos(zenith angle)
 * - μs (mu_s): 태양 방향과 천정 방향의 코사인값
 * - ν (nu): 시선 방향과 태양 방향의 코사인값 (위상 함수에 사용)
 *
 * 텍스처 구조:
 * - Transmittance: 2D (r, μ) → T(r, μ)
 * - Scattering: 4D → 3D로 매핑 (r, μ, μs, ν) → S(r, μ, μs, ν)
 * - Irradiance: 2D (r, μs) → E(r, μs)
 */

#include <metal_stdlib>
using namespace metal;

#include "AtmosphereShaderTypes.h"
#include "AtmosphereConstants.h"

// =============================================================================
// 물리량 타입 별칭 (Type Aliases for Physical Quantities)
// =============================================================================
// 논문에서 사용하는 물리량들의 차원을 코드에서 명시적으로 표현합니다.
// 실제로는 모두 float/float3이지만, 코드 가독성과 물리적 의미 전달을 위해 사용합니다.
//
// 길이/거리 관련:
//   Length - 거리 (미터 단위)
//   Area - 면적 (미터² 단위)
//
// 각도 관련:
//   Number - 무차원수 (코사인값 등)
//   Angle - 라디안 각도
//   SolidAngle - 입체각 (스테라디안)
//
// 스펙트럼/광학 관련:
//   DimensionlessSpectrum - 무차원 스펙트럼 (투과율 등)
//   IrradianceSpectrum - 복사 조도 스펙트럼 (W/m²/nm)
//   RadianceSpectrum - 복사 휘도 스펙트럼 (W/m²/sr/nm)
//   RadianceDensitySpectrum - 복사 밀도 스펙트럼
//   ScatteringSpectrum - 산란 계수 스펙트럼 (1/m)
//
// 기하학적:
//   Position - 3D 위치 벡터 (미터)
//   Direction - 정규화된 방향 벡터

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
// 유틸리티 함수 (Utility Functions)
// =============================================================================
// 수치 안정성과 유효 범위 보장을 위한 헬퍼 함수들입니다.
// 대기 산란 계산에서 발생할 수 있는 수치적 문제를 방지합니다.

// 코사인 값을 [-1, 1] 범위로 제한
// μ = cos(θ)이므로 물리적으로 이 범위를 벗어날 수 없음
inline Number ClampCosine(Number mu) {
    return clamp(mu, Number(-1.0), Number(1.0));
}

// 거리를 음수가 되지 않도록 제한
// 광선-구체 교차점 계산에서 음수 거리는 "뒤쪽" 교차점을 의미
inline Length ClampDistance(Length d) {
    return max(d, Length(0.0));
}

// 반지름을 대기 경계 내로 제한
// r ∈ [bottom_radius, top_radius] = [6,360km, 6,420km] (지구 기준)
inline Length ClampRadius(constant AtmosphereParameters& atmosphere, Length r) {
    return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

// 음수 입력에 대해 0을 반환하는 안전한 제곱근
// 수치 오차로 인한 약간의 음수값 처리
inline Length SafeSqrt(Area a) {
    return sqrt(max(a, Area(0.0)));
}

// =============================================================================
// 투과율 계산 (Transmittance Computation)
// =============================================================================
// 논문 Section 2.2, Equation 1에 해당
//
// 투과율(Transmittance) T는 빛이 대기를 통과하면서 감쇠되는 비율입니다.
// Beer-Lambert 법칙에 따라:
//
//   T(A→B) = exp(-∫[A→B] σ(x) dx)
//
// 여기서 σ(x)는 x 지점에서의 소멸 계수(extinction coefficient)입니다.
// 소멸 = 산란(scattering) + 흡수(absorption)
//
// 대기는 구형(spherical)이므로, 두 파라미터로 광선을 정의할 수 있습니다:
// - r: 시작점의 지구 중심으로부터의 거리
// - μ = cos(θ): 시선 방향과 천정 방향 사이의 각도의 코사인
//
// 이를 통해 T(r, μ)를 2D 텍스처로 사전 계산할 수 있습니다.

/**
 * 광선-구체 교차: 대기 상단 경계까지의 거리 계산
 *
 * 논문 Section 4의 파라미터화에 필수적인 기하학적 계산입니다.
 *
 * 반지름 r인 위치에서 천정 각도 θ (μ = cos θ) 방향으로 광선을 쏘았을 때,
 * 대기 상단 경계(top_radius)와 교차하는 거리 d를 계산합니다.
 *
 * 수학적 유도:
 * 광선: P(t) = origin + t * direction
 * 구체: |P|² = R²
 *
 * origin = (0, 0, r), direction = (sin θ, 0, cos θ) 형태로 설정하면
 * |origin + t * direction|² = R² 에서 이차방정식 유도:
 *   t² + 2rμt + (r² - R²) = 0
 *   t = -rμ ± √(r²μ² - r² + R²) = -rμ ± √(r²(μ² - 1) + R²)
 *
 * 대기 상단까지는 "멀리 있는" 교차점이므로 + 부호 사용
 */
inline Length DistanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    // 판별식: r²(μ² - 1) + R_top²
    // μ² - 1 = -sin²θ ≤ 0 이므로, R_top > r이면 항상 양수
    Area discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.top_radius * atmosphere.top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

/**
 * 광선-구체 교차: 지표면(대기 하단 경계)까지의 거리 계산
 *
 * 대기 상단과 달리 "가까운" 교차점이므로 - 부호 사용
 * 이 값이 유효하려면 광선이 실제로 지표면과 교차해야 함 (아래로 향하는 광선)
 */
inline Length DistanceToBottomAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    Area discriminant = r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

/**
 * 광선이 지표면과 교차하는지 검사
 *
 * 두 조건이 모두 필요:
 * 1. μ < 0: 광선이 아래쪽을 향함 (지평선 아래)
 * 2. 판별식 ≥ 0: 수학적으로 교차점이 존재
 *
 * 이 검사는 투과율 텍스처의 μ 파라미터화에서 중요합니다.
 * 지표면과 교차하는 광선은 별도로 처리해야 합니다.
 */
inline bool RayIntersectsGround(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) +
        atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}

/**
 * 단일 밀도 프로파일 레이어에서 고도별 밀도 계산
 *
 * 논문 Section 2.1의 밀도 프로파일 모델:
 *   ρ(h) = exp_term * exp(exp_scale * h) + linear_term * h + constant_term
 *
 * Rayleigh/Mie 산란의 경우 지수 감쇠 모델:
 *   ρ(h) = exp(-h / H)  →  exp_term=1, exp_scale=-1/H, linear_term=0, constant_term=0
 *
 * 오존(흡수)의 경우 더 복잡한 프로파일 사용 (성층권에서 최대)
 */
inline Number GetLayerDensity(DensityProfileLayer layer, Length altitude) {
    Number density = layer.exp_term * exp(layer.exp_scale * altitude) +
        layer.linear_term * altitude + layer.constant_term;
    return clamp(density, Number(0.0), Number(1.0));
}

/**
 * 2개 레이어로 구성된 밀도 프로파일에서 밀도 조회
 *
 * 오존 흡수를 모델링하기 위해 2개 레이어 사용:
 * - 하층 (0 ~ width): 대류권
 * - 상층 (width ~ top): 성층권 (오존층 포함)
 *
 * Rayleigh/Mie는 단일 레이어만 사용 (layers[1]만 유효)
 */
inline Number GetProfileDensity(DensityProfile profile, Length altitude) {
    return altitude < profile.layers[0].width ?
        GetLayerDensity(profile.layers[0], altitude) :
        GetLayerDensity(profile.layers[1], altitude);
}

/**
 * 광학 깊이(Optical Depth) 적분 계산
 *
 * 논문 Equation 1의 적분 부분:
 *   τ(r, μ) = ∫[0→d] ρ(h(t)) dt
 *
 * 여기서 h(t)는 광선 위의 점 t에서의 고도입니다.
 * h(t) = √(t² + 2rμt + r²) - bottom_radius
 *
 * 수치 적분: 사다리꼴 공식 (Trapezoidal rule)
 * 500개 샘플로 충분한 정확도 확보
 * (사전 계산이므로 성능보다 정확도 우선)
 */
inline Length ComputeOpticalLengthToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    DensityProfile profile,
    Length r, Number mu) {
    // 적분 구간을 500개로 분할 (사전 계산이므로 정확도 우선)
    const int SAMPLE_COUNT = 500;
    Length dx = DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / Number(SAMPLE_COUNT);

    Length result = 0.0;
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        Length d_i = Number(i) * dx;
        // 광선 위 거리 d_i에서의 지구 중심으로부터의 거리 r_i
        // r_i = |origin + d_i * direction| = √(d² + 2rμd + r²)
        Length r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        // 해당 지점의 밀도 (고도 = r_i - bottom_radius)
        Number y_i = GetProfileDensity(profile, r_i - atmosphere.bottom_radius);
        // 사다리꼴 공식: 양 끝점은 0.5 가중치
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        result += y_i * weight_i * dx;
    }
    return result;
}

/**
 * 투과율(Transmittance) 계산 - 논문 Equation 1
 *
 *   T(r, μ) = exp(-σ_R * τ_R - σ_M * τ_M - σ_A * τ_A)
 *
 * 세 가지 성분의 광학 깊이를 합산:
 * - Rayleigh 산란 (σ_R): 공기 분자에 의한 산란
 * - Mie 소멸 (σ_M): 에어로졸에 의한 산란 + 흡수
 * - 흡수 (σ_A): 오존 등에 의한 순수 흡수
 *
 * 각 성분은 파장에 따라 다른 계수를 가지므로 RGB 3채널로 계산
 */
inline DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    return exp(-(
        // Rayleigh 산란: β_s^R * τ_R
        atmosphere.rayleigh_scattering *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.rayleigh_density, r, mu) +
        // Mie 소멸 (산란 + 흡수): β_e^M * τ_M
        atmosphere.mie_extinction *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.mie_density, r, mu) +
        // 오존 등 흡수: β_a * τ_A
        atmosphere.absorption_extinction *
            ComputeOpticalLengthToTopAtmosphereBoundary(
                atmosphere, atmosphere.absorption_density, r, mu)));
}

// =============================================================================
// 투과율 텍스처 매핑 (Transmittance Texture Mapping)
// =============================================================================
// 논문 Section 4의 투과율 텍스처 파라미터화
//
// 투과율 T(r, μ)를 2D 텍스처에 저장하기 위한 좌표 변환입니다.
// 핵심 아이디어: 직접 r, μ를 사용하면 정밀도 손실이 발생합니다.
// 특히 수평선 근처(μ ≈ 0)에서 투과율이 급격히 변하므로
// 거리 기반 파라미터화로 더 균일한 샘플링을 확보합니다.
//
// 파라미터화 변수:
// - x_r = ρ / H
//   여기서 ρ = √(r² - R_bottom²) : 고도에 따른 "수평 거리"
//        H = √(R_top² - R_bottom²) : 최대 수평 거리
//
// - x_μ = (d - d_min) / (d_max - d_min)
//   여기서 d = 대기 상단까지의 거리
//        d_min, d_max = 해당 r에서 가능한 거리의 범위

/**
 * [0, 1] 범위를 텍스처 좌표로 변환 (반 픽셀 오프셋 적용)
 *
 * 텍스처 샘플링 시 경계에서의 클램핑 문제를 방지합니다.
 * x ∈ [0, 1] → u ∈ [0.5/size, 1 - 0.5/size]
 *
 * 이렇게 하면 텍스처 가장자리 픽셀의 중심이 정확히 0과 1에 대응됩니다.
 */
inline Number GetTextureCoordFromUnitRange(Number x, int texture_size) {
    return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
}

/**
 * 텍스처 좌표를 [0, 1] 범위로 역변환
 */
inline Number GetUnitRangeFromTextureCoord(Number u, int texture_size) {
    return (u - 0.5 / Number(texture_size)) / (1.0 - 1.0 / Number(texture_size));
}

/**
 * (r, μ) → 투과율 텍스처 UV 좌표 변환
 *
 * 논문 Section 4의 투과율 파라미터화:
 *
 * r 파라미터화 (y축):
 *   ρ = √(r² - R_bottom²)  : 지표면에서의 "접선 거리"
 *   H = √(R_top² - R_bottom²) ≈ 79.7km (지구 기준)
 *   x_r = ρ / H ∈ [0, 1]
 *
 * μ 파라미터화 (x축):
 *   거리 기반으로 변환하여 수평선 근처 정밀도 향상
 *   d = 대기 상단까지 거리
 *   d_min = R_top - r  (수직 상방, μ=1)
 *   d_max = ρ + H      (접선 방향, μ=0)
 *   x_μ = (d - d_min) / (d_max - d_min) ∈ [0, 1]
 *
 * 왜 이렇게 하나?
 * - μ를 직접 사용하면 수평선(μ≈0) 근처에서 샘플이 부족
 * - 거리 기반 파라미터화로 시각적으로 중요한 영역에 더 많은 샘플 할당
 */
inline float2 GetTransmittanceTextureUvFromRMu(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu) {
    // H: 대기 두께에 해당하는 "최대 접선 거리"
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    // ρ: 현재 고도에서의 접선 거리
    Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    // d: (r, μ)에서 대기 상단까지의 실제 거리
    Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    // d의 최소/최대 범위 (해당 r에서)
    Length d_min = atmosphere.top_radius - r;  // μ=1 (수직 상방)
    Length d_max = rho + H;                     // μ 최소 (접선 방향)
    // 정규화
    Number x_mu = (d - d_min) / (d_max - d_min);
    Number x_r = rho / H;
    return float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
                  GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

/**
 * 투과율 텍스처 UV → (r, μ) 역변환
 *
 * 사전 계산 단계에서 사용: 각 텍셀이 어떤 (r, μ)에 대응하는지 계산
 *
 * μ 역계산:
 *   d가 주어지면, 광선-구체 교차 공식을 역으로 풀어 μ를 구함
 *   d² = r² - 2rμd + R²  (광선 방정식에서)
 *   → μ = (H² - ρ² - d²) / (2rd)
 */
inline void GetRMuFromTransmittanceTextureUv(
    constant AtmosphereParameters& atmosphere,
    float2 uv, thread Length& r, thread Number& mu) {
    // 텍스처 좌표 → [0, 1] 범위
    Number x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    Number x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    // r 복원
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = H * x_r;
    r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);
    // μ 복원
    Length d_min = atmosphere.top_radius - r;
    Length d_max = rho + H;
    Length d = d_min + x_mu * (d_max - d_min);
    // 기하학적 관계에서 μ 계산
    mu = d == 0.0 ? Number(1.0) : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = ClampCosine(mu);
}

/**
 * 투과율 텍스처 사전 계산 - 컴퓨트 셰이더용
 *
 * 각 텍셀의 fragment 좌표를 받아 해당하는 T(r, μ)를 계산합니다.
 * AtmospherePrecomputation.m에서 호출됩니다.
 */
inline DimensionlessSpectrum ComputeTransmittanceToTopAtmosphereBoundaryTexture(
    constant AtmosphereParameters& atmosphere, float2 frag_coord) {
    const float2 TRANSMITTANCE_TEXTURE_SIZE =
        float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
    Length r;
    Number mu;
    // 픽셀 좌표 → (r, μ)
    GetRMuFromTransmittanceTextureUv(
        atmosphere, frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
    // 해당 (r, μ)에서의 투과율 계산
    return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

// =============================================================================
// 투과율 텍스처 룩업 함수 (Transmittance Lookup Functions)
// =============================================================================
// 런타임에 사전 계산된 투과율 텍스처를 조회하는 함수들입니다.
// 직접 적분 계산 대신 텍스처 샘플링으로 빠르게 투과율을 얻습니다.

/**
 * 특정 지점에서 대기 상단까지의 투과율 조회
 *
 * 가장 기본적인 텍스처 룩업입니다.
 * (r, μ)를 텍스처 좌표로 변환 후 샘플링합니다.
 */
inline DimensionlessSpectrum GetTransmittanceToTopAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu) {
    float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
    return DimensionlessSpectrum(transmittance_texture.sample(s, uv).rgb);
}

/**
 * 두 지점 사이의 투과율 계산 (거리 d만큼 떨어진 점까지)
 *
 * 논문에서 설명한 핵심 트릭:
 * 임의의 두 점 A→B 사이의 투과율을 직접 저장하지 않습니다.
 * 대신 "대기 상단까지의 투과율"만 저장하고, 나눗셈으로 계산합니다.
 *
 *   T(A→B) = T(A→top) / T(B→top)
 *
 * 이렇게 하면 4D 함수가 아닌 2D 함수만 저장하면 됩니다.
 *
 * 주의: 광선이 지표면과 교차하는 경우, 방향을 반전시켜야 합니다.
 * 지표면 아래로 향하는 광선은 대기 상단에 도달하지 않으므로,
 * 반대 방향(-μ)의 투과율을 사용합니다.
 *
 * @param r 시작점의 반지름
 * @param mu 시선 방향 코사인
 * @param d 목표 지점까지의 거리
 * @param ray_r_mu_intersects_ground 광선이 지표면과 교차하는지 여부
 */
inline DimensionlessSpectrum GetTransmittance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {

    // 거리 d만큼 이동한 후의 위치 계산
    // r_d = √(d² + 2rμd + r²) : 새 위치의 반지름
    Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    // μ_d = (rμ + d) / r_d : 새 위치에서의 시선 방향 코사인
    Number mu_d = ClampCosine((r * mu + d) / r_d);

    if (ray_r_mu_intersects_ground) {
        // 지표면과 교차하는 경우: 반대 방향 사용
        // T(A→B) = T(B→top, -μ) / T(A→top, -μ)
        return min(
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r, -mu),
            DimensionlessSpectrum(1.0));
    } else {
        // 일반적인 경우
        // T(A→B) = T(A→top) / T(B→top)
        return min(
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r, mu) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, s, r_d, mu_d),
            DimensionlessSpectrum(1.0));
    }
}

/**
 * 태양까지의 투과율 (태양 원반 효과 포함)
 *
 * 태양이 수평선 근처에 있을 때, 태양 원반의 일부만 보이는 경우를 처리합니다.
 *
 * 수평선 각도 θ_h:
 *   sin(θ_h) = R_bottom / r  (지표면 접선)
 *   cos(θ_h) = -√(1 - sin²θ_h)  (음수: 수평선은 아래쪽)
 *
 * smoothstep으로 태양이 수평선을 통과할 때 부드럽게 전환합니다.
 * 태양 각반지름(sun_angular_radius)만큼의 전환 영역을 사용합니다.
 */
inline DimensionlessSpectrum GetTransmittanceToSun(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu_s) {
    // 수평선 각도 계산
    Number sin_theta_h = atmosphere.bottom_radius / r;
    Number cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    // 기본 투과율 × 수평선 페이드
    return GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, s, r, mu_s) *
        smoothstep(-sin_theta_h * atmosphere.sun_angular_radius,
                   sin_theta_h * atmosphere.sun_angular_radius,
                   mu_s - cos_theta_h);
}

// =============================================================================
// 단일 산란 계산 (Single Scattering Computation)
// =============================================================================
// 논문 Section 3, Equation 6-8에 해당
//
// 단일 산란(Single Scattering)은 태양에서 나온 빛이 대기를 통과하면서
// **정확히 한 번** 산란되어 관측자에게 도달하는 것을 의미합니다.
//
// 수학적 표현 (논문 Equation 6):
//   S(x, v, s) = ∫[x→대기경계] T(x→y) · σ_s(y) · T(y→sun) · P(v·s) dy
//
// 여기서:
// - T(x→y): x에서 y까지의 투과율
// - σ_s(y): y에서의 산란 계수
// - T(y→sun): y에서 태양까지의 투과율
// - P(v·s): 위상 함수 (산란 방향 분포)
//
// Rayleigh와 Mie 산란을 별도로 계산 후 합산합니다.
// 각각 다른 위상 함수를 사용하기 때문입니다.

/**
 * Rayleigh 위상 함수 - 논문 Equation 2
 *
 *   P_R(θ) = 3/(16π) · (1 + cos²θ)
 *
 * 공기 분자(크기 << 파장)에 의한 산란 각도 분포입니다.
 * 전방(θ=0)과 후방(θ=π)으로 동일하게 산란하는 특성이 있습니다.
 *
 * @param nu cos(θ) = v · s (시선 방향과 태양 방향의 내적)
 */
inline float RayleighPhaseFunction(Number nu) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + nu * nu);
}

/**
 * Cornette-Shanks Mie 위상 함수 - 논문 Equation 4
 *
 *   P_M(θ) = 3(1-g²)/(8π(2+g²)) · (1+cos²θ)/(1+g²-2g·cosθ)^(3/2)
 *
 * 에어로졸(입자 크기 ~ 파장)에 의한 산란입니다.
 * g > 0일 때 전방 산란이 우세합니다 (지구 대기: g ≈ 0.76-0.8)
 *
 * Henyey-Greenstein 함수의 개선 버전으로,
 * 후방 산란도 더 정확하게 모델링합니다.
 *
 * @param g 비대칭 계수 (asymmetry parameter), [-1, 1]
 * @param nu cos(θ)
 */
inline float MiePhaseFunction(Number g, Number nu) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

/**
 * 가장 가까운 대기 경계까지의 거리
 *
 * 광선이 지표면과 교차하면 지표면까지,
 * 그렇지 않으면 대기 상단까지의 거리를 반환합니다.
 * 산란 적분의 구간 결정에 사용됩니다.
 */
inline Length DistanceToNearestAtmosphereBoundary(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu, bool ray_r_mu_intersects_ground) {
    if (ray_r_mu_intersects_ground) {
        return DistanceToBottomAtmosphereBoundary(atmosphere, r, mu);
    } else {
        return DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    }
}

/**
 * 단일 산란 적분의 피적분함수 계산
 *
 * 광선 위의 특정 점 (거리 d)에서의 산란 기여도를 계산합니다.
 *
 * 구조:
 *   기여도 = T(camera→point) × T(point→sun) × ρ(point)
 *
 * 여기서:
 * - T(camera→point): 카메라에서 산란 지점까지의 투과율
 * - T(point→sun): 산란 지점에서 태양까지의 투과율
 * - ρ(point): 산란 지점의 밀도
 *
 * μ_s_d 계산 설명:
 *   새 위치에서의 태양 방향 코사인을 구합니다.
 *   μ_s_d = (r·μ_s + d·ν) / r_d
 *   여기서 ν = v·s (시선과 태양 방향의 내적)
 */
inline void ComputeSingleScatteringIntegrand(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu, Length d,
    bool ray_r_mu_intersects_ground,
    thread DimensionlessSpectrum& rayleigh, thread DimensionlessSpectrum& mie) {

    // 거리 d에서의 위치 계산
    Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    // 새 위치에서 태양 방향의 천정각 코사인
    Number mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);

    // 총 투과율 = T(camera→point) × T(point→sun)
    DimensionlessSpectrum transmittance =
        GetTransmittance(atmosphere, transmittance_texture, s, r, mu, d,
            ray_r_mu_intersects_ground) *
        GetTransmittanceToSun(atmosphere, transmittance_texture, s, r_d, mu_s_d);

    // Rayleigh/Mie 밀도와 투과율의 곱
    rayleigh = transmittance * GetProfileDensity(
        atmosphere.rayleigh_density, r_d - atmosphere.bottom_radius);
    mie = transmittance * GetProfileDensity(
        atmosphere.mie_density, r_d - atmosphere.bottom_radius);
}

/**
 * 단일 산란 적분 계산 - 논문 Equation 6
 *
 *   S_R(r,μ,μs,ν) = ∫ T(r→r_i) · T(r_i→sun) · ρ_R(h_i) · dx
 *   S_M(r,μ,μs,ν) = ∫ T(r→r_i) · T(r_i→sun) · ρ_M(h_i) · dx
 *
 * 최종 결과:
 *   S_R = 적분결과 × E_sun × β_R  (태양 복사량 × Rayleigh 산란계수)
 *   S_M = 적분결과 × E_sun × β_M  (태양 복사량 × Mie 산란계수)
 *
 * 위상 함수는 아직 적용하지 않음 (렌더링 시 적용)
 * 이렇게 하면 텍스처에 저장된 값을 다양한 시선 각도에 재사용 가능
 *
 * 수치 적분: 사다리꼴 공식, 50개 샘플
 */
inline void ComputeSingleScattering(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground,
    thread IrradianceSpectrum& rayleigh, thread IrradianceSpectrum& mie) {

    // 적분 구간 분할 (50개 샘플은 사전 계산용으로 적당)
    const int SAMPLE_COUNT = 50;
    Length dx = DistanceToNearestAtmosphereBoundary(atmosphere, r, mu,
        ray_r_mu_intersects_ground) / Number(SAMPLE_COUNT);

    DimensionlessSpectrum rayleigh_sum = DimensionlessSpectrum(0.0);
    DimensionlessSpectrum mie_sum = DimensionlessSpectrum(0.0);

    // 사다리꼴 적분
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        Length d_i = Number(i) * dx;
        DimensionlessSpectrum rayleigh_i;
        DimensionlessSpectrum mie_i;
        ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture, s,
            r, mu, mu_s, nu, d_i, ray_r_mu_intersects_ground, rayleigh_i, mie_i);
        // 양 끝점은 가중치 0.5
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_sum += rayleigh_i * weight_i;
        mie_sum += mie_i * weight_i;
    }

    // 최종 결과: 적분값 × 간격 × 태양 복사량 × 산란 계수
    rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance *
        atmosphere.rayleigh_scattering;
    mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
}

// =============================================================================
// 산란 텍스처 매핑 (Scattering Texture Mapping)
// =============================================================================
// 논문 Section 4의 4D 산란 텍스처 파라미터화
//
// 산란 함수 S(r, μ, μs, ν)는 4차원입니다.
// 4D 텍스처를 직접 사용할 수 없으므로, 3D 텍스처에 4D 데이터를 저장합니다.
//
// 파라미터:
// - r: 관측자의 반지름 (고도)
// - μ: 시선 방향 천정각 코사인
// - μs: 태양 방향 천정각 코사인
// - ν: 시선-태양 방향 코사인 (위상 함수에 필요)
//
// 3D 매핑 전략:
// - z축: r (고도)
// - y축: μ (시선 방향)
// - x축: μs와 ν를 결합 (ν × SCATTERING_TEXTURE_MU_S_SIZE + μs)
//
// 핵심 최적화:
// - μ 축을 절반으로 나눔: [0, 0.5) = 지표면 교차, [0.5, 1] = 하늘
// - μs 축에 비선형 매핑으로 수평선 근처 정밀도 향상
// - ν는 선형 매핑 (큰 변화가 없음)

/**
 * (r, μ, μs, ν) → 산란 텍스처 UVWZ 좌표 변환
 *
 * 논문 Section 4의 파라미터화 구현입니다.
 *
 * 주요 특징:
 *
 * 1. r 파라미터화 (w 좌표):
 *    투과율 텍스처와 동일한 방식
 *    u_r = ρ / H where ρ = √(r² - R_bottom²)
 *
 * 2. μ 파라미터화 (z 좌표) - 특별 처리:
 *    텍스처를 절반으로 분할:
 *    - 하위 절반 [0, 0.5): 지표면과 교차하는 광선 (μ < μ_horizon)
 *    - 상위 절반 [0.5, 1]: 하늘로 향하는 광선 (μ ≥ μ_horizon)
 *
 *    각 절반 내에서 거리 기반 파라미터화:
 *    - 지표면 교차: d = 지표면까지 거리
 *    - 하늘: d = 대기 상단까지 거리
 *
 * 3. μs 파라미터화 (y 좌표) - 비선형:
 *    태양 고도가 낮을수록 더 많은 샘플 할당
 *    수평선 근처가 시각적으로 가장 중요하기 때문
 *
 *    a = (d - d_min) / (d_max - d_min)  (거리 기반)
 *    u_μs = (1 - a/A) / (1 + a)  (비선형 변환)
 *
 * 4. ν 파라미터화 (x 좌표):
 *    단순 선형: u_ν = (ν + 1) / 2
 */
inline float4 GetScatteringTextureUvwzFromRMuMuSNu(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {

    // === r 파라미터화 (w축) ===
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

    // === μ 파라미터화 (z축) - 지표면 교차 여부에 따라 다른 처리 ===
    Length r_mu = r * mu;
    // 지표면 교차 판별식
    Area discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
    Number u_mu;

    if (ray_r_mu_intersects_ground) {
        // 지표면과 교차: 텍스처 하위 절반 [0, 0.5) 사용
        Length d = -r_mu - SafeSqrt(discriminant);  // 지표면까지 거리
        Length d_min = r - atmosphere.bottom_radius;  // 최소 (수직 하강)
        Length d_max = rho;                           // 최대 (수평선 방향)
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    } else {
        // 하늘로 향함: 텍스처 상위 절반 [0.5, 1] 사용
        Length d = -r_mu + SafeSqrt(discriminant + H * H);  // 대기 상단까지 거리
        Length d_min = atmosphere.top_radius - r;            // 최소 (수직 상승)
        Length d_max = rho + H;                              // 최대 (수평선 방향)
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
    }

    // === μs 파라미터화 (y축) - 비선형 매핑 ===
    // 지표면에서의 거리 기반 계산 (모든 고도에서 동일한 태양 각도 범위 보장)
    Length d = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, mu_s);
    Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;  // μs = 1 (천정)
    Length d_max = H;                                                  // μs = 0 (수평선)
    Number a = (d - d_min) / (d_max - d_min);

    // 최소 태양 고도(mu_s_min)에서의 값 - 정규화용
    Length D = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
    Number A = (D - d_min) / (d_max - d_min);

    // 비선형 변환: 수평선 근처에 더 많은 샘플
    Number u_mu_s = GetTextureCoordFromUnitRange(
        max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

    // === ν 파라미터화 (x축) - 단순 선형 ===
    Number u_nu = (nu + 1.0) / 2.0;

    // 반환: (u, v, w, z) = (ν, μs, μ, r)
    return float4(u_nu, u_mu_s, u_mu, u_r);
}

/**
 * 산란 텍스처 UVWZ → (r, μ, μs, ν) 역변환
 *
 * 사전 계산 단계에서 사용: 각 텍셀이 어떤 파라미터에 대응하는지 계산
 * GetScatteringTextureUvwzFromRMuMuSNu의 역함수입니다.
 */
inline void GetRMuMuSNuFromScatteringTextureUvwz(
    constant AtmosphereParameters& atmosphere,
    float4 uvwz, thread Length& r, thread Number& mu, thread Number& mu_s,
    thread Number& nu, thread bool& ray_r_mu_intersects_ground) {

    // r 복원
    Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
        atmosphere.bottom_radius * atmosphere.bottom_radius);
    Length rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
    r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

    // μ 복원 - 텍스처 절반 기준으로 분기
    if (uvwz.z < 0.5) {
        // 하위 절반: 지표면 교차
        Length d_min = r - atmosphere.bottom_radius;
        Length d_max = rho;
        Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
        // 기하학에서 μ 역계산
        mu = d == 0.0 ? Number(-1.0) :
            ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = true;
    } else {
        // 상위 절반: 하늘
        Length d_min = atmosphere.top_radius - r;
        Length d_max = rho + H;
        Length d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
        mu = d == 0.0 ? Number(1.0) :
            ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = false;
    }

    // μs 복원 - 비선형 역변환
    Number x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
    Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    Length d_max = H;
    Length D = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
    Number A = (D - d_min) / (d_max - d_min);
    // 비선형 역변환
    Number a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    Length d = d_min + min(a, A) * (d_max - d_min);
    mu_s = d == 0.0 ? Number(1.0) :
        ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

    // ν 복원 - 선형
    nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

/**
 * 3D 텍스처 fragment 좌표 → (r, μ, μs, ν) 변환
 *
 * 컴퓨트 셰이더에서 사용: 각 복셀(voxel)의 좌표를 파라미터로 변환
 *
 * 4D → 3D 매핑 디코딩:
 * x축에 μs와 ν가 결합되어 있으므로 이를 분리합니다.
 * - frag_coord_nu = floor(x / MU_S_SIZE)
 * - frag_coord_mu_s = x mod MU_S_SIZE
 */
inline void GetRMuMuSNuFromScatteringTextureFragCoord(
    constant AtmosphereParameters& atmosphere, float3 frag_coord,
    thread Length& r, thread Number& mu, thread Number& mu_s, thread Number& nu,
    thread bool& ray_r_mu_intersects_ground) {

    const float4 SCATTERING_TEXTURE_SIZE = float4(
        SCATTERING_TEXTURE_NU_SIZE - 1,
        SCATTERING_TEXTURE_MU_S_SIZE,
        SCATTERING_TEXTURE_MU_SIZE,
        SCATTERING_TEXTURE_R_SIZE);

    // x축에서 ν와 μs 분리
    Number frag_coord_nu = floor(frag_coord.x / Number(SCATTERING_TEXTURE_MU_S_SIZE));
    Number frag_coord_mu_s = fmod(frag_coord.x, Number(SCATTERING_TEXTURE_MU_S_SIZE));
    float4 uvwz = float4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
        SCATTERING_TEXTURE_SIZE;
    GetRMuMuSNuFromScatteringTextureUvwz(
        atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    // ν를 유효 범위로 클램프
    // ν = v · s 이고, v와 s 모두 μ, μs로 제약되므로
    // ν ∈ [μ·μs - sin(θ)·sin(θs), μ·μs + sin(θ)·sin(θs)]
    nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
        mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

/**
 * 단일 산란 텍스처 사전 계산 - 컴퓨트 셰이더용
 *
 * 3D 텍스처의 각 복셀에 대해 단일 산란 값을 계산합니다.
 * AtmospherePrecomputation.m에서 호출됩니다.
 */
inline void ComputeSingleScatteringTexture(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    float3 frag_coord,
    thread IrradianceSpectrum& rayleigh, thread IrradianceSpectrum& mie) {

    Length r;
    Number mu;
    Number mu_s;
    Number nu;
    bool ray_r_mu_intersects_ground;
    // 복셀 좌표 → 파라미터
    GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, frag_coord,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    // 해당 파라미터에서 단일 산란 계산
    ComputeSingleScattering(atmosphere, transmittance_texture, s,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground, rayleigh, mie);
}

// =============================================================================
// 산란 텍스처 룩업 함수 (Scattering Lookup Functions)
// =============================================================================
// 런타임에 사전 계산된 산란 텍스처를 조회합니다.
// 4D 파라미터를 3D 텍스처 좌표로 변환하여 샘플링합니다.

/**
 * 산란 텍스처에서 값 조회 (4선형 보간)
 *
 * 4D 데이터가 3D 텍스처에 저장되어 있으므로,
 * ν 축에 대해서는 수동으로 선형 보간을 수행합니다.
 *
 * 3D 텍스처는 (r, μ, μs) 3개 축에 대해 하드웨어 선형 보간을 사용하고,
 * ν 축은 인접한 두 텍스처 샘플을 선형 보간합니다.
 * 결과적으로 4선형 보간(quadrilinear interpolation)이 됩니다.
 */
inline float3 GetScattering(
    constant AtmosphereParameters& atmosphere,
    texture3d<float> scattering_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {

    // 파라미터 → 텍스처 좌표
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);

    // ν 축 수동 보간 준비
    Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
    Number tex_x = floor(tex_coord_x);     // 정수 인덱스
    Number lerp = tex_coord_x - tex_x;     // 보간 계수

    // 인접한 두 ν 슬라이스의 좌표
    float3 uvw0 = float3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);

    // 두 샘플의 선형 보간
    return float3(scattering_texture.sample(s, uvw0).rgb * (1.0 - lerp) +
        scattering_texture.sample(s, uvw1).rgb * lerp);
}

// =============================================================================
// 복사 조도 계산 (Irradiance Computation)
// =============================================================================
// 논문 Section 4의 간접 조명 계산
//
// 복사 조도(Irradiance) E는 표면에 도달하는 총 광에너지입니다.
// 두 가지 성분으로 구성됩니다:
//
// 1. 직접 조도 (Direct Irradiance):
//    태양에서 직접 오는 빛 × 대기 투과율 × cos(입사각)
//
// 2. 간접 조도 (Indirect Irradiance):
//    대기에서 산란되어 오는 빛 (하늘광, sky light)
//    복수 산란까지 포함
//
// 파라미터화:
// - r: 고도 (반지름)
// - μs: 태양 천정각 코사인
//
// 2D 텍스처에 저장: E(r, μs)

/**
 * (r, μs) → 복사 조도 텍스처 UV 좌표 변환
 *
 * 단순한 선형 파라미터화 사용 (산란 텍스처와 달리 비선형 매핑 불필요)
 * - x축: μs ∈ [-1, 1] → [0, 1]
 * - y축: r ∈ [bottom, top] → [0, 1]
 */
inline float2 GetIrradianceTextureUvFromRMuS(
    constant AtmosphereParameters& atmosphere,
    Length r, Number mu_s) {
    Number x_r = (r - atmosphere.bottom_radius) /
        (atmosphere.top_radius - atmosphere.bottom_radius);
    Number x_mu_s = mu_s * 0.5 + 0.5;  // [-1, 1] → [0, 1]
    return float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
                  GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

/**
 * 복사 조도 텍스처 UV → (r, μs) 역변환
 *
 * 사전 계산 단계에서 사용
 */
inline void GetRMuSFromIrradianceTextureUv(
    constant AtmosphereParameters& atmosphere,
    float2 uv, thread Length& r, thread Number& mu_s) {
    Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    r = atmosphere.bottom_radius +
        x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
    mu_s = ClampCosine(2.0 * x_mu_s - 1.0);  // [0, 1] → [-1, 1]
}

/**
 * 직접 복사 조도 계산
 *
 * 태양에서 표면으로 직접 도달하는 복사량:
 *   E_direct = E_sun × T(point→sun) × cos(θ_s)
 *
 * 태양이 수평선 근처일 때 특별 처리:
 * - 태양이 완전히 수평선 위: average_cosine = μs
 * - 태양이 완전히 수평선 아래: average_cosine = 0
 * - 태양이 부분적으로 보임: 부분 적분 결과 사용
 *
 * 부분 가시성 공식:
 *   average_cosine = (μs + αs)² / (4αs)
 *   여기서 αs = 태양 각반지름
 */
inline IrradianceSpectrum ComputeDirectIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    Length r, Number mu_s) {

    Number alpha_s = atmosphere.sun_angular_radius;
    // 태양 원반의 평균 코사인 계수 계산
    Number average_cosine_factor =
        mu_s < -alpha_s ? 0.0 : (mu_s > alpha_s ? mu_s :
            (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

    return atmosphere.solar_irradiance *
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, s, r, mu_s) * average_cosine_factor;
}

/**
 * 직접 복사 조도 텍스처 사전 계산 - 컴퓨트 셰이더용
 *
 * 각 텍셀의 fragment 좌표를 받아 해당하는 E_direct(r, μs)를 계산합니다.
 */
inline IrradianceSpectrum ComputeDirectIrradianceTexture(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    sampler s,
    float2 frag_coord) {

    const float2 IRRADIANCE_TEXTURE_SIZE =
        float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);
    Length r;
    Number mu_s;
    GetRMuSFromIrradianceTextureUv(
        atmosphere, frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
    return ComputeDirectIrradiance(atmosphere, transmittance_texture, s, r, mu_s);
}

/**
 * 복사 조도 텍스처에서 값 조회
 *
 * 사전 계산된 조도 텍스처를 샘플링합니다.
 * 간접 조도(sky light)까지 포함된 총 조도를 반환합니다.
 */
inline IrradianceSpectrum GetIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> irradiance_texture,
    sampler s,
    Length r, Number mu_s) {
    float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
    return IrradianceSpectrum(irradiance_texture.sample(s, uv).rgb);
}

// =============================================================================
// 렌더링 함수 (Rendering Functions)
// =============================================================================
// 논문 Section 5: 런타임 렌더링
//
// 이 함수들은 실제 렌더링에서 호출됩니다.
// 사전 계산된 텍스처를 조회하여 최종 색상을 계산합니다.
//
// 렌더링 과정:
// 1. 카메라 위치와 시선 방향에서 파라미터 (r, μ, μs, ν) 계산
// 2. 산란 텍스처에서 값 조회
// 3. 위상 함수 적용하여 최종 휘도(radiance) 계산
// 4. 톤 매핑 및 감마 보정

/**
 * Combined 텍스처에서 단일 Mie 산란 추출
 *
 * 메모리 최적화 기법:
 * Rayleigh와 Mie 산란을 별도 텍스처에 저장하면 2배의 메모리가 필요합니다.
 * 대신 "combined" 텍스처를 사용:
 * - RGB: Rayleigh + 복수 산란 (Mie 포함)
 * - A: 단일 Mie 산란의 R 채널
 *
 * 단일 Mie 산란의 G, B 채널은 스펙트럼 비율로 추정합니다:
 *   Mie_rgb = combined.a * (β_M / β_R) * (β_R.r / β_M.r)
 *
 * 이 근사는 Mie 산란이 파장에 거의 독립적이라는 가정에 기반합니다.
 */
inline float3 GetExtrapolatedSingleMieScattering(
    constant AtmosphereParameters& atmosphere, float4 scattering) {
    if (scattering.r <= 0.0) {
        return float3(0.0);
    }
    // combined.a / combined.r = Mie.r / (Rayleigh.r + multiple.r)
    // 이를 스펙트럼 전체로 확장
    return scattering.rgb * scattering.a / scattering.r *
        (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
        (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}

/**
 * 결합된 산란 값 조회 (Rayleigh + 복수 산란 + 단일 Mie)
 *
 * 렌더링에서 가장 자주 호출되는 핵심 함수입니다.
 *
 * 두 가지 모드:
 * 1. use_combined_textures = true:
 *    - 단일 RGBA 텍스처 사용 (RGB: Rayleigh+multiple, A: Mie.r)
 *    - 메모리 효율적, 근사적 Mie 복원
 *
 * 2. use_combined_textures = false:
 *    - Rayleigh/복수 산란과 Mie 산란을 별도 텍스처에 저장
 *    - 더 정확하지만 2배 메모리 사용
 *
 * @return Rayleigh + 복수 산란 값 (위상 함수 미적용)
 * @param single_mie_scattering [out] 단일 Mie 산란 값 (위상 함수 미적용)
 */
inline IrradianceSpectrum GetCombinedScattering(
    constant AtmosphereParameters& atmosphere,
    texture3d<float> scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    sampler s,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground,
    bool use_combined_textures,
    thread IrradianceSpectrum& single_mie_scattering) {

    // 4선형 보간을 위한 텍스처 좌표 계산
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
    Number tex_x = floor(tex_coord_x);
    Number lerp = tex_coord_x - tex_x;
    float3 uvw0 = float3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);
    float3 uvw1 = float3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
        uvwz.z, uvwz.w);

    if (use_combined_textures) {
        // Combined 모드: RGBA 텍스처에서 Mie 추출
        float4 combined_scattering =
            scattering_texture.sample(s, uvw0) * (1.0 - lerp) +
            scattering_texture.sample(s, uvw1) * lerp;
        IrradianceSpectrum scattering = IrradianceSpectrum(combined_scattering.rgb);
        single_mie_scattering = GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
        return scattering;
    } else {
        // 분리 모드: 별도 텍스처에서 조회
        IrradianceSpectrum scattering = IrradianceSpectrum(
            scattering_texture.sample(s, uvw0).rgb * (1.0 - lerp) +
            scattering_texture.sample(s, uvw1).rgb * lerp);
        single_mie_scattering = IrradianceSpectrum(
            single_mie_scattering_texture.sample(s, uvw0).rgb * (1.0 - lerp) +
            single_mie_scattering_texture.sample(s, uvw1).rgb * lerp);
        return scattering;
    }
}

/**
 * 태양 원반의 휘도(Radiance) 계산
 *
 * 태양을 균일한 밝기의 원반으로 모델링합니다.
 *   L_sun = E_sun / (π × α_s²)
 *
 * 여기서:
 * - E_sun: 태양 복사 조도 (W/m²)
 * - α_s: 태양 각반지름 (라디안)
 *
 * 투과율은 별도로 적용됩니다 (GetSkyRadiance에서).
 */
inline RadianceSpectrum GetSolarRadiance(constant AtmosphereParameters& atmosphere) {
    return atmosphere.solar_irradiance /
        (PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius);
}

/**
 * 하늘 휘도 계산 - 논문 Section 5의 핵심 렌더링 함수
 *
 * 카메라에서 특정 방향을 바라볼 때의 하늘 색상을 계산합니다.
 * 대기 산란으로 인한 모든 색상 기여를 합산합니다.
 *
 * 수학적 표현:
 *   L(camera, view) = S(camera→∞) × P(ν)
 *
 * 여기서:
 * - S: 사전 계산된 산란 값 (Rayleigh + Mie + 복수 산란)
 * - P(ν): 위상 함수 (시선-태양 각도에 따른 산란 강도)
 *
 * shadow_length 처리:
 * - 0이면 일반적인 하늘 렌더링
 * - > 0이면 광선 축(light shaft) 효과 적용
 *   그림자 영역에서는 태양광이 차단되어 산란이 감소
 *
 * @param camera 카메라 위치 (지구 중심 기준, 미터)
 * @param view_ray 시선 방향 (정규화)
 * @param shadow_length 그림자 영역 길이 (광선 축 효과용)
 * @param sun_direction 태양 방향 (정규화)
 * @param transmittance [out] 해당 방향의 투과율 (태양 원반 렌더링에 필요)
 */
inline RadianceSpectrum GetSkyRadiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    texture3d<float> scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    sampler s,
    Position camera, Direction view_ray, Length shadow_length,
    Direction sun_direction,
    bool use_combined_textures,
    thread DimensionlessSpectrum& transmittance) {

    // 카메라 위치에서 파라미터 계산
    Length r = length(camera);  // 지구 중심으로부터의 거리
    Length rmu = dot(camera, view_ray);  // r × μ

    // 카메라가 대기 밖에 있으면 대기 경계로 이동
    Length distance_to_top_atmosphere_boundary = -rmu -
        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);

    if (distance_to_top_atmosphere_boundary > 0.0) {
        // 카메라를 대기 경계로 이동
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = atmosphere.top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    } else if (r > atmosphere.top_radius) {
        // 대기 밖에서 대기를 보지 않는 경우 → 검은색
        transmittance = DimensionlessSpectrum(1.0);
        return RadianceSpectrum(0.0);
    }

    // 텍스처 조회용 파라미터 계산
    Number mu = rmu / r;                            // 시선 방향 코사인
    Number mu_s = dot(camera, sun_direction) / r;   // 태양 방향 코사인
    Number nu = dot(view_ray, sun_direction);       // 시선-태양 각도 코사인
    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

    // 투과율 계산 (태양 원반 렌더링에 필요)
    transmittance = ray_r_mu_intersects_ground ? DimensionlessSpectrum(0.0) :
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, s, r, mu);

    IrradianceSpectrum single_mie_scattering;
    IrradianceSpectrum scattering;

    if (shadow_length == 0.0) {
        // 일반 하늘 렌더링
        scattering = GetCombinedScattering(
            atmosphere, scattering_texture, single_mie_scattering_texture, s,
            r, mu, mu_s, nu, ray_r_mu_intersects_ground,
            use_combined_textures, single_mie_scattering);
    } else {
        // 광선 축 효과: 그림자 영역을 건너뛴 위치에서 산란 조회
        Length d = shadow_length;
        Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
        Number mu_p = (r * mu + d) / r_p;
        Number mu_s_p = (r * mu_s + d * nu) / r_p;

        scattering = GetCombinedScattering(
            atmosphere, scattering_texture, single_mie_scattering_texture, s,
            r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
            use_combined_textures, single_mie_scattering);
        // 그림자 영역의 투과율 적용
        DimensionlessSpectrum shadow_transmittance =
            GetTransmittance(atmosphere, transmittance_texture, s,
                r, mu, shadow_length, ray_r_mu_intersects_ground);
        scattering = scattering * shadow_transmittance;
        single_mie_scattering = single_mie_scattering * shadow_transmittance;
    }

    // 위상 함수 적용하여 최종 휘도 계산
    // Rayleigh: P_R(ν) = 3/(16π) × (1 + cos²ν)
    // Mie: P_M(ν, g) = Cornette-Shanks 함수
    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

/**
 * 특정 지점까지의 하늘 휘도 계산 - 공중 원근법(Aerial Perspective)
 *
 * 카메라에서 특정 3D 지점을 바라볼 때, 그 사이 대기에서의 산란을 계산합니다.
 * GetSkyRadiance와 달리, 무한대가 아닌 유한 거리까지의 산란만 계산합니다.
 *
 * 용도:
 * - 지표면/객체 렌더링 시 대기 안개 효과 적용
 * - 멀리 있는 물체가 푸르게 보이는 현상 (공중 원근법)
 * - 산 등이 거리에 따라 희미해지는 효과
 *
 * 수학적 표현:
 *   L_inscatter = S(camera→point) - T(camera→point) × S(point→∞)
 *
 * 첫 번째 항: 카메라부터 무한대까지의 총 산란
 * 두 번째 항: point부터 무한대까지의 산란 (투과율 적용)
 * 차이 = camera→point 구간의 산란
 *
 * 이 값을 객체 색상에 더하고, 투과율을 곱합니다:
 *   final_color = object_color × transmittance + inscatter
 *
 * @param camera 카메라 위치 (미터)
 * @param point 목표 지점 (미터)
 * @param shadow_length 광선 축 효과용 그림자 길이
 * @param transmittance [out] camera→point 투과율
 */
inline RadianceSpectrum GetSkyRadianceToPoint(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    texture3d<float> scattering_texture,
    texture3d<float> single_mie_scattering_texture,
    sampler s,
    Position camera, Position point, Length shadow_length,
    Direction sun_direction,
    bool use_combined_textures,
    thread DimensionlessSpectrum& transmittance) {

    // 시선 방향과 거리 계산
    Direction view_ray = normalize(point - camera);
    Length r = length(camera);
    Length rmu = dot(camera, view_ray);

    // 카메라가 대기 밖이면 대기 경계로 이동
    Length distance_to_top_atmosphere_boundary = -rmu -
        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);

    if (distance_to_top_atmosphere_boundary > 0.0) {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = atmosphere.top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    }

    // 파라미터 계산
    Number mu = rmu / r;
    Number mu_s = dot(camera, sun_direction) / r;
    Number nu = dot(view_ray, sun_direction);
    Length d = length(point - camera);  // 목표 지점까지의 거리
    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

    // 목표 지점까지의 투과율
    transmittance = GetTransmittance(atmosphere, transmittance_texture, s,
        r, mu, d, ray_r_mu_intersects_ground);

    // 카메라에서 무한대까지의 산란 (S_0)
    IrradianceSpectrum single_mie_scattering;
    IrradianceSpectrum scattering = GetCombinedScattering(
        atmosphere, scattering_texture, single_mie_scattering_texture, s,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground,
        use_combined_textures, single_mie_scattering);

    // 목표 지점에서 무한대까지의 산란 (S_p)
    // 그림자 영역을 고려하여 계산
    d = max(d - shadow_length, 0.0);
    Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    Number mu_p = (r * mu + d) / r_p;
    Number mu_s_p = (r * mu_s + d * nu) / r_p;

    IrradianceSpectrum single_mie_scattering_p;
    IrradianceSpectrum scattering_p = GetCombinedScattering(
        atmosphere, scattering_texture, single_mie_scattering_texture, s,
        r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
        use_combined_textures, single_mie_scattering_p);

    // 그림자 영역 투과율
    DimensionlessSpectrum shadow_transmittance = transmittance;
    if (shadow_length > 0.0) {
        shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture, s,
            r, mu, d, ray_r_mu_intersects_ground);
    }

    // 구간 산란 = S_0 - T(camera→point) × S_p
    scattering = scattering - shadow_transmittance * scattering_p;
    single_mie_scattering = single_mie_scattering - shadow_transmittance * single_mie_scattering_p;

    // Combined 텍스처 모드일 때, 뺄셈 후 Mie 산란 재추출
    // (뺄셈 후 scattering과 single_mie_scattering의 관계가 깨질 수 있으므로 재계산 필요)
    if (use_combined_textures) {
        single_mie_scattering = GetExtrapolatedSingleMieScattering(
            atmosphere, float4(scattering, single_mie_scattering.r));
    }

    // 태양이 수평선 아래일 때 Mie 산란 페이드 아웃
    // (수평선 아래에서는 직사광이 없으므로 단일 Mie 산란도 없음)
    single_mie_scattering = single_mie_scattering *
        smoothstep(Number(0.0), Number(0.01), mu_s);

    // 위상 함수 적용
    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

/**
 * 지표면/객체에 도달하는 태양 및 하늘 복사 조도 계산
 *
 * 지표면이나 객체의 조명을 계산할 때 사용합니다.
 * 두 가지 광원을 고려합니다:
 *
 * 1. 직사광 (Direct Sun Irradiance):
 *    태양에서 직접 오는 빛
 *    E_sun = E_0 × T(point→sun) × max(N·L, 0)
 *
 * 2. 하늘광 (Sky Irradiance):
 *    대기에서 산란되어 오는 빛 (간접 조명)
 *    하늘 전체가 광원으로 작용
 *
 * 하늘광 계산:
 *   사전 계산된 조도 텍스처를 사용합니다.
 *   법선 방향 보정: (1 + N·up) / 2
 *   - 위를 향한 면: 전체 하늘광 수신
 *   - 수평면: 절반의 하늘광 수신
 *   - 아래를 향한 면: 하늘광 없음
 *
 * 사용 예:
 *   float3 sky_irr;
 *   float3 sun_irr = GetSunAndSkyIrradiance(..., sky_irr);
 *   float3 color = albedo * (1/PI) * (sun_irr + sky_irr);
 *
 * @param point 표면 위치 (지구 중심 기준, 미터)
 * @param normal 표면 법선 (정규화)
 * @param sun_direction 태양 방향 (정규화)
 * @param sky_irradiance [out] 하늘광 조도
 * @return 직사광 조도
 */
inline IrradianceSpectrum GetSunAndSkyIrradiance(
    constant AtmosphereParameters& atmosphere,
    texture2d<float> transmittance_texture,
    texture2d<float> irradiance_texture,
    sampler s,
    Position point, Direction normal, Direction sun_direction,
    thread IrradianceSpectrum& sky_irradiance) {

    // 해당 위치의 고도와 태양 방향 코사인
    Length r = length(point);
    Number mu_s = dot(point, sun_direction) / r;

    // 하늘광 조도 = 사전 계산된 조도 × 법선 방향 보정
    // (1 + N·up) / 2 = 위를 향할수록 더 많은 하늘광 수신
    // up = point / r (천정 방향)
    sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, s, r, mu_s) *
        (1.0 + dot(normal, point) / r) * 0.5;

    // 직사광 조도 = 태양 복사량 × 투과율 × lambert 코사인
    return atmosphere.solar_irradiance *
        GetTransmittanceToSun(atmosphere, transmittance_texture, s, r, mu_s) *
        max(dot(normal, sun_direction), 0.0);
}
