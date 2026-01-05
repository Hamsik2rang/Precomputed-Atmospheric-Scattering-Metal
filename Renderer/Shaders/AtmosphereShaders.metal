/**
 * ============================================================================
 * 논문 기반 주석 (Paper-Based Comments)
 * ============================================================================
 * 이 파일은 "Precomputed Atmospheric Scattering" 논문의 렌더링 단계를 구현합니다.
 * AtmosphereFunctions.metal에서 제공하는 산란/투과율 함수를 사용하여
 * 최종 화면에 표시될 색상을 계산합니다.
 *
 * 렌더링 파이프라인:
 * 1. Vertex Shader: 풀스크린 쿼드 + 시선 방향(view ray) 계산
 * 2. Fragment Shader: 각 픽셀에서의 하늘/지표면/객체 색상 계산
 *
 * 주요 렌더링 요소:
 * - 하늘 (Sky): GetSkyRadiance() 호출
 * - 태양 원반 (Sun Disc): GetSolarRadiance() × transmittance
 * - 지표면 (Ground): GetSunAndSkyIrradiance() × albedo
 * - 공중 원근법 (Aerial Perspective): GetSkyRadianceToPoint()
 * - 광선 축 (Light Shafts): shadow_length 파라미터 사용
 *
 * 좌표계:
 * - 모든 위치는 지구 중심 기준 (미터 단위로 대기 함수에 전달)
 * - 셰이더 내부에서는 km 단위(kLengthUnitInMeters=1000) 사용
 * - 대기 함수 호출 시 × kLengthUnitInMeters로 변환 필요
 */

#include <metal_stdlib>
using namespace metal;

#import "../Atmosphere/AtmosphereShaderTypes.h"
#import "../Atmosphere/AtmosphereConstants.h"

// 대기 산란 함수 라이브러리 포함
#import "../Atmosphere/AtmosphereFunctions.metal"

// ============================================================================
// 상수 정의 (Constants)
// ============================================================================

// 단위 변환 상수: km → m
// 셰이더에서는 km 단위를 사용하고, 대기 함수 호출 시 미터로 변환합니다.
// 이렇게 하는 이유: 수치 정밀도 (6,360,000m vs 6,360km)
constant float kLengthUnitInMeters = 1000.0;

// 데모 장면 구체 (1km 반지름, 1km 고도에 위치)
// 원본 demo.glsl과 동일한 설정
constant float3 kSphereCenter = float3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;  // km 단위
constant float kSphereRadius = 1000.0 / kLengthUnitInMeters;                      // km 단위
constant float3 kSphereAlbedo = float3(0.8);      // 구체 반사율 (밝은 회색)
constant float3 kGroundAlbedo = float3(0.0, 0.0, 0.04);  // 지표면 반사율 (어두운 파란색)

// ============================================================================
// 셰이더 입출력 구조체 (Shader I/O Structures)
// ============================================================================

// 버텍스 입력 (사용하지 않음 - 풀스크린 쿼드는 vertexID로 생성)
struct AtmosphereVertexIn {
    float4 position [[attribute(0)]];
};

// 버텍스 출력 / 프래그먼트 입력
struct AtmosphereVertexOut {
    float4 position [[position]];  // 클립 공간 위치 (래스터라이저용)
    float3 view_ray;               // 월드 공간 시선 방향 (정규화되지 않음)
};

// 프래그먼트 출력
struct AtmosphereFragmentOut {
    float4 color [[color(0)]];     // 최종 색상 (톤매핑/감마 보정 적용됨)
};

// ============================================================================
// 버텍스 셰이더 (Vertex Shader)
// ============================================================================
// 풀스크린 쿼드를 렌더링하고, 각 픽셀의 시선 방향(view ray)을 계산합니다.
//
// 대기 렌더링은 화면 전체를 덮는 후처리(post-process) 형태입니다.
// 각 픽셀에서 카메라 → 시선 방향으로 레이마칭을 수행합니다.

vertex AtmosphereVertexOut atmosphereVertexShader(
    uint vertexID [[vertex_id]],
    constant AtmosphereUniforms &uniforms [[buffer(BufferIndexUniforms)]])
{
    // 풀스크린 쿼드 정점 (트라이앵글 스트립)
    // NDC 좌표계: (-1,-1) 좌하단, (1,1) 우상단
    const float2 positions[4] = {
        float2(-1.0, -1.0),  // 좌하단
        float2( 1.0, -1.0),  // 우하단
        float2(-1.0,  1.0),  // 좌상단
        float2( 1.0,  1.0)   // 우상단
    };

    AtmosphereVertexOut out;
    float2 pos = positions[vertexID];
    out.position = float4(pos, 0.0, 1.0);

    // 시선 방향 계산:
    // 1. NDC 위치 → 클립 공간
    // 2. 클립 공간 → 뷰 공간 (view_from_clip 역투영 행렬)
    // 3. 뷰 공간 → 월드 공간 (model_from_view 회전 행렬)
    float4 clipPos = float4(pos.x, pos.y, 0.0, 1.0);
    float4 viewPos = uniforms.view_from_clip * clipPos;
    float3 viewDir = viewPos.xyz;

    // 카메라 공간 → 월드 공간 변환
    // w=0으로 방향 벡터만 회전 (이동 적용 안 함)
    out.view_ray = (uniforms.model_from_view * float4(viewDir, 0.0)).xyz;

    return out;
}

// ============================================================================
// 그림자 및 가시성 함수 (Shadow and Visibility Functions)
// ============================================================================
// 이 섹션의 함수들은 논문의 핵심 대기 산란 모델이 아닌,
// 데모 장면에서 구체(sphere)에 의한 그림자 효과를 계산합니다.
//
// 구현된 효과:
// 1. GetSunVisibility(): 구체가 태양을 가릴 때의 소프트 섀도우
// 2. GetSkyVisibility(): 구체 주변의 앰비언트 오클루전
// 3. GetSphereShadowInOut(): 광선 축(light shafts) 효과를 위한 그림자 원뿔 계산
//
// 소프트 섀도우 원리:
// - 태양은 점광원이 아닌 각 크기(sun_size)를 가진 원반입니다.
// - 구체 가장자리에서 부드러운 그림자 전이(penumbra)가 발생합니다.
// - smoothstep()으로 부드러운 경계 처리

/**
 * 태양 가시성 계산 - 구체에 의한 소프트 섀도우
 *
 * @param point         그림자를 계산할 위치 (km 단위)
 * @param sun_direction 태양 방향 (정규화된 벡터)
 * @param sun_size      x: tan(angular_radius), y: cos(angular_radius)
 * @return              가시성 (0.0 = 완전한 그림자, 1.0 = 완전히 보임)
 *
 * 알고리즘:
 * 1. point에서 태양 방향으로 레이를 발사
 * 2. 레이-구체 교차 검사
 * 3. 교차 시, 각도 거리를 기준으로 소프트 섀도우 계산
 */
float GetSunVisibility(float3 point, float3 sun_direction, float2 sun_size) {
    float3 p = point - kSphereCenter;
    float p_dot_v = dot(p, sun_direction);
    float p_dot_p = dot(p, p);
    float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
    float discriminant = kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;

    if (discriminant >= 0.0) {
        float distance_to_intersection = -p_dot_v - sqrt(discriminant);
        if (distance_to_intersection > 0.0) {
            float ray_sphere_distance = kSphereRadius - sqrt(ray_sphere_center_squared_distance);
            float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
            return smoothstep(1.0, 0.0, ray_sphere_angular_distance / sun_size.x);
        }
    }
    return 1.0;  // 그림자 밖 = 완전히 보임
}

/**
 * 하늘 가시성 계산 - 구체에 의한 앰비언트 오클루전
 *
 * @param point 가시성을 계산할 위치 (km 단위)
 * @return      하늘 가시성 (0.0~1.0)
 *
 * 원리:
 * - 구체 아래쪽 점은 하늘의 일부가 구체에 가려집니다.
 * - 간단한 근사식으로 구체 가림 비율을 계산합니다.
 * - 정확한 구적분 대신 빠른 근사치를 사용합니다.
 */
float GetSkyVisibility(float3 point) {
    float3 p = point - kSphereCenter;
    float p_dot_p = dot(p, p);
    return 1.0 + p.z / sqrt(p_dot_p) * kSphereRadius * kSphereRadius / p_dot_p;
}

/**
 * 그림자 원뿔 교차점 계산 - 광선 축(Light Shafts) 효과용
 *
 * @param camera         카메라 위치 (km 단위)
 * @param view_direction 시선 방향 (정규화됨)
 * @param sun_direction  태양 방향 (정규화됨)
 * @param sun_size       x: tan(angular_radius), y: cos(angular_radius)
 * @param d_in           [출력] 시선이 그림자에 들어가는 거리
 * @param d_out          [출력] 시선이 그림자에서 나오는 거리
 *
 * 광선 축(Light Shafts) 원리:
 * - 구체가 태양빛을 가리면 원뿔 형태의 그림자가 생깁니다.
 * - 시선(view ray)이 이 그림자 원뿔을 통과하는 구간을 계산합니다.
 * - 그림자 구간에서는 산란광이 감소하여 어두운 줄무늬가 보입니다.
 *
 * 기하학적 구조:
 *   [태양] ----> [구체] ----> [그림자 원뿔]
 *                  ^               ^
 *              원뿔 정점        원뿔 밑면
 *                           (무한히 확장)
 *
 * 출력 사용법:
 * - shadow_length = max(0, min(d_out, distance) - d_in)
 * - 이 길이만큼 그림자 영역을 통과하므로 산란 계산에서 제외
 */
void GetSphereShadowInOut(float3 camera, float3 view_direction, float3 sun_direction,
                          float2 sun_size, thread float &d_in, thread float &d_out) {
    float3 pos = camera - kSphereCenter;
    float pos_dot_sun = dot(pos, sun_direction);
    float view_dot_sun = dot(view_direction, sun_direction);
    float k = sun_size.x;
    float l = 1.0 + k * k;
    float a = 1.0 - l * view_dot_sun * view_dot_sun;
    float b = dot(pos, view_direction) - l * pos_dot_sun * view_dot_sun -
              k * kSphereRadius * view_dot_sun;
    float c = dot(pos, pos) - l * pos_dot_sun * pos_dot_sun -
              2.0 * k * kSphereRadius * pos_dot_sun - kSphereRadius * kSphereRadius;
    float discriminant = b * b - a * c;

    if (discriminant > 0.0) {
        d_in = max(0.0, (-b - sqrt(discriminant)) / a);
        d_out = (-b + sqrt(discriminant)) / a;

        float d_base = -pos_dot_sun / view_dot_sun;
        float d_apex = -(pos_dot_sun + kSphereRadius / k) / view_dot_sun;

        if (view_dot_sun > 0.0) {
            d_in = max(d_in, d_apex);
            d_out = a > 0.0 ? min(d_out, d_base) : d_base;
        } else {
            d_in = a > 0.0 ? max(d_in, d_base) : d_base;
            d_out = min(d_out, d_apex);
        }
    } else {
        d_in = 0.0;
        d_out = 0.0;
    }
}

// ============================================================================
// 프래그먼트 셰이더 (Fragment Shader)
// ============================================================================
// 논문의 렌더링 단계를 구현합니다. 각 픽셀에서 다음을 계산:
//
// 1. 하늘 색상: GetSkyRadiance() - 논문 Section 4
//    - 카메라에서 시선 방향으로 무한히 먼 점까지의 산란광
//    - Rayleigh + Mie 산란의 합
//
// 2. 태양 원반: GetSolarRadiance() × transmittance
//    - 태양의 각 크기(~0.5°) 내에 있으면 태양 복사 추가
//    - 대기 투과율을 곱해 감쇠된 태양빛 표현
//
// 3. 지표면/객체: GetSunAndSkyIrradiance() × albedo
//    - 직사광(sun_irradiance): 태양에서 직접 오는 빛
//    - 간접광(sky_irradiance): 대기 산란에 의한 하늘빛
//    - 람베르트 BRDF: albedo / π
//
// 4. 공중 원근법: GetSkyRadianceToPoint() - 논문 Section 4
//    - 카메라와 객체 사이의 대기 산란
//    - 멀리 있는 물체일수록 파랗게/흐릿하게 보이는 효과
//
// 5. 광선 축: shadow_length 파라미터
//    - 그림자 영역에서는 태양빛 산란이 감소
//    - 구름 틈새로 비치는 빛줄기 효과
//
// 합성 순서 (뒤에서 앞으로):
//   하늘 → 지표면 → 구체 (alpha blending)

fragment AtmosphereFragmentOut atmosphereFragmentShader(
    AtmosphereVertexOut in [[stage_in]],
    constant AtmosphereParameters &atmosphere [[buffer(BufferIndexAtmosphere)]],
    constant AtmosphereUniforms &uniforms [[buffer(BufferIndexUniforms)]],
    texture2d<float> transmittance_texture [[texture(TextureIndexTransmittance)]],
    texture3d<float> scattering_texture [[texture(TextureIndexScattering)]],
    texture2d<float> irradiance_texture [[texture(TextureIndexIrradiance)]],
    texture3d<float> single_mie_scattering_texture [[texture(TextureIndexSingleMieScattering)]],
    sampler textureSampler [[sampler(0)]])
{
    AtmosphereFragmentOut out;

    // ========================================================================
    // 초기화 및 사전 계산
    // ========================================================================

    // 시선 방향 정규화
    // 버텍스 셰이더에서 보간된 view_ray를 정규화하여 단위 벡터로 만듦
    float3 view_direction = normalize(in.view_ray);

    // 픽셀 각도 크기 (안티앨리어싱용)
    // OpenGL의 dFdx/dFdy 대신 근사값 사용
    // 이 값은 구체/지표면 경계의 부드러운 블렌딩에 사용됨
    float fragment_angular_size = 0.0001;

    // 광선 축(light shafts) 계산을 위한 그림자 원뿔 교차점
    // shadow_in ~ shadow_out 구간이 그림자 영역
    float shadow_in, shadow_out;
    GetSphereShadowInOut(uniforms.camera, view_direction, uniforms.sun_direction,
                         uniforms.sun_size, shadow_in, shadow_out);

    // 광선 축 페이드인
    // 태양이 지평선 근처일 때 광선 축 효과를 부드럽게 감소시킴
    // (태양이 수평선 아래로 가면 그림자가 없어지므로)
    float3 up_direction = normalize(uniforms.camera - uniforms.earth_center);
    float lightshaft_fadein_hack = smoothstep(0.02, 0.04,
                                               dot(up_direction, uniforms.sun_direction));

    // ========================================================================
    // 데모 구체 교차 계산 (Demo Sphere Intersection)
    // ========================================================================
    // 시선(view ray)과 데모 구체의 교차점을 계산합니다.
    // 레이-구체 교차는 이차 방정식을 풀어 계산합니다:
    //   |camera + t * view_direction - sphere_center|² = radius²
    //
    // 전개하면:
    //   t² + 2(p·v)t + (p·p - r²) = 0
    //   여기서 p = camera - sphere_center, v = view_direction
    //
    // 판별식 discriminant = r² - (p·p - (p·v)²)
    //                     = r² - ray_sphere_center_squared_distance

    float3 p = uniforms.camera - kSphereCenter;
    float p_dot_v = dot(p, view_direction);          // 레이 방향으로 투영된 거리
    float p_dot_p = dot(p, p);                       // 카메라-구체중심 거리²
    float ray_sphere_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;  // 레이-구체중심 최단거리²
    float discriminant = kSphereRadius * kSphereRadius - ray_sphere_center_squared_distance;

    float sphere_alpha = 0.0;           // 구체 가시성 (0=투명, 1=불투명)
    float3 sphere_radiance = float3(0.0);  // 구체에서 오는 복사휘도

    if (discriminant >= 0.0) {  // 레이가 구체와 교차함
        // 교차점까지의 거리 (가까운 쪽)
        float distance_to_intersection = -p_dot_v - sqrt(discriminant);

        if (distance_to_intersection > 0.0) {  // 카메라 앞에 있음
            // 안티앨리어싱 알파 계산
            // 구체 경계에서 부드러운 블렌딩을 위해 각도 거리 사용
            float ray_sphere_distance = kSphereRadius - sqrt(ray_sphere_center_squared_distance);
            float ray_sphere_angular_distance = -ray_sphere_distance / p_dot_v;
            sphere_alpha = min(ray_sphere_angular_distance / fragment_angular_size, 1.0);

            // 교차점과 법선 계산
            float3 point = uniforms.camera + view_direction * distance_to_intersection;
            float3 normal = normalize(point - kSphereCenter);

            // ----------------------------------------------------------------
            // 구체 조명 계산 - 논문 Section 5: Rendering
            // ----------------------------------------------------------------
            // GetSunAndSkyIrradiance()로 직사광 + 하늘광 복사조도 획득
            // 주의: 대기 함수는 미터 단위 위치를 기대함 → × kLengthUnitInMeters
            float3 sky_irradiance;
            float3 sun_irradiance = GetSunAndSkyIrradiance(
                atmosphere,
                transmittance_texture,
                irradiance_texture,
                textureSampler,
                (point - uniforms.earth_center) * kLengthUnitInMeters,  // 미터 변환
                normal,
                uniforms.sun_direction,
                sky_irradiance);

            // 람베르트 BRDF 적용: L = (ρ/π) × E
            // 여기서 ρ = albedo, E = irradiance
            sphere_radiance = kSphereAlbedo * (1.0 / PI) * (sun_irradiance + sky_irradiance);

            // ----------------------------------------------------------------
            // 공중 원근법(Aerial Perspective) - 논문 Section 4
            // ----------------------------------------------------------------
            // 카메라와 구체 사이의 대기 산란을 계산합니다.
            // - transmittance: 구체에서 오는 빛의 감쇠
            // - in_scatter: 카메라-구체 경로에서 산란된 빛 추가
            //
            // 최종 색상 = 원래 색상 × transmittance + in_scatter

            // 그림자 영역 길이 계산 (광선 축 효과)
            float shadow_length = max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
                                  lightshaft_fadein_hack * kLengthUnitInMeters;
            float3 transmittance;
            float3 in_scatter = GetSkyRadianceToPoint(
                atmosphere,
                transmittance_texture,
                scattering_texture,
                single_mie_scattering_texture,
                textureSampler,
                (uniforms.camera - uniforms.earth_center) * kLengthUnitInMeters,  // 카메라 (미터)
                (point - uniforms.earth_center) * kLengthUnitInMeters,             // 구체 표면 (미터)
                shadow_length,
                uniforms.sun_direction,
                true,  // use_combined_textures: Mie 산란이 alpha 채널에 있음
                transmittance);

            // 공중 원근법 적용
            sphere_radiance = sphere_radiance * transmittance + in_scatter;
        }
    }

    // ========================================================================
    // 지표면(행성) 교차 계산 (Ground/Planet Intersection)
    // ========================================================================
    // 시선이 지구 표면과 교차하는지 검사합니다.
    // 구체 교차와 동일한 알고리즘이지만, 지구 중심과 반지름을 사용합니다.
    //
    // 좌표계 참고:
    // - earth_center = (0, 0, -bottom_radius/kLengthUnitInMeters)
    // - 지표면 반지름 = |earth_center.z| = bottom_radius / kLengthUnitInMeters

    p = uniforms.camera - uniforms.earth_center;
    p_dot_v = dot(p, view_direction);
    p_dot_p = dot(p, p);
    float ray_earth_center_squared_distance = p_dot_p - p_dot_v * p_dot_v;
    float earth_radius = -uniforms.earth_center.z;  // bottom_radius (z가 음수이므로 부호 반전)
    discriminant = earth_radius * earth_radius - ray_earth_center_squared_distance;

    float ground_alpha = 0.0;               // 지표면 가시성
    float3 ground_radiance = float3(0.0);   // 지표면에서 오는 복사휘도

    if (discriminant >= 0.0) {  // 레이가 지표면과 교차함
        float distance_to_intersection = -p_dot_v - sqrt(discriminant);

        if (distance_to_intersection > 0.0) {  // 카메라 앞에 있음 (아래를 보는 경우)
            float3 point = uniforms.camera + view_direction * distance_to_intersection;
            float3 normal = normalize(point - uniforms.earth_center);  // 지표면 법선 = 상향 벡터

            // ----------------------------------------------------------------
            // 지표면 조명 계산 - 논문 Section 5: Rendering
            // ----------------------------------------------------------------
            // 구체와 동일하게 직사광 + 하늘광 복사조도 획득
            float3 sky_irradiance;
            float3 sun_irradiance = GetSunAndSkyIrradiance(
                atmosphere,
                transmittance_texture,
                irradiance_texture,
                textureSampler,
                (point - uniforms.earth_center) * kLengthUnitInMeters,  // 미터 변환
                normal,
                uniforms.sun_direction,
                sky_irradiance);

            // 가시성 팩터 적용 (데모 구체에 의한 그림자)
            // - GetSunVisibility(): 태양이 구체에 가려지면 직사광 감소
            // - GetSkyVisibility(): 구체 아래면 하늘빛 차단
            ground_radiance = kGroundAlbedo * (1.0 / PI) * (
                sun_irradiance * GetSunVisibility(point, uniforms.sun_direction, uniforms.sun_size) +
                sky_irradiance * GetSkyVisibility(point));

            // ----------------------------------------------------------------
            // 공중 원근법 - 논문 Section 4
            // ----------------------------------------------------------------
            // 카메라-지표면 경로의 대기 산란
            float shadow_length = max(0.0, min(shadow_out, distance_to_intersection) - shadow_in) *
                                  lightshaft_fadein_hack * kLengthUnitInMeters;
            float3 transmittance;
            float3 in_scatter = GetSkyRadianceToPoint(
                atmosphere,
                transmittance_texture,
                scattering_texture,
                single_mie_scattering_texture,
                textureSampler,
                (uniforms.camera - uniforms.earth_center) * kLengthUnitInMeters,
                (point - uniforms.earth_center) * kLengthUnitInMeters,
                shadow_length,
                uniforms.sun_direction,
                true,  // use_combined_textures
                transmittance);

            // 공중 원근법 적용
            ground_radiance = ground_radiance * transmittance + in_scatter;
            ground_alpha = 1.0;  // 지표면은 완전 불투명
        }
    }

    // ========================================================================
    // 하늘 복사휘도 계산 (Sky Radiance) - 논문 Section 4
    // ========================================================================
    // 시선 방향으로 무한히 먼 점까지의 산란광을 계산합니다.
    // 이것이 하늘의 기본 색상이 됩니다.
    //
    // GetSkyRadiance() 함수:
    // - 카메라 위치에서 view_direction 방향으로 레이마칭
    // - 경로상의 Rayleigh + Mie 산란 적분
    // - 사전 계산된 텍스처에서 빠르게 조회
    //
    // 반환값:
    // - radiance: 하늘에서 오는 총 복사휘도 (산란광)
    // - transmittance: 이 방향의 대기 투과율 (태양 디스크 감쇠에 사용)

    float shadow_length = max(0.0, shadow_out - shadow_in) * lightshaft_fadein_hack * kLengthUnitInMeters;
    float3 transmittance;
    float3 radiance = GetSkyRadiance(
        atmosphere,
        transmittance_texture,
        scattering_texture,
        single_mie_scattering_texture,
        textureSampler,
        (uniforms.camera - uniforms.earth_center) * kLengthUnitInMeters,  // 미터 변환
        view_direction,
        shadow_length,
        uniforms.sun_direction,
        true,  // use_combined_textures: Mie가 alpha 채널에 저장됨
        transmittance);

    // ========================================================================
    // 태양 원반 (Sun Disc) 렌더링
    // ========================================================================
    // 시선 방향이 태양 방향과 일치하면 태양 복사를 추가합니다.
    //
    // 조건: dot(view, sun) > cos(sun_angular_radius)
    //   → 시선과 태양 방향의 각도 < 태양 각반경 (~0.26°)
    //
    // 태양 복사량:
    //   L_sun = E_sun / (π × θ_sun²)   (논문 참고)
    //
    // 대기 투과율을 곱해 감쇠된 태양빛 표현 (일출/일몰 시 붉은 태양)
    if (dot(view_direction, uniforms.sun_direction) > uniforms.sun_size.y) {
        radiance = radiance + transmittance * GetSolarRadiance(atmosphere);
    }

    // ========================================================================
    // 레이어 합성 (Compositing - Back to Front)
    // ========================================================================
    // 페인터 알고리즘: 뒤에서 앞으로 레이어를 합성합니다.
    //
    // 합성 순서:
    // 1. 하늘 (가장 멀리) - 기본 배경
    // 2. 지표면 - 하늘 위에 덮어씌움
    // 3. 구체 (가장 가까이) - 모든 것 위에 덮어씌움
    //
    // mix(a, b, t) = a × (1-t) + b × t
    // - alpha = 0: 뒤의 레이어가 보임 (하늘)
    // - alpha = 1: 앞의 레이어가 보임 (지표면/구체)

    radiance = mix(radiance, ground_radiance, ground_alpha);  // 하늘 + 지표면
    radiance = mix(radiance, sphere_radiance, sphere_alpha);  // + 구체

    // ========================================================================
    // 톤 매핑 (Tone Mapping)
    // ========================================================================
    // HDR → LDR 변환: 물리적으로 정확한 복사휘도 값(HDR)을
    // 모니터가 표시할 수 있는 0~1 범위(LDR)로 변환합니다.
    //
    // 사용된 공식: 지수 톤매핑 (Exponential Tone Mapping)
    //   color = 1 - exp(-L × exposure / white_point)
    //
    // 특징:
    // - 어두운 부분: 거의 선형 (0 근처)
    // - 밝은 부분: 점진적 압축 (1에 수렴)
    // - exposure: 전체 밝기 조절
    // - white_point: 순백으로 매핑되는 복사휘도 값
    //
    // 감마 보정: sRGB 모니터용 (γ = 2.2)
    //   선형 → sRGB: pow(color, 1/2.2)

    float3 color = float3(1.0) - exp(-radiance / uniforms.white_point * uniforms.exposure);
    color = pow(color, float3(1.0 / 2.2));  // 감마 보정

    // ========================================================================
    // 디더링 (Dithering) - 8비트 밴딩 방지
    // ========================================================================
    // 하늘 그라데이션에서 8비트 색상 양자화로 인한 밴딩을 줄입니다.
    //
    // 삼각 분포(Triangular Distribution) 디더링:
    // - 균일 분포보다 시각적으로 자연스러움
    // - 범위: ±0.5 LSB (8비트 기준 ±1/255)
    //
    // 알고리즘:
    // 1. 화면 위치 기반 의사 난수 생성
    // 2. 균일 분포 → 삼각 분포 변환
    // 3. ±0.5/255 범위로 스케일링 후 색상에 추가
    //
    // 원리: 양자화 오차를 무작위화하여 규칙적인 밴딩 대신
    //       덜 눈에 띄는 노이즈로 분산시킴
    // -> 구현 결과 디더링 여부가 큰 차이를 발생시키지 않아 주석 처리합니다.
    //    float2 screenPos = in.position.xy;
    //    float dither = fract(dot(screenPos, float2(0.06711056, 0.00583715))) * 2.0 - 1.0;
    //    dither = sign(dither) * (1.0 - sqrt(1.0 - abs(dither)));  // 삼각 분포 변환
    //    color += dither / 255.0;  // ±0.5 LSB (8비트)

    out.color = float4(color, 1.0);
    return out;
}

// ============================================================================
// 대체 버텍스 셰이더 (버텍스 버퍼 사용 버전)
// ============================================================================
// 위의 atmosphereVertexShader와 동일한 기능이지만,
// vertexID 대신 버텍스 버퍼에서 정점 데이터를 읽습니다.
//
// 차이점:
// 1. 정점 위치가 버퍼에서 전달됨 (SimpleVertexIn)
// 2. Y축 플립 처리 (Metal vs OpenGL 좌표계 차이)
//
// OpenGL vs Metal 좌표계:
// - OpenGL: 화면 좌하단이 원점, Y축 위로
// - Metal: 화면 좌상단이 원점, Y축 아래로
// 따라서 Y를 뒤집어 OpenGL 스타일 셰이더 로직과 호환

struct SimpleVertexIn {
    float4 position [[attribute(0)]];  // 버텍스 버퍼에서 읽은 클립 공간 위치
};

vertex AtmosphereVertexOut atmosphereVertexShaderWithBuffer(
    SimpleVertexIn in [[stage_in]],
    constant AtmosphereUniforms &uniforms [[buffer(BufferIndexUniforms)]])
{
    AtmosphereVertexOut out;
    out.position = in.position;  // 클립 공간 위치 그대로 전달

    // Y축 플립: Metal 좌표계 → OpenGL 스타일
    // 이렇게 하면 원본 GLSL 셰이더 로직을 그대로 사용 가능
    float4 flippedPos = float4(in.position.x, -in.position.y, in.position.z, in.position.w);

    // 클립 → 뷰 → 월드 변환으로 시선 방향 계산
    float4 viewPos = uniforms.view_from_clip * flippedPos;
    float3 viewDir = viewPos.xyz;

    // 카메라(뷰) 공간 → 월드 공간
    out.view_ray = (uniforms.model_from_view * float4(viewDir, 0.0)).xyz;

    return out;
}
