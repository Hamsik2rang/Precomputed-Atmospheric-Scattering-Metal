/**
 * ============================================================================
 * 대기 산란 사전 계산 (Atmosphere Precomputation) - 논문 Section 4
 * ============================================================================
 * ■ 사전 계산의 목적
 * ------------------
 * 실시간 렌더링에서 대기 산란을 매 프레임 적분하는 것은 너무 느립니다.
 * 따라서 자주 사용되는 값들을 미리 텍스처에 저장해두고 빠르게 조회합니다.
 *
 * ■ 사전 계산되는 텍스처 (논문 Section 4)
 * ---------------------------------------
 * 1. Transmittance (투과율) - 2D 텍스처 [256×64]
 *    T(r, μ) = exp(-∫ σ dx)  (Equation 1)
 *    - r: 관측점 고도 (bottom_radius ~ top_radius)
 *    - μ: cos(zenith angle) = 시선과 천정의 각도 코사인
 *
 * 2. Scattering (산란) - 3D 텍스처 [256×128×32]
 *    4D 함수 (r, μ, μs, ν)를 3D로 압축:
 *    - r, μ: 관측점 위치 (투과율과 동일)
 *    - μs: cos(sun_zenith) = 태양 천정각 코사인
 *    - ν: cos(view, sun) = 시선-태양 사잇각 코사인
 *
 * 3. Irradiance (복사조도) - 2D 텍스처 [64×16]
 *    E(r, μs) = ∫ L cos(θ) dω  (Equation 5)
 *    - 지표면에 도달하는 하늘빛의 총량
 *
 * ■ 다중 산란 계산 (논문 Section 3.3)
 * ------------------------------------
 * 빛은 대기 중에서 여러 번 산란됩니다:
 * - 1차 산란: 태양 → 대기 입자 → 관측점
 * - 2차 산란: 태양 → 입자A → 입자B → 관측점
 * - n차 산란: 점점 약해지므로 보통 4차까지 계산
 *
 * 계산 순서:
 * 1. Transmittance 계산
 * 2. Direct Irradiance 계산 (직사광)
 * 3. Single Scattering 계산 (1차 산란)
 * 4. 반복 (order = 2 ~ numScatteringOrders):
 *    a. Scattering Density 계산
 *    b. Indirect Irradiance 누적
 *    c. Multiple Scattering 누적
 *
 * ■ GPU 컴퓨트 셰이더 활용
 * ------------------------
 * 텍스처의 각 텍셀을 독립적으로 계산할 수 있으므로
 * Metal Compute Shader로 병렬 처리하여 빠르게 계산합니다.
 */

#import "AtmospherePrecomputation.h"

@interface AtmospherePrecomputation ()

// ============================================================================
// Metal 리소스
// ============================================================================
@property (nonatomic, strong, readwrite) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLLibrary> library;

// ============================================================================
// 컴퓨트 파이프라인 상태 (Compute Pipeline States)
// ============================================================================
// 각 사전 계산 단계에 대응하는 컴퓨트 셰이더 파이프라인입니다.
// AtmosphereCompute.metal의 커널 함수들을 사용합니다.

@property (nonatomic, strong) id<MTLComputePipelineState> computeTransmittancePipeline;        // 투과율
@property (nonatomic, strong) id<MTLComputePipelineState> computeDirectIrradiancePipeline;    // 직사 복사조도
@property (nonatomic, strong) id<MTLComputePipelineState> computeSingleScatteringPipeline;   // 1차 산란
@property (nonatomic, strong) id<MTLComputePipelineState> computeScatteringDensityPipeline;  // 산란 밀도
@property (nonatomic, strong) id<MTLComputePipelineState> computeIndirectIrradiancePipeline; // 간접 복사조도
@property (nonatomic, strong) id<MTLComputePipelineState> computeMultipleScatteringPipeline; // 다중 산란

// ============================================================================
// 최종 출력 텍스처 (Final Output Textures)
// ============================================================================
// 렌더링 시 셰이더에서 읽을 사전 계산된 텍스처들입니다.

// 투과율: T(r, μ) - 2D [256×64]
@property (nonatomic, strong, readwrite, nullable) id<MTLTexture> transmittanceTexture;

// 산란: S(r, μ, μs, ν) - 3D [256×128×32] ( == [8x32x128x32]의 패킹)
// Rayleigh + Mie 산란이 합쳐진 텍스처 (useCombinedTextures=YES인 경우 Mie는 alpha에)
@property (nonatomic, strong, readwrite, nullable) id<MTLTexture> scatteringTexture;

// 복사조도: E(r, μs) - 2D [64×16]
@property (nonatomic, strong, readwrite, nullable) id<MTLTexture> irradianceTexture;

// 선택적 Mie 산란 텍스처 (useCombinedTextures=NO인 경우에만 사용)
@property (nonatomic, strong, readwrite, nullable) id<MTLTexture> optionalSingleMieScatteringTexture;

// ============================================================================
// 임시 텍스처 (Temporary/Delta Textures)
// ============================================================================
// 다중 산란 계산 시 중간 결과를 저장하는 텍스처들입니다.

// 이번 반복의 복사조도 변화량
@property (nonatomic, strong) id<MTLTexture> deltaIrradianceTexture;

// 이번 반복의 Rayleigh 산란 (또는 1차 산란의 Rayleigh 성분)
@property (nonatomic, strong) id<MTLTexture> deltaRayleighScatteringTexture;

// 이번 반복의 Mie 산란 (또는 1차 산란의 Mie 성분)
@property (nonatomic, strong) id<MTLTexture> deltaMieScatteringTexture;

// 산란 밀도 (n차 산란을 계산하기 위한 (n-1)차 산란의 기여도)
@property (nonatomic, strong) id<MTLTexture> deltaScatteringDensityTexture;

// 이번 반복의 다중 산란 결과
@property (nonatomic, strong) id<MTLTexture> deltaMultipleScatteringTexture;

// ============================================================================
// 대기 파라미터 및 변환 행렬
// ============================================================================

// 대기 물리 파라미터 (밀도 프로필, 산란/흡수 계수 등)
@property (nonatomic, readwrite) AtmosphereParameters atmosphereParameters;

// 복사휘도 → 휘도 변환 행렬 (현재는 단위행렬 - radiance 모드)
// luminance 모드에서는 CIE XYZ → sRGB 변환 행렬을 사용
@property (nonatomic, strong) id<MTLBuffer> luminanceFromRadianceBuffer;

@end

@implementation AtmospherePrecomputation

// ============================================================================
#pragma mark - 초기화 (Initialization)
// ============================================================================

/**
 * 대기 사전계산 관리자 초기화
 *
 * @param device Metal 디바이스
 *
 * 기본 설정:
 * - useCombinedTextures=YES: Mie 산란을 alpha 채널에 저장 (메모리 절약)
 * - useHalfPrecision=NO: 32비트 부동소수점 사용 (정밀도 우선)
 * - numScatteringOrders=4: 4차 산란까지 계산 (품질과 성능의 균형점)
 */
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _useCombinedTextures = YES;   // Mie를 alpha 채널에 저장
        _useHalfPrecision = NO;       // Float32 사용 (정밀도 우선)
        _numScatteringOrders = 4;     // 4차 산란까지 계산

        [self setupPipelineStates];
    }
    return self;
}

/**
 * 컴퓨트 파이프라인 상태 설정
 *
 * AtmosphereCompute.metal에 정의된 6개의 컴퓨트 커널을 로드합니다:
 * 1. ComputeTransmittance - 투과율 텍스처 계산
 * 2. ComputeDirectIrradiance - 직사광 복사조도
 * 3. ComputeSingleScattering - 1차 산란 (Rayleigh + Mie)
 * 4. ComputeScatteringDensity - n차 산란 밀도
 * 5. ComputeIndirectIrradiance - 간접광 복사조도 누적
 * 6. ComputeMultipleScattering - 다중 산란 누적
 */
- (void)setupPipelineStates {
    NSError *error = nil;

    _library = [_device newDefaultLibrary];
    if (!_library) {
        NSLog(@"Failed to load default Metal library");
        return;
    }

    // 각 사전계산 단계에 대한 컴퓨트 파이프라인 생성
    _computeTransmittancePipeline = [self createPipelineWithFunction:@"ComputeTransmittance" error:&error];
    if (error) NSLog(@"ComputeTransmittance error: %@", error);

    _computeDirectIrradiancePipeline = [self createPipelineWithFunction:@"ComputeDirectIrradiance" error:&error];
    if (error) NSLog(@"ComputeDirectIrradiance error: %@", error);

    _computeSingleScatteringPipeline = [self createPipelineWithFunction:@"ComputeSingleScattering" error:&error];
    if (error) NSLog(@"ComputeSingleScattering error: %@", error);

    _computeScatteringDensityPipeline = [self createPipelineWithFunction:@"ComputeScatteringDensity" error:&error];
    if (error) NSLog(@"ComputeScatteringDensity error: %@", error);

    _computeIndirectIrradiancePipeline = [self createPipelineWithFunction:@"ComputeIndirectIrradiance" error:&error];
    if (error) NSLog(@"ComputeIndirectIrradiance error: %@", error);

    _computeMultipleScatteringPipeline = [self createPipelineWithFunction:@"ComputeMultipleScattering" error:&error];
    if (error) NSLog(@"ComputeMultipleScattering error: %@", error);
}

- (id<MTLComputePipelineState>)createPipelineWithFunction:(NSString *)functionName error:(NSError **)error {
    id<MTLFunction> function = [_library newFunctionWithName:functionName];
    if (!function) {
        NSLog(@"Failed to find function: %@", functionName);
        return nil;
    }
    return [_device newComputePipelineStateWithFunction:function error:error];
}

// ============================================================================
#pragma mark - 텍스처 생성 (Texture Creation)
// ============================================================================

/// 2D 텍스처 생성 (투과율, 복사조도용)
- (id<MTLTexture>)createTexture2DWithWidth:(NSUInteger)width
                                    height:(NSUInteger)height
                                    format:(MTLPixelFormat)format
                                     usage:(MTLTextureUsage)usage {
    MTLTextureDescriptor *desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:NO];
    desc.usage = usage;
    desc.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:desc];
}

/// 3D 텍스처 생성 (산란 텍스처용 - 4D→3D 매핑)
- (id<MTLTexture>)createTexture3DWithWidth:(NSUInteger)width
                                    height:(NSUInteger)height
                                     depth:(NSUInteger)depth
                                    format:(MTLPixelFormat)format
                                     usage:(MTLTextureUsage)usage {
    MTLTextureDescriptor *desc = [[MTLTextureDescriptor alloc] init];
    desc.textureType = MTLTextureType3D;
    desc.pixelFormat = format;
    desc.width = width;
    desc.height = height;
    desc.depth = depth;
    desc.usage = usage;
    desc.storageMode = MTLStorageModePrivate;
    return [_device newTextureWithDescriptor:desc];
}

/**
 * 모든 사전계산 텍스처 생성
 *
 * 논문 Section 4의 텍스처 크기 정의:
 * - Transmittance: 256×64 (r, μ 파라미터화)
 * - Scattering: 256×128×32 (r, μ, μs, ν를 3D로 매핑)
 * - Irradiance: 64×16 (r, μs 파라미터화)
 *
 * 픽셀 포맷:
 * - Half precision (16-bit): 메모리 절약, 약간의 정밀도 손실
 * - Full precision (32-bit): 최고 품질, 메모리 2배 사용
 */
- (void)createTextures {
    // 픽셀 포맷 선택 (half vs full precision)
    MTLPixelFormat format2D = _useHalfPrecision ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA32Float;
    MTLPixelFormat format3D = _useHalfPrecision ? MTLPixelFormatRGBA16Float : MTLPixelFormatRGBA32Float;

    // ---- 최종 출력 텍스처 ----
    _transmittanceTexture = [self createTexture2DWithWidth:TRANSMITTANCE_TEXTURE_WIDTH
                                                    height:TRANSMITTANCE_TEXTURE_HEIGHT
                                                    format:format2D
                                                     usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _scatteringTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                 height:SCATTERING_TEXTURE_HEIGHT
                                                  depth:SCATTERING_TEXTURE_DEPTH
                                                 format:format3D
                                                  usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _irradianceTexture = [self createTexture2DWithWidth:IRRADIANCE_TEXTURE_WIDTH
                                                 height:IRRADIANCE_TEXTURE_HEIGHT
                                                 format:format2D
                                                  usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    // useCombinedTextures=NO일 때만 별도 Mie 텍스처 생성
    if (!_useCombinedTextures) {
        _optionalSingleMieScatteringTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                                      height:SCATTERING_TEXTURE_HEIGHT
                                                                       depth:SCATTERING_TEXTURE_DEPTH
                                                                      format:format3D
                                                                       usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    }

    // ---- 임시(delta) 텍스처: 다중 산란 계산용 ----
    _deltaIrradianceTexture = [self createTexture2DWithWidth:IRRADIANCE_TEXTURE_WIDTH
                                                      height:IRRADIANCE_TEXTURE_HEIGHT
                                                      format:format2D
                                                       usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _deltaRayleighScatteringTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                              height:SCATTERING_TEXTURE_HEIGHT
                                                               depth:SCATTERING_TEXTURE_DEPTH
                                                              format:format3D
                                                               usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _deltaMieScatteringTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                         height:SCATTERING_TEXTURE_HEIGHT
                                                          depth:SCATTERING_TEXTURE_DEPTH
                                                         format:format3D
                                                          usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _deltaScatteringDensityTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                             height:SCATTERING_TEXTURE_HEIGHT
                                                              depth:SCATTERING_TEXTURE_DEPTH
                                                             format:format3D
                                                              usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

    _deltaMultipleScatteringTexture = [self createTexture3DWithWidth:SCATTERING_TEXTURE_WIDTH
                                                              height:SCATTERING_TEXTURE_HEIGHT
                                                               depth:SCATTERING_TEXTURE_DEPTH
                                                              format:format3D
                                                               usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
}

// ============================================================================
#pragma mark - 대기 파라미터 설정 (Atmosphere Setup)
// ============================================================================

/**
 * 지구 대기 파라미터 설정 - 논문 Appendix
 *
 * Eric Bruneton의 모델에 기반한 지구 대기 물리 파라미터입니다.
 * 모든 값은 SI 단위 (미터, 1/미터 등)를 사용합니다.
 *
 * 밀도 프로필 (Density Profile):
 * - 대기 밀도는 고도에 따라 지수적으로 감소합니다
 * - ρ(h) = exp(-h / H) 형태, H는 스케일 하이트
 * - Rayleigh H ≈ 8km, Mie H ≈ 1.2km
 *
 * 오존층:
 * - 고도 25km 근처에 피크가 있는 복잡한 분포
 * - 두 레이어로 근사: 아래층(증가) + 위층(감소)
 */
- (void)setupEarthAtmosphere {
    AtmosphereParameters params;

    // ---- 태양 복사 ----
    // 대기권 상단에서의 태양 복사조도 (W/m²/nm)
    // RGB ≈ 680nm, 550nm, 440nm 파장에서의 값
    params.solar_irradiance = simd_make_float3(1.474, 1.8504, 1.91198);

    // 태양 각반경 (약 0.536° ≈ 0.00935 rad)
    params.sun_angular_radius = 0.00935;

    // ---- 행성 반지름 (미터) ----
    params.bottom_radius = 6360000.0;  // 지표면 = 6,360 km
    params.top_radius = 6420000.0;     // 대기권 상단 = 6,420 km (고도 60km)

    // ---- Rayleigh 산란 (공기 분자) ----
    // 스케일 하이트 H = 8km: ρ(h) = exp(-h/8000)
    DensityProfileLayer rayleigh_layer0 = {0, 0, 0, 0, 0, {0, 0, 0}};
    DensityProfileLayer rayleigh_layer1 = {0, 1.0, -1.0 / 8000.0, 0, 0, {0, 0, 0}};
    params.rayleigh_density.layers[0] = rayleigh_layer0;
    params.rayleigh_density.layers[1] = rayleigh_layer1;
    // 산란 계수 (1/m): 파장 의존적 - 파란색(440nm)이 가장 큼
    params.rayleigh_scattering = simd_make_float3(5.802e-6, 13.558e-6, 33.1e-6);

    // ---- Mie 산란 (에어로졸/먼지) ----
    // 스케일 하이트 H = 1.2km: 지표면 근처에 집중
    DensityProfileLayer mie_layer0 = {0, 0, 0, 0, 0, {0, 0, 0}};
    DensityProfileLayer mie_layer1 = {0, 1.0, -1.0 / 1200.0, 0, 0, {0, 0, 0}};
    params.mie_density.layers[0] = mie_layer0;
    params.mie_density.layers[1] = mie_layer1;
    // Mie는 파장 독립적 (회색빛)
    params.mie_scattering = simd_make_float3(3.996e-6, 3.996e-6, 3.996e-6);
    params.mie_extinction = simd_make_float3(4.44e-6, 4.44e-6, 4.44e-6);
    // g = 0.8: 강한 전방 산란 (햇무리 효과)
    params.mie_phase_function_g = 0.8;

    // ---- 오존 흡수 ----
    // 고도 ~25km에서 피크를 가지는 두 레이어 구성
    DensityProfileLayer absorption_layer0 = {25000.0, 0, 0, 1.0 / 15000.0, -2.0 / 3.0, {0, 0, 0}};
    DensityProfileLayer absorption_layer1 = {0, 0, 0, -1.0 / 15000.0, 8.0 / 3.0, {0, 0, 0}};
    params.absorption_density.layers[0] = absorption_layer0;
    params.absorption_density.layers[1] = absorption_layer1;
    // 오존은 주로 가시광선(특히 파란색)을 흡수
    params.absorption_extinction = simd_make_float3(0.65e-6, 1.881e-6, 0.085e-6);

    // ---- 지표면 ----
    params.ground_albedo = simd_make_float3(0.1, 0.1, 0.1);  // 어두운 회색
    params.mu_s_min = -0.2;  // cos(102°) - 태양이 지평선 아래 12°까지 허용

    [self setupAtmosphereWithParameters:params];
}

/// 대기 파라미터 적용 및 변환 행렬 초기화
- (void)setupAtmosphereWithParameters:(AtmosphereParameters)params {
    _atmosphereParameters = params;

    // 복사휘도→휘도 변환 행렬 (radiance 모드에서는 단위행렬)
    simd_float3x3 identity = matrix_identity_float3x3;
    _luminanceFromRadianceBuffer = [_device newBufferWithBytes:&identity
                                                        length:sizeof(simd_float3x3)
                                                       options:MTLResourceStorageModeShared];
}

// ============================================================================
#pragma mark - 사전 계산 실행 (Precomputation)
// ============================================================================

/**
 * 대기 산란 사전 계산 실행 - 논문 Section 4
 *
 * GPU 컴퓨트 셰이더를 사용하여 모든 LUT 텍스처를 계산합니다.
 * 계산 순서는 논문의 알고리즘을 따릅니다:
 *
 * 1. Transmittance (투과율) - Equation 1
 *    T(r,μ) = exp(-∫σ dx)
 *
 * 2. Direct Irradiance (직사 복사조도)
 *    태양에서 직접 오는 빛의 지표면 도달량
 *
 * 3. Single Scattering (1차 산란) - Equation 6
 *    태양 → 대기입자 → 관측점 경로의 산란
 *
 * 4. Multiple Scattering 반복 (order = 2 ~ numScatteringOrders):
 *    a. Scattering Density: n차 산란의 소스 계산
 *    b. Indirect Irradiance: n차 산란에 의한 지표면 조도 누적
 *    c. Multiple Scattering: n차 산란 결과를 최종 텍스처에 누적
 *
 * @param commandQueue Metal 명령 큐
 * @param completion   완료 콜백 (메인 스레드에서 호출됨)
 */
- (void)precomputeWithCommandQueue:(id<MTLCommandQueue>)commandQueue
                        completion:(void (^)(void))completion {

    [self createTextures];

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // ---- Step 1: 투과율 계산 ----
    // T(r,μ) = exp(-∫ σ dx), 2D 텍스처 [256×64]
    [self computeTransmittanceWithCommandBuffer:commandBuffer];

    // ---- Step 2: 직사 복사조도 계산 ----
    // 태양 → 대기권 상단 → 지표면 경로의 조도
    [self computeDirectIrradianceWithCommandBuffer:commandBuffer];

    // ---- Step 3: 1차 산란 계산 ----
    // Rayleigh + Mie 단일 산란, 3D 텍스처 [256×128×32]
    [self computeSingleScatteringWithCommandBuffer:commandBuffer];

    // ---- Step 4: 다중 산란 반복 (2차 ~ n차) ----
    // 각 반복에서 이전 차수의 산란이 다음 차수의 광원이 됨
    for (int order = 2; order <= _numScatteringOrders; order++) {
        // 4a. 산란 밀도: (n-1)차 산란이 n차 산란의 소스가 됨
        [self computeScatteringDensityWithCommandBuffer:commandBuffer order:order];

        // 4b. 간접 복사조도: n차 산란에 의한 지표면 조도
        [self computeIndirectIrradianceWithCommandBuffer:commandBuffer order:order];

        // 4c. 다중 산란: 결과를 최종 텍스처에 누적
        [self computeMultipleScatteringWithCommandBuffer:commandBuffer];
    }

    // 비동기 완료 처리
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
        if (completion) {
            dispatch_async(dispatch_get_main_queue(), completion);
        }
    }];

    [commandBuffer commit];
}

// ============================================================================
#pragma mark - 개별 컴퓨트 패스 (Individual Compute Passes)
// ============================================================================

/// 투과율 계산 - 논문 Equation 1: T(r,μ) = exp(-∫σ dx)
- (void)computeTransmittanceWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:@"Compute Transmittance"];
    [encoder setComputePipelineState:_computeTransmittancePipeline];

    [encoder setTexture:_transmittanceTexture atIndex:0];  // 출력: 2D [256×64]
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];

    // 2D 디스패치: 각 텍셀이 하나의 (r, μ) 조합
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

/// 직사 복사조도 계산: 태양 → 대기 → 지표면 직접 도달량
- (void)computeDirectIrradianceWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:@"Compute Direct Irradiance"];
    [encoder setComputePipelineState:_computeDirectIrradiancePipeline];

    [encoder setTexture:_deltaIrradianceTexture atIndex:0];  // 출력: delta E
    [encoder setTexture:_irradianceTexture atIndex:1];       // 출력: 누적 E
    [encoder setTexture:_transmittanceTexture atIndex:2];    // 입력: T
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

/// 1차 산란 계산 - 논문 Equation 6: 태양→입자→관측점
- (void)computeSingleScatteringWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:@"Compute Single Scattering"];
    [encoder setComputePipelineState:_computeSingleScatteringPipeline];

    // 출력 텍스처들
    [encoder setTexture:_deltaRayleighScatteringTexture atIndex:0];  // Rayleigh 성분
    [encoder setTexture:_deltaMieScatteringTexture atIndex:1];       // Mie 성분
    [encoder setTexture:_scatteringTexture atIndex:2];               // 최종 합산 (Rayleigh + Mie)
    [encoder setTexture:_optionalSingleMieScatteringTexture ?: _deltaMieScatteringTexture atIndex:3];
    [encoder setTexture:_transmittanceTexture atIndex:4];            // 입력: T
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];
    [encoder setBuffer:_luminanceFromRadianceBuffer offset:0 atIndex:1];

    // 3D 디스패치: 4D 파라미터 (r, μ, μs, ν)를 3D 텍스처로 매핑
    MTLSize threadgroupSize = MTLSizeMake(8, 8, 4);
    MTLSize gridSize = MTLSizeMake(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

/// n차 산란 밀도 계산: (n-1)차 산란이 n차의 광원이 됨
- (void)computeScatteringDensityWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer order:(int)order {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:[NSString stringWithFormat:@"Compute Scattering Density (order %d)", order]];
    [encoder setComputePipelineState:_computeScatteringDensityPipeline];

    [encoder setTexture:_deltaScatteringDensityTexture atIndex:0];   // 출력
    [encoder setTexture:_transmittanceTexture atIndex:1];
    [encoder setTexture:_deltaRayleighScatteringTexture atIndex:2];
    [encoder setTexture:_deltaMieScatteringTexture atIndex:3];
    [encoder setTexture:_deltaMultipleScatteringTexture atIndex:4];
    [encoder setTexture:_deltaIrradianceTexture atIndex:5];
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];
    [encoder setBytes:&order length:sizeof(int) atIndex:1];  // 현재 산란 차수

    MTLSize threadgroupSize = MTLSizeMake(8, 8, 4);
    MTLSize gridSize = MTLSizeMake(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

/// n차 산란에 의한 간접 복사조도 계산 및 누적
- (void)computeIndirectIrradianceWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer order:(int)order {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:[NSString stringWithFormat:@"Compute Indirect Irradiance (order %d)", order]];
    [encoder setComputePipelineState:_computeIndirectIrradiancePipeline];

    [encoder setTexture:_deltaIrradianceTexture atIndex:0];          // 출력: delta E
    [encoder setTexture:_irradianceTexture atIndex:1];               // 누적: E += delta E
    [encoder setTexture:_deltaRayleighScatteringTexture atIndex:2];
    [encoder setTexture:_deltaMieScatteringTexture atIndex:3];
    [encoder setTexture:_deltaMultipleScatteringTexture atIndex:4];
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];
    [encoder setBuffer:_luminanceFromRadianceBuffer offset:0 atIndex:1];

    int scattering_order = order - 1;  // 산란 차수 (order는 반복 인덱스)
    [encoder setBytes:&scattering_order length:sizeof(int) atIndex:2];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

/// 다중 산란 결과 계산 및 최종 텍스처에 누적
- (void)computeMultipleScatteringWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setLabel:@"Compute Multiple Scattering"];
    [encoder setComputePipelineState:_computeMultipleScatteringPipeline];

    [encoder setTexture:_deltaMultipleScatteringTexture atIndex:0];  // 출력: 이번 차수 결과
    [encoder setTexture:_scatteringTexture atIndex:1];               // 누적: S += delta S
    [encoder setTexture:_transmittanceTexture atIndex:2];
    [encoder setTexture:_deltaScatteringDensityTexture atIndex:3];   // 입력: 산란 밀도
    [encoder setBytes:&_atmosphereParameters length:sizeof(AtmosphereParameters) atIndex:0];
    [encoder setBuffer:_luminanceFromRadianceBuffer offset:0 atIndex:1];

    MTLSize threadgroupSize = MTLSizeMake(8, 8, 4);
    MTLSize gridSize = MTLSizeMake(SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
}

// ============================================================================
#pragma mark - 유틸리티 (Utilities)
// ============================================================================

/// 텍스처 샘플러 생성
/// 선형 보간 + 에지 클램핑으로 부드러운 LUT 조회 지원
- (id<MTLSamplerState>)createSampler {
    MTLSamplerDescriptor *desc = [[MTLSamplerDescriptor alloc] init];
    desc.minFilter = MTLSamplerMinMagFilterLinear;   // 축소 시 선형 보간
    desc.magFilter = MTLSamplerMinMagFilterLinear;   // 확대 시 선형 보간
    desc.sAddressMode = MTLSamplerAddressModeClampToEdge;  // U 축 클램핑
    desc.tAddressMode = MTLSamplerAddressModeClampToEdge;  // V 축 클램핑
    desc.rAddressMode = MTLSamplerAddressModeClampToEdge;  // W 축 클램핑 (3D용)
    return [_device newSamplerStateWithDescriptor:desc];
}

@end
