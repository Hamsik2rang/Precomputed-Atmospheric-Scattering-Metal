/**
 * Atmosphere renderer implementation for Metal.
 */

#import "AtmosphereRenderer.h"
#import "AtmosphereConstants.h"

static const double kPi = 3.14159265358979323846;
static const double kSunAngularRadius = 0.00935 / 2.0;
static const double kLengthUnitInMeters = 1000.0;
static const double kBottomRadius = 6360000.0;

@implementation AtmosphereRenderer {
    // Precomputation
    AtmospherePrecomputation *_precomputation;

    // Render pipeline
    id<MTLRenderPipelineState> _renderPipelineState;
    id<MTLDepthStencilState> _depthStencilState;
    id<MTLSamplerState> _samplerState;

    // Vertex buffer for fullscreen quad
    id<MTLBuffer> _quadVertexBuffer;

    // Uniform buffers
    id<MTLBuffer> _atmosphereParamsBuffer;
    id<MTLBuffer> _uniformsBuffer;

    // Viewport size
    vector_uint2 _viewportSize;

    // Internal state
    BOOL _isPrecomputed;
}

#pragma mark - Initialization

- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView {
    self = [super init];
    if (self) {
        _device = mtkView.device;
        _commandQueue = [_device newCommandQueue];

        // Set default view parameters (orbit around sphere)
        _viewDistanceMeters = 8000.0;  // 8km orbit radius from sphere
        _viewZenithAngleRadians = 1.3;  // slightly above horizontal
        _viewAzimuthAngleRadians = 0.0;
        _sunZenithAngleRadians = 1.3;
        _sunAzimuthAngleRadians = 2.9;
        _exposure = 10.0;
        _doWhiteBalance = NO;

        _isPrecomputed = NO;

        // Configure view pixel formats BEFORE creating pipeline
        mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm_sRGB;
        mtkView.depthStencilPixelFormat = MTLPixelFormatDepth32Float;

        // Initialize precomputation manager
        _precomputation = [[AtmospherePrecomputation alloc] initWithDevice:_device];
        [_precomputation setupEarthAtmosphere];

        // Setup rendering pipeline (must be after view configuration)
        [self setupRenderPipelineWithView:mtkView];
        [self setupVertexBuffers];
        [self setupUniformBuffers];
    }
    return self;
}

#pragma mark - Setup

- (void)setupRenderPipelineWithView:(MTKView *)mtkView {
    NSError *error = nil;

    // Load shaders
    id<MTLLibrary> library = [_device newDefaultLibrary];

    id<MTLFunction> vertexFunction = [library newFunctionWithName:@"atmosphereVertexShader"];
    id<MTLFunction> fragmentFunction = [library newFunctionWithName:@"atmosphereFragmentShader"];

    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"Error: Could not find atmosphere shader functions");
        return;
    }

    // Create render pipeline descriptor
    MTLRenderPipelineDescriptor *pipelineDescriptor = [[MTLRenderPipelineDescriptor alloc] init];
    pipelineDescriptor.label = @"Atmosphere Render Pipeline";
    pipelineDescriptor.vertexFunction = vertexFunction;
    pipelineDescriptor.fragmentFunction = fragmentFunction;
    pipelineDescriptor.colorAttachments[0].pixelFormat = mtkView.colorPixelFormat;
    pipelineDescriptor.depthAttachmentPixelFormat = mtkView.depthStencilPixelFormat;

    _renderPipelineState = [_device newRenderPipelineStateWithDescriptor:pipelineDescriptor
                                                                   error:&error];
    if (!_renderPipelineState) {
        NSLog(@"Error creating render pipeline state: %@", error);
    }

    // Create depth stencil state
    MTLDepthStencilDescriptor *depthDescriptor = [[MTLDepthStencilDescriptor alloc] init];
    depthDescriptor.depthCompareFunction = MTLCompareFunctionLess;
    depthDescriptor.depthWriteEnabled = YES;
    _depthStencilState = [_device newDepthStencilStateWithDescriptor:depthDescriptor];

    // Create sampler state
    MTLSamplerDescriptor *samplerDescriptor = [[MTLSamplerDescriptor alloc] init];
    samplerDescriptor.minFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.magFilter = MTLSamplerMinMagFilterLinear;
    samplerDescriptor.mipFilter = MTLSamplerMipFilterNotMipmapped;
    samplerDescriptor.sAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.tAddressMode = MTLSamplerAddressModeClampToEdge;
    samplerDescriptor.rAddressMode = MTLSamplerAddressModeClampToEdge;
    _samplerState = [_device newSamplerStateWithDescriptor:samplerDescriptor];
}

- (void)setupVertexBuffers {
    // Fullscreen quad vertices (triangle strip)
    static const float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 0.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 0.0f, 1.0f,
    };

    _quadVertexBuffer = [_device newBufferWithBytes:quadVertices
                                             length:sizeof(quadVertices)
                                            options:MTLResourceStorageModeShared];
    _quadVertexBuffer.label = @"Quad Vertex Buffer";
}

- (void)setupUniformBuffers {
    // Atmosphere parameters buffer
    _atmosphereParamsBuffer = [_device newBufferWithLength:sizeof(AtmosphereParameters)
                                                   options:MTLResourceStorageModeShared];
    _atmosphereParamsBuffer.label = @"Atmosphere Parameters Buffer";

    // Copy atmosphere parameters from precomputation
    AtmosphereParameters params = _precomputation.atmosphereParameters;
    memcpy(_atmosphereParamsBuffer.contents,
           &params,
           sizeof(AtmosphereParameters));

    // Uniforms buffer
    _uniformsBuffer = [_device newBufferWithLength:sizeof(AtmosphereUniforms)
                                           options:MTLResourceStorageModeShared];
    _uniformsBuffer.label = @"Uniforms Buffer";
}

#pragma mark - Precomputation

- (void)precomputeAtmosphereWithCompletion:(void (^)(void))completion {
    [_precomputation precomputeWithCommandQueue:_commandQueue completion:^{
        self->_isPrecomputed = YES;
        if (completion) {
            completion();
        }
    }];
}

- (BOOL)isPrecomputed {
    return _isPrecomputed;
}

#pragma mark - MTKViewDelegate

- (void)mtkView:(nonnull MTKView *)view drawableSizeWillChange:(CGSize)size {
    _viewportSize.x = size.width;
    _viewportSize.y = size.height;
}

- (void)drawInMTKView:(nonnull MTKView *)view {
    if (!_isPrecomputed) {
        // Clear to dark blue while waiting for precomputation
        view.clearColor = MTLClearColorMake(0.0, 0.0, 0.1, 1.0);
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
        if (renderPassDescriptor) {
            id<MTLRenderCommandEncoder> encoder =
                [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
            [encoder endEncoding];
            [commandBuffer presentDrawable:view.currentDrawable];
        }
        [commandBuffer commit];
        return;
    }

    // Update uniforms
    [self updateUniforms];

    // Create command buffer
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    commandBuffer.label = @"Atmosphere Render Command";

    // Get render pass descriptor
    MTLRenderPassDescriptor *renderPassDescriptor = view.currentRenderPassDescriptor;
    if (renderPassDescriptor == nil) {
        [commandBuffer commit];
        return;
    }

    // Clear color
    renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

    // Create render command encoder
    id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    renderEncoder.label = @"Atmosphere Render Encoder";

    // Set viewport
    [renderEncoder setViewport:(MTLViewport){
        0.0, 0.0,
        (double)_viewportSize.x, (double)_viewportSize.y,
        0.0, 1.0
    }];

    // Set pipeline state
    [renderEncoder setRenderPipelineState:_renderPipelineState];
    [renderEncoder setDepthStencilState:_depthStencilState];

    // Set buffers
    [renderEncoder setVertexBuffer:_uniformsBuffer
                            offset:0
                           atIndex:BufferIndexUniforms];

    [renderEncoder setFragmentBuffer:_atmosphereParamsBuffer
                              offset:0
                             atIndex:BufferIndexAtmosphere];

    [renderEncoder setFragmentBuffer:_uniformsBuffer
                              offset:0
                             atIndex:BufferIndexUniforms];

    // Set textures
    [renderEncoder setFragmentTexture:_precomputation.transmittanceTexture
                              atIndex:TextureIndexTransmittance];
    [renderEncoder setFragmentTexture:_precomputation.scatteringTexture
                              atIndex:TextureIndexScattering];
    [renderEncoder setFragmentTexture:_precomputation.irradianceTexture
                              atIndex:TextureIndexIrradiance];

    // Set optional single Mie scattering texture
    if (_precomputation.optionalSingleMieScatteringTexture) {
        [renderEncoder setFragmentTexture:_precomputation.optionalSingleMieScatteringTexture
                                  atIndex:TextureIndexSingleMieScattering];
    } else {
        [renderEncoder setFragmentTexture:_precomputation.scatteringTexture
                                  atIndex:TextureIndexSingleMieScattering];
    }

    // Set sampler
    [renderEncoder setFragmentSamplerState:_samplerState atIndex:0];

    // Draw fullscreen quad
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangleStrip
                      vertexStart:0
                      vertexCount:4];

    [renderEncoder endEncoding];

    // Present and commit
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
}

#pragma mark - Uniform Updates

- (void)updateUniforms {
    AtmosphereUniforms *uniforms = (AtmosphereUniforms *)_uniformsBuffer.contents;

    // Sphere center (demo sphere at 1km altitude)
    static const double kSphereCenterAltitude = 1000.0;  // meters
    simd_float3 sphereCenter = simd_make_float3(0.0, 0.0, kSphereCenterAltitude / kLengthUnitInMeters);

    // Compute camera position orbiting around sphere
    double cos_z = cos(_viewZenithAngleRadians);
    double sin_z = sin(_viewZenithAngleRadians);
    double cos_a = cos(_viewAzimuthAngleRadians);
    double sin_a = sin(_viewAzimuthAngleRadians);

    // Camera offset from sphere center (spherical coordinates)
    double orbitRadius = _viewDistanceMeters / kLengthUnitInMeters;
    simd_float3 cameraOffset = simd_make_float3(
        sin_z * cos_a * orbitRadius,
        sin_z * sin_a * orbitRadius,
        cos_z * orbitRadius
    );

    // Camera position = sphere center + offset
    uniforms->camera = sphereCenter + cameraOffset;

    // Camera looks at sphere center
    simd_float3 lookDir = simd_normalize(simd_make_float3(-cameraOffset.x, -cameraOffset.y, -cameraOffset.z));

    // Build camera frame (uz = forward, ux = right, uy = up)
    simd_float3 worldUp = simd_make_float3(0.0, 0.0, 1.0);
    simd_float3 uz = lookDir;  // forward (towards sphere)

    // Handle case when looking straight up or down
    simd_float3 ux;
    if (fabs(simd_dot(uz, worldUp)) > 0.999) {
        // Use alternative up vector when looking along z-axis
        simd_float3 altUp = simd_make_float3(0.0, 1.0, 0.0);
        ux = simd_normalize(simd_cross(altUp, uz));
    } else {
        ux = simd_normalize(simd_cross(worldUp, uz));
    }
    simd_float3 uy = simd_cross(uz, ux);  // up = forward × right (right-handed convention)

    // Earth center (at negative bottom radius in z)
    uniforms->earth_center = simd_make_float3(0.0, 0.0, -kBottomRadius / kLengthUnitInMeters);

    // Sun direction
    uniforms->sun_direction = simd_make_float3(
        cos(_sunAzimuthAngleRadians) * sin(_sunZenithAngleRadians),
        sin(_sunAzimuthAngleRadians) * sin(_sunZenithAngleRadians),
        cos(_sunZenithAngleRadians)
    );

    // Sun size (tangent and cosine of angular radius)
    uniforms->sun_size = simd_make_float2(tan(kSunAngularRadius), cos(kSunAngularRadius));

    // Exposure
    uniforms->exposure = _exposure;

    // White point (for color correction)
    if (_doWhiteBalance) {
        // TODO: Implement proper white balance calculation
        uniforms->white_point = simd_make_float3(1.0, 1.0, 1.0);
    }
    else {
        uniforms->white_point = simd_make_float3(1.0, 1.0, 1.0);
    }

    // Model from view matrix (camera to world transform)
    // Column-major: each simd_make_float4 is a column
    uniforms->model_from_view = simd_matrix(
        simd_make_float4(ux.x, ux.y, ux.z, 0.0),
        simd_make_float4(uy.x, uy.y, uy.z, 0.0),
        simd_make_float4(uz.x, uz.y, uz.z, 0.0),
        simd_make_float4(uniforms->camera.x, uniforms->camera.y, uniforms->camera.z, 1.0)
    );

    // View from clip matrix (projection inverse)
    float kFovY = 50.0 / 180.0 * kPi;
    float kTanFovY = tan(kFovY / 2.0);
    float aspectRatio = (float)_viewportSize.x / (float)_viewportSize.y;

    uniforms->view_from_clip = simd_matrix(
        simd_make_float4(kTanFovY * aspectRatio, 0.0, 0.0, 0.0),
        simd_make_float4(0.0, kTanFovY, 0.0, 0.0),
        simd_make_float4(0.0, 0.0, 0.0, -1.0),
        simd_make_float4(0.0, 0.0, 1.0, 1.0)
    );
}

#pragma mark - Camera Control

- (void)setViewDistance:(double)meters
            zenithAngle:(double)zenith
           azimuthAngle:(double)azimuth {
    _viewDistanceMeters = meters;
    _viewZenithAngleRadians = zenith;
    _viewAzimuthAngleRadians = azimuth;
}

- (void)setSunZenithAngle:(double)zenith azimuthAngle:(double)azimuth {
    _sunZenithAngleRadians = zenith;
    _sunAzimuthAngleRadians = azimuth;
}

- (void)setPresetView:(int)viewIndex {
    switch (viewIndex) {
        case 1:
            [self setViewDistance:9000.0 zenithAngle:1.47 azimuthAngle:0.0];
            [self setSunZenithAngle:1.3 azimuthAngle:3.0];
            _exposure = 10.0;
            break;
        case 2:
            [self setViewDistance:9000.0 zenithAngle:1.47 azimuthAngle:0.0];
            [self setSunZenithAngle:1.564 azimuthAngle:-3.0];
            _exposure = 10.0;
            break;
        case 3:
            [self setViewDistance:7000.0 zenithAngle:1.57 azimuthAngle:0.0];
            [self setSunZenithAngle:1.54 azimuthAngle:-2.96];
            _exposure = 10.0;
            break;
        case 4:
            [self setViewDistance:7000.0 zenithAngle:1.57 azimuthAngle:0.0];
            [self setSunZenithAngle:1.328 azimuthAngle:-3.044];
            _exposure = 10.0;
            break;
        case 5:
            [self setViewDistance:9000.0 zenithAngle:1.39 azimuthAngle:0.0];
            [self setSunZenithAngle:1.2 azimuthAngle:0.7];
            _exposure = 10.0;
            break;
        case 6:
            [self setViewDistance:9000.0 zenithAngle:1.5 azimuthAngle:0.0];
            [self setSunZenithAngle:1.628 azimuthAngle:1.05];
            _exposure = 200.0;
            break;
        case 7:
            [self setViewDistance:7000.0 zenithAngle:1.43 azimuthAngle:0.0];
            [self setSunZenithAngle:1.57 azimuthAngle:1.34];
            _exposure = 40.0;
            break;
        case 8:
            [self setViewDistance:2.7e6 zenithAngle:0.81 azimuthAngle:0.0];
            [self setSunZenithAngle:1.57 azimuthAngle:2.0];
            _exposure = 10.0;
            break;
        case 9:
            [self setViewDistance:1.2e7 zenithAngle:0.0 azimuthAngle:0.0];
            [self setSunZenithAngle:0.93 azimuthAngle:-2.0];
            _exposure = 10.0;
            break;
        default:
            break;
    }
}

#pragma mark - Input Handling

- (void)handleMouseDragDeltaX:(float)deltaX
                       deltaY:(float)deltaY
                 withModifier:(BOOL)isCtrlPressed {
    const double kScale = 500.0;

    if (isCtrlPressed) {
        // Control sun direction (invert Y for intuitive control: drag up = sun up)
        _sunZenithAngleRadians -= deltaY / kScale;  // 부호 반전
        _sunZenithAngleRadians = fmax(0.0, fmin(kPi, _sunZenithAngleRadians));
        _sunAzimuthAngleRadians += deltaX / kScale;
    } else {
        // Control camera direction (compensate for Y-flip in shader)
        // Mouse down = look up = sphere moves up on screen
        _viewZenithAngleRadians -= deltaY / kScale;
        _viewZenithAngleRadians = fmax(0.0, fmin(kPi / 2.0, _viewZenithAngleRadians));
        _viewAzimuthAngleRadians -= deltaX / kScale;
    }
}

- (void)handleScrollDelta:(float)delta {
    if (delta < 0) {
        _viewDistanceMeters *= 1.05;
    } else {
        _viewDistanceMeters /= 1.05;
    }
}

@end
