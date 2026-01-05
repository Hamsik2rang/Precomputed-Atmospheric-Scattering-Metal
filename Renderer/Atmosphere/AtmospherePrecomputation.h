/**
 * Atmosphere precomputation manager for Metal.
 */

#import <Metal/Metal.h>
#import <simd/simd.h>

#import "AtmosphereShaderTypes.h"
#import "AtmosphereConstants.h"

NS_ASSUME_NONNULL_BEGIN

@interface AtmospherePrecomputation : NSObject

// Device and pipeline states
@property (nonatomic, strong, readonly) id<MTLDevice> device;

// Precomputed textures (output)
@property (nonatomic, strong, readonly, nullable) id<MTLTexture> transmittanceTexture;
@property (nonatomic, strong, readonly, nullable) id<MTLTexture> scatteringTexture;
@property (nonatomic, strong, readonly, nullable) id<MTLTexture> irradianceTexture;
@property (nonatomic, strong, readonly, nullable) id<MTLTexture> optionalSingleMieScatteringTexture;

// Atmosphere parameters
@property (nonatomic, readonly) AtmosphereParameters atmosphereParameters;

// Configuration
@property (nonatomic, assign) BOOL useCombinedTextures;
@property (nonatomic, assign) BOOL useHalfPrecision;
@property (nonatomic, assign) int numScatteringOrders;

// Initialization
- (instancetype)initWithDevice:(id<MTLDevice>)device;

// Setup atmosphere with Earth parameters
- (void)setupEarthAtmosphere;

// Setup atmosphere with custom parameters
- (void)setupAtmosphereWithParameters:(AtmosphereParameters)params;

// Precompute all textures
- (void)precomputeWithCommandQueue:(id<MTLCommandQueue>)commandQueue
                        completion:(void (^_Nullable)(void))completion;

// Get sampler for texture lookups
- (id<MTLSamplerState>)createSampler;

@end

NS_ASSUME_NONNULL_END
