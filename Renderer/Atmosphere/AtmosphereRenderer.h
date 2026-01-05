/**
 * Atmosphere renderer for Metal.
 * Handles precomputation and real-time rendering of atmospheric scattering.
 */

#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <simd/simd.h>

#import "AtmosphereShaderTypes.h"
#import "AtmospherePrecomputation.h"

NS_ASSUME_NONNULL_BEGIN

@interface AtmosphereRenderer : NSObject <MTKViewDelegate>

// Device and state
@property (nonatomic, strong, readonly) id<MTLDevice> device;
@property (nonatomic, strong, readonly) id<MTLCommandQueue> commandQueue;
@property (nonatomic, assign, readonly) BOOL isPrecomputed;

// Camera parameters
@property (nonatomic, assign) double viewDistanceMeters;
@property (nonatomic, assign) double viewZenithAngleRadians;
@property (nonatomic, assign) double viewAzimuthAngleRadians;

// Sun parameters
@property (nonatomic, assign) double sunZenithAngleRadians;
@property (nonatomic, assign) double sunAzimuthAngleRadians;

// Rendering parameters
@property (nonatomic, assign) double exposure;
@property (nonatomic, assign) BOOL doWhiteBalance;

// Initialization
- (nonnull instancetype)initWithMetalKitView:(nonnull MTKView *)mtkView;

// Precomputation
- (void)precomputeAtmosphereWithCompletion:(void (^_Nullable)(void))completion;

// Camera control
- (void)setViewDistance:(double)meters
            zenithAngle:(double)zenith
           azimuthAngle:(double)azimuth;

// Sun control
- (void)setSunZenithAngle:(double)zenith
             azimuthAngle:(double)azimuth;

// Predefined views (matching original demo)
- (void)setPresetView:(int)viewIndex;

// Input handling
- (void)handleMouseDragDeltaX:(float)deltaX
                       deltaY:(float)deltaY
                 withModifier:(BOOL)isCtrlPressed;
- (void)handleScrollDelta:(float)delta;

@end

NS_ASSUME_NONNULL_END
