/*
See the LICENSE.txt file for this sample's licensing information.

Abstract:
Implementation of our cross-platform view controller
*/

#import "AAPLViewController.h"
#import "AtmosphereRenderer.h"

@implementation AAPLViewController
{
    MTKView *_view;
    AtmosphereRenderer *_renderer;

    // Mouse tracking
    NSPoint _previousMouseLocation;
    BOOL _isDragging;
}

- (void)viewDidLoad
{
    [super viewDidLoad];

    // Set the view to use the default device
    _view = (MTKView *)self.view;

    _view.device = MTLCreateSystemDefaultDevice();

    NSAssert(_view.device, @"Metal is not supported on this device");

    // Create atmosphere renderer
    _renderer = [[AtmosphereRenderer alloc] initWithMetalKitView:_view];

    NSAssert(_renderer, @"Renderer failed initialization");

    // Initialize our renderer with the view size
    [_renderer mtkView:_view drawableSizeWillChange:_view.drawableSize];

    _view.delegate = _renderer;

    // Start atmosphere precomputation
    NSLog(@"Starting atmosphere precomputation...");
    [_renderer precomputeAtmosphereWithCompletion:^{
        NSLog(@"Atmosphere precomputation complete!");
    }];

    // Enable mouse tracking
    _isDragging = NO;

#if !defined(TARGET_IOS) && !defined(TARGET_TVOS)
    // macOS: Enable key events
    [self.view.window makeFirstResponder:self];
#endif
}

#if !defined(TARGET_IOS) && !defined(TARGET_TVOS)
// ============================================================================
// macOS Input Handling
// ============================================================================

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)keyDown:(NSEvent *)event {
    unichar key = [[event characters] characterAtIndex:0];

    switch (key) {
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            [_renderer setPresetView:key - '0'];
            break;

        case '+':
        case '=':
            _renderer.exposure *= 1.1;
            break;

        case '-':
        case '_':
            _renderer.exposure /= 1.1;
            break;

        case 'w':
        case 'W':
            _renderer.doWhiteBalance = !_renderer.doWhiteBalance;
            break;

        case 27: // Escape
            [NSApp terminate:nil];
            break;

        default:
            [super keyDown:event];
            break;
    }
}

- (void)mouseDown:(NSEvent *)event {
    _previousMouseLocation = [event locationInWindow];
    _isDragging = YES;
}

- (void)mouseUp:(NSEvent *)event {
    _isDragging = NO;
}

- (void)mouseDragged:(NSEvent *)event {
    if (!_isDragging) return;

    NSPoint currentLocation = [event locationInWindow];
    float deltaX = currentLocation.x - _previousMouseLocation.x;
    float deltaY = currentLocation.y - _previousMouseLocation.y;

    BOOL isCtrlPressed = ([event modifierFlags] & NSEventModifierFlagControl) != 0;

    [_renderer handleMouseDragDeltaX:deltaX
                              deltaY:deltaY
                        withModifier:isCtrlPressed];

    _previousMouseLocation = currentLocation;
}

- (void)scrollWheel:(NSEvent *)event {
    float delta = event.deltaY;
    if (fabs(delta) > 0.01) {
        [_renderer handleScrollDelta:delta];
    }
}

- (void)rightMouseDown:(NSEvent *)event {
    _previousMouseLocation = [event locationInWindow];
    _isDragging = YES;
}

- (void)rightMouseUp:(NSEvent *)event {
    _isDragging = NO;
}

- (void)rightMouseDragged:(NSEvent *)event {
    if (!_isDragging) return;

    NSPoint currentLocation = [event locationInWindow];
    float deltaX = currentLocation.x - _previousMouseLocation.x;
    float deltaY = currentLocation.y - _previousMouseLocation.y;

    // Right mouse drag controls sun direction
    [_renderer handleMouseDragDeltaX:deltaX
                              deltaY:deltaY
                        withModifier:YES];

    _previousMouseLocation = currentLocation;
}

#else
// ============================================================================
// iOS/tvOS Input Handling (Touch)
// ============================================================================

- (void)touchesBegan:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    UITouch *touch = [touches anyObject];
    _previousMouseLocation = [touch locationInView:self.view];
    _isDragging = YES;
}

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    _isDragging = NO;
}

- (void)touchesMoved:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
    if (!_isDragging) return;

    UITouch *touch = [touches anyObject];
    CGPoint currentLocation = [touch locationInView:self.view];

    float deltaX = currentLocation.x - _previousMouseLocation.x;
    float deltaY = currentLocation.y - _previousMouseLocation.y;

    // Two-finger touch controls sun
    BOOL isTwoFingerTouch = touches.count >= 2;

    [_renderer handleMouseDragDeltaX:deltaX
                              deltaY:-deltaY  // Invert Y for natural touch
                        withModifier:isTwoFingerTouch];

    _previousMouseLocation = currentLocation;
}

#endif

@end
