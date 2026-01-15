// Package image provides SIMD-friendly 2D image types and operations.
//
// The core types are Image[T] for single-channel images and Image3[T] for
// three-plane images (e.g., RGB or YUV). Rows are aligned to SIMD vector width
// for efficient vectorized processing.
//
// # Point Operations
//
// Point operations transform each pixel independently:
//
//	BrightnessContrast(img, out, scale, offset) // out = img * scale + offset
//	ClampImage(img, out, minVal, maxVal)        // clamp to range
//	Threshold(img, out, thresh, below, above)   // binary threshold
//
// # Usage Example
//
//	// Create a 1080p image
//	img := image.NewImage[float32](1920, 1080)
//
//	// Apply brightness/contrast adjustment
//	out := image.NewImage[float32](1920, 1080)
//	image.BrightnessContrast(img, out, 1.5, 0.1)
//
// # Edge Handling
//
// Coordinate helper functions for handling out-of-bounds pixel access:
//
//	Mirror(index, size) - reflect at boundaries
//	Clamp(index, size)  - repeat edge pixels
//	Wrap(index, size)   - tile/wrap around
package image
