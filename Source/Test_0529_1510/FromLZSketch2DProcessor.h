#pragma once

#include "CoreMinimal.h"

class UWorld;

// Migrated 2D sketch-analysis pipeline (steps 1-8 of the Python extrusion script),
// reimplemented in pure C++ with no third-party dependencies.
//
// Triggered after the Space-key composite (wContextSketch_raw.png) is produced.
// All debug artifacts are written under <ProjectSaved>/2DDebug/.
//
// Implemented incrementally:
//   [x] Step 1 preprocess (non-white binarize + close/dilate + remove small comps)
//   [x] Step 2 skeletonize (Zhang-Suen + remove small skeleton comps)
//   [x] Step 3 skeleton gap repair (connect endpoints + small-loop / dangling prune)
//   [x] Step 4 stroke tracing (crossing-number nodes + 8-connected polylines)
//             + color classification (red/green/blue/black) and color-boundary split
//   [x] Step 5 corner split (PCA arc-window axis-angle + iterative NMS), color preserved
//   [x] Step 6 same-color merge (RGB/black classes never merged across colors)
//   [x] Step 7 stroke metrics (arc/chord/straightness/PCA errors/direction)
//   [x] Step 8 enclosed-region mask (endpoint-nearest-connect + flood)
//   [x] Step 9 per-component red cap-loops -> per-component longest-green side -> translate-copy
//       Each run writes to Saved/2DDebug/Press_##/, with one Component_%% subfolder per cap.
//   Each stroke step emits a per-stroke palette PNG, a class-color PNG, and a JSON
//   (id / color / kind / metrics / endpoints / neighbors / points).
//   [ ] Step 6 post-split merge
//   [ ] Step 7 stroke metrics
//   [ ] Step 8 enclosed-region mask
// Records which FromLZCaptures / FromSketch source files a Space press consumed.
// Paths are relative to the project's Saved/ folder (e.g. "FromLZCaptures/FromLZ_xx.png").
struct FSketchSourceInfo
{
	bool bHasCapture = false;
	FString CaptureStem;     // e.g. "FromLZ_20260531_131543"
	FString CapturePngRel;   // "FromLZCaptures/FromLZ_...png"
	FString CaptureJsonRel;  // "FromLZCaptures/FromLZ_...json"
	FString FacesPngRel;     // "FromLZCaptures/FromLZ_..._faces.png"
	FString FacesJsonRel;    // "FromLZCaptures/FromLZ_..._faces.json"
	FString SketchPngRel;    // "FromSketch/....png"
};

class FFromLZSketch2DProcessor
{
public:
	// Copies the RGBA buffer and runs the pipeline on a background thread so the
	// game thread never blocks on the (potentially heavy) thinning pass.
	static void ProcessCompositeAsync(TArray<uint8> RGBA, int32 Width, int32 Height, const FString& DebugDir, const FSketchSourceInfo& Source, UWorld* World);

	// Synchronous entry point (runs on the calling thread). Returns true on success.
	static bool ProcessComposite(const TArray<uint8>& RGBA, int32 Width, int32 Height, const FString& DebugDir, const FSketchSourceInfo& Source, TWeakObjectPtr<UWorld> World);
};
