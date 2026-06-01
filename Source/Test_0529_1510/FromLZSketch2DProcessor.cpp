#include "FromLZSketch2DProcessor.h"

#include "Async/Async.h"
#include "FromLZFaceReconstructor.h"
#include "FromLZImageOps.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformTime.h"
#include "Misc/DateTime.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"

void FFromLZSketch2DProcessor::ProcessCompositeAsync(TArray<uint8> RGBA, int32 Width, int32 Height, const FString& DebugDir, const FSketchSourceInfo& Source, UWorld* World)
{
	// RGBA is taken by value; move it into the async task so the heavy work runs off the game thread.
	const FString DebugDirCopy = DebugDir;
	const FSketchSourceInfo SourceCopy = Source;
	const TWeakObjectPtr<UWorld> WorldCopy(World);
	TArray<uint8> Pixels = MoveTemp(RGBA);

	Async(EAsyncExecution::ThreadPool, [Pixels = MoveTemp(Pixels), Width, Height, DebugDirCopy, SourceCopy, WorldCopy]() mutable
	{
		FFromLZSketch2DProcessor::ProcessComposite(Pixels, Width, Height, DebugDirCopy, SourceCopy, WorldCopy);
	});
}

bool FFromLZSketch2DProcessor::ProcessComposite(const TArray<uint8>& RGBA, int32 Width, int32 Height, const FString& DebugDir, const FSketchSourceInfo& Source, TWeakObjectPtr<UWorld> World)
{
	if (Width <= 0 || Height <= 0 || RGBA.Num() < Width * Height * 4)
	{
		UE_LOG(LogTemp, Warning, TEXT("Sketch2D: invalid input (%dx%d, %d bytes)"), Width, Height, RGBA.Num());
		return false;
	}

	IFileManager::Get().MakeDirectory(*DebugDir, true);

	// Each pipeline run (one Space press) gets its own Press_## folder. Number continues
	// from the highest existing Press_* so runs accumulate across restarts.
	int32 MaxPress = 0;
	IFileManager::Get().IterateDirectory(*DebugDir, [&MaxPress](const TCHAR* InPath, bool bIsDir) -> bool
	{
		if (bIsDir)
		{
			const FString Name = FPaths::GetCleanFilename(FString(InPath));
			if (Name.StartsWith(TEXT("Press_")))
			{
				MaxPress = FMath::Max(MaxPress, FCString::Atoi(*Name.RightChop(6)));
			}
		}
		return true;
	});
	const FString PressName = FString::Printf(TEXT("Press_%02d"), MaxPress + 1);
	const FString PressDir = DebugDir / PressName;
	IFileManager::Get().MakeDirectory(*PressDir, true);

	// Action.json outputs live under Saved/FromAction/Press_##/ (sibling of 2DDebug).
	const FString ActionPressDir = FPaths::GetPath(DebugDir) / TEXT("FromAction") / PressName;
	IFileManager::Get().MakeDirectory(*ActionPressDir, true);

	// Record which FromLZCaptures / FromSketch source files this press consumed
	// (paths relative to Saved/). Written into the Press_## folder as capture_ref.json.
	{
		const FString ProcessedAt = FDateTime::Now().ToString(TEXT("%Y%m%d_%H%M%S"));
		FString RefJson;
		RefJson += TEXT("{\n");
		RefJson += FString::Printf(TEXT("  \"press\": \"%s\",\n"), *PressName);
		RefJson += FString::Printf(TEXT("  \"processed_at\": \"%s\",\n"), *ProcessedAt);
		RefJson += FString::Printf(TEXT("  \"has_capture\": %s,\n"), Source.bHasCapture ? TEXT("true") : TEXT("false"));
		RefJson += FString::Printf(TEXT("  \"capture_stem\": \"%s\",\n"), *Source.CaptureStem);
		RefJson += FString::Printf(TEXT("  \"capture_png\": \"%s\",\n"), *Source.CapturePngRel);
		RefJson += FString::Printf(TEXT("  \"capture_json\": \"%s\",\n"), *Source.CaptureJsonRel);
		RefJson += FString::Printf(TEXT("  \"faces_png\": \"%s\",\n"), *Source.FacesPngRel);
		RefJson += FString::Printf(TEXT("  \"faces_json\": \"%s\",\n"), *Source.FacesJsonRel);
		RefJson += FString::Printf(TEXT("  \"sketch_png\": \"%s\"\n"), *Source.SketchPngRel);
		RefJson += TEXT("}\n");
		FFileHelper::SaveStringToFile(RefJson, *(PressDir / TEXT("capture_ref.json")));
	}

	const double StartTime = FPlatformTime::Seconds();

	// ---- Step 1: preprocess -------------------------------------------------
	// Non-white -> foreground (replaces grayscale+Gaussian+Otsu for the colored composite),
	// then morphological close + 2x2 dilate, then remove small components (Python preprocess()).
	TArray<uint8> Bin;
	FromLZImageOps::BinarizeNonWhite(RGBA, Width, Height, /*WhiteThreshold*/ 240, Bin);
	FromLZImageOps::MorphClose(Bin, Width, Height, /*Kernel*/ 3, /*Iterations*/ 1);
	FromLZImageOps::Dilate2x2(Bin, Width, Height, /*Iterations*/ 1);
	FromLZImageOps::RemoveSmallComponents(Bin, Width, Height, /*MinArea*/ 12);

	// 00_input is the raw composite (saved for reference), 01_binary is the cleaned mask.
	{
		TArray<uint8> InputRGBA = RGBA;
		IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
		if (IW.IsValid())
		{
			IW->SetRaw(InputRGBA.GetData(), InputRGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
			const TArray64<uint8>& C = IW->GetCompressed();
			FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *(PressDir / TEXT("00_input.png")));
		}
	}
	FromLZImageOps::SaveMaskPng(Bin, Width, Height, PressDir / TEXT("01_binary.png"), /*bInvertForDisplay*/ true);

	// ---- Step 2: skeletonize ------------------------------------------------
	TArray<uint8> Skel;
	FromLZImageOps::ZhangSuenThinning(Bin, Width, Height, Skel);
	FromLZImageOps::RemoveSmallComponents(Skel, Width, Height, /*MinArea*/ 6);
	FromLZImageOps::SaveMaskPng(Skel, Width, Height, PressDir / TEXT("02_skeleton.png"), /*bInvertForDisplay*/ true);

	// ---- Step 3: skeleton gap repair ---------------------------------------
	// Connect mutual-nearest endpoints within --skeleton-gap-tol, prune connections
	// that only close a small loop (--skeleton-small-loop-bbox-area-thresh), then
	// trim short dangling branches. Mirrors Python cleanup_skeleton_endpoints.
	TArray<uint8> SkelConnected, SkelSmallLoopPruned, SkelClean;
	FromLZImageOps::CleanupSkeletonEndpoints(
		Skel, Width, Height,
		/*GapTol*/ 50.0f,
		/*ConnectThickness*/ 1,
		/*SmallLoopBboxAreaThresh*/ 1500.0f,
		/*BranchPruneMaxPixels*/ 0.0f, // 0 -> auto = max(30, 3*GapTol)
		SkelConnected, SkelSmallLoopPruned, SkelClean);
	FromLZImageOps::SaveMaskPng(SkelConnected, Width, Height, PressDir / TEXT("03a_skeleton_connected.png"), /*bInvertForDisplay*/ true);
	FromLZImageOps::SaveMaskPng(SkelSmallLoopPruned, Width, Height, PressDir / TEXT("03b_skeleton_small_loop_pruned.png"), /*bInvertForDisplay*/ true);
	FromLZImageOps::SaveMaskPng(SkelClean, Width, Height, PressDir / TEXT("03_skeleton_clean.png"), /*bInvertForDisplay*/ true);

	// Per-pixel color-class map of the composite (red/green/blue/black/none),
	// used to color strokes and to keep RGB strokes from being merged together.
	TArray<uint8> ColorMap;
	FromLZImageOps::BuildColorClassMap(RGBA, Width, Height, ColorMap);
	const int32 ColorSampleRadius = 2;
	const float EndpointTol = 3.0f;
	const float ColorMinRunArc = 18.0f; // absorb short color "blips" at crossings into neighbors

	// ---- Step 4: stroke tracing + color classification --------------------
	// Crossing-number node classification + 8-connected polyline tracing, then
	// assign red/green/blue/black per stroke and split at color boundaries
	// (gap-repair runs reclassified by neighbor color/direction).
	const int32 TraceMinPixels = 3; // --trace-min-pixels
	TArray<FromLZImageOps::FStroke> Strokes;
	FromLZImageOps::TraceStrokes(SkelClean, Width, Height, TraceMinPixels, Strokes);
	FromLZImageOps::SaveStrokesPng(Strokes, Width, Height, PressDir / TEXT("04_strokes.png"));

	TArray<FromLZImageOps::FColoredStroke> ColoredStrokes;
	FromLZImageOps::ColorizeAndSplitStrokes(Strokes, ColorMap, Width, Height, ColorSampleRadius, ColorMinRunArc, ColoredStrokes);
	FromLZImageOps::SaveColoredStrokesPng(ColoredStrokes, Width, Height, PressDir / TEXT("04_strokes_colored.png"));
	FromLZImageOps::SaveColoredStrokesJson(ColoredStrokes, Width, Height, PressDir / TEXT("04_strokes.json"), EndpointTol);

	// ---- Step 5: corner splitting (color preserved, RGB never merged) -----
	TArray<FromLZImageOps::FColoredStroke> SplitColored;
	FromLZImageOps::SplitColoredStrokesAtCorners(
		ColoredStrokes,
		/*AngleThresh*/ 25.0f,            // --split-corner-angle
		/*MinPixels*/ TraceMinPixels,
		/*SegmentArc*/ 30.0f,             // --split-segment-arc
		/*SplitPeakMinDistance*/ 10.0f,   // default
		/*MaxIters*/ 5,                   // default
		SplitColored);

	// Raw per-stroke palette render (each piece a distinct color), same style as 04_strokes.png.
	{
		TArray<FromLZImageOps::FStroke> SplitGeom;
		SplitGeom.Reserve(SplitColored.Num());
		for (const FromLZImageOps::FColoredStroke& CS : SplitColored)
		{
			SplitGeom.Add(CS.Points);
		}
		FromLZImageOps::SaveStrokesPng(SplitGeom, Width, Height, PressDir / TEXT("05_strokes_split.png"));
	}
	FromLZImageOps::SaveColoredStrokesPng(SplitColored, Width, Height, PressDir / TEXT("05_strokes_split_colored.png"));
	FromLZImageOps::SaveColoredStrokesJson(SplitColored, Width, Height, PressDir / TEXT("05_strokes_split.json"), EndpointTol);

	// Helper: emit the three standard outputs for a stroke set
	// (per-stroke palette PNG, class-color PNG, detailed JSON).
	auto SaveStrokeTriplet = [&](const TArray<FromLZImageOps::FColoredStroke>& Set, const FString& Stem)
	{
		TArray<FromLZImageOps::FStroke> Geom;
		Geom.Reserve(Set.Num());
		for (const FromLZImageOps::FColoredStroke& CS : Set)
		{
			Geom.Add(CS.Points);
		}
		FromLZImageOps::SaveStrokesPng(Geom, Width, Height, PressDir / (Stem + TEXT(".png")));
		FromLZImageOps::SaveColoredStrokesPng(Set, Width, Height, PressDir / (Stem + TEXT("_colored.png")));
		FromLZImageOps::SaveColoredStrokesJson(Set, Width, Height, PressDir / (Stem + TEXT(".json")), EndpointTol);
	};

	// ---- Step 6: same-color merge ----------------------------------------
	// Only strokes of the same red/green/blue/black class are merged.
	TArray<FromLZImageOps::FColoredStroke> Merged;
	FromLZImageOps::MergeColoredStrokesSameColor(
		SplitColored,
		/*MaxGap*/ 3.0f,                 // --post-split-merge-gap
		/*MaxAngle*/ 12.0f,              // --post-split-merge-angle
		/*MaxIters*/ 80,
		/*ProtectJunctionRadius*/ 3.0f,
		Merged);
	SaveStrokeTriplet(Merged, TEXT("06_merged"));

	// ---- Step 7: stroke metrics ------------------------------------------
	// arc / chord / straightness / PCA-line errors / direction; JSON gains kind + metrics.
	FromLZImageOps::ComputeStrokeMetrics(Merged);
	SaveStrokeTriplet(Merged, TEXT("07_stroke_info"));

	// ---- Step 8: enclosed-region mask ------------------------------------
	TArray<uint8> EnclosedMask, EnclosedBarrier;
	FromLZImageOps::ComputeEnclosedRegionMask(Merged, Width, Height, /*Thickness*/ 3, EnclosedMask, EnclosedBarrier);
	FromLZImageOps::SaveMaskPng(EnclosedBarrier, Width, Height, PressDir / TEXT("08a_enclosed_barrier.png"), /*bInvertForDisplay*/ true);
	FromLZImageOps::SaveMaskPng(EnclosedMask, Width, Height, PressDir / TEXT("08_enclosed_mask.png"), /*bInvertForDisplay*/ true);

	// ---- Step 9: per-component red cap-loops -> per-component side -> translate-copy --
	// Each connected red(+nearby black) component is one Component_%%: its cap loop is found
	// (red-only, or bridged by black), the longest green near it is the extrusion side, and
	// the cap is translate-copied along the side vector. Components are processed in parallel.
	TArray<FromLZImageOps::FCapExtrusionResult> Caps;
	const int32 NumCaps = FromLZImageOps::RecoverCapExtrusionsPerComponent(
		Merged, /*NodeTol*/ 20.0f, /*BlackSelectTol*/ 50.0f, Width, Height, PressDir, ActionPressDir, Caps);

	// ---- Step 10: per-component action -> face-mask overlap -> runtime face rebuild --
	FFromLZFaceReconstructor::ProcessPress(PressDir, ActionPressDir, World);

	const double Elapsed = FPlatformTime::Seconds() - StartTime;
	UE_LOG(LogTemp, Log, TEXT("Sketch2D: steps 1-10 done in %.3fs (%dx%d): %d merged strokes; %d cap component(s) -> %s"),
		Elapsed, Width, Height, Merged.Num(), NumCaps, *PressDir);

	return true;
}
