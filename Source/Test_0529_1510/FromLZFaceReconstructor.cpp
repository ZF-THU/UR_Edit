#include "FromLZFaceReconstructor.h"

#include "Algo/Reverse.h"
#include "Async/Async.h"
#include "Async/ParallelFor.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Components/PrimitiveComponent.h"
#include "Components/StaticMeshComponent.h"
#include "DynamicMesh/DynamicMesh3.h"
#include "Engine/Engine.h"
#include "Engine/StaticMesh.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Materials/Material.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "Operations/MeshBoolean.h"
#include "ProceduralMeshComponent.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "StaticMeshResources.h"

namespace
{
	const FName ReconstructedFaceTag(TEXT("FromLZ_ReconstructedFace"));
	const FName ReconstructedSolidTag(TEXT("FromLZ_ReconstructedSolid"));
	const FName Step11BooleanResultTag(TEXT("FromLZ_Step11BooleanResult"));
	const FName Step11HiddenSourceTag(TEXT("FromLZ_Step11HiddenSource"));
	const FName Step11ActionAttachTag(TEXT("FromLZ_Action_Attach"));
	const FName Step11ActionExcavateCutterTag(TEXT("FromLZ_Action_ExcavateCutter"));
	constexpr double MinOverlapRatio = 0.05;
	constexpr double NormalParallelThresholdDegrees = 10.0;
	constexpr double MinProjectedNormalPixels = 1.0;
	constexpr double SolidCollinearTolerancePixels = 0.75;
	constexpr double SolidRdpTolerancePixels = 1.25;
	constexpr int32 SolidTargetMaxLoopPoints = 384;
	constexpr int32 MinSolidDepthSamples = 3;
	constexpr double ExcavationCutterNormalScale = 1.2;
	constexpr double Step11BooleanSnapToleranceCm = 0.05;
	constexpr double Step11BooleanMinRenderableEdgeCm = 0.05;
	constexpr double Step11BooleanBoundsExpandCm = 1.0;
	const FColor ReconstructedDebugBlue(0, 120, 255, 255);
	static FString GActiveUndoPressId;

	struct FFaceInfo
	{
		int32 Id = -1;
		FColor Color = FColor::Black;
		FVector PlanePoint = FVector::ZeroVector;
		FVector Normal = FVector::UpVector;
		TArray<FVector2D> KeyPoints2D;
		TArray<FVector> KeyPoints3D;
	};

	struct FCameraInfo
	{
		FVector Location = FVector::ZeroVector;
		FVector Forward = FVector::ForwardVector;
		FVector Right = FVector::RightVector;
		FVector Up = FVector::UpVector;
		double Fov = 90.0;
		double OrthoWidth = 1536.0;
		bool bOrthographic = false;
	};

	struct FOverlapAccum
	{
		int32 Pixels = 0;
		double SumX = 0.0;
		double SumY = 0.0;
	};

	struct FFaceCandidate
	{
		int32 FaceId = -1;
		int32 OverlapPixels = 0;
		double OverlapRatio = 0.0;
		FVector2D MaskCentroid = FVector2D::ZeroVector;
		bool bHasPlaneHit = false;
		FVector PlaneHit = FVector::ZeroVector;
		double DistanceToCamera = 0.0;
		bool bHasProjectedNormal = false;
		FVector2D ProjectedNormal2D = FVector2D::ZeroVector;
		double NormalGreenAngleDegrees = -1.0;
		bool bNormalParallelPass = false;
	};

	struct FReconstructedMesh
	{
		FString ActorName;
		FName Tag;
		TArray<FVector> VerticesWorld;
		TArray<int32> Triangles;
		FVector Normal = FVector::UpVector;
		FColor Color = ReconstructedDebugBlue;
		bool bIsExcavateCutter = false;
		FString PressId;
	};

	struct FSolidDepthSample
	{
		int32 Index = -1;
		bool bValid = false;
		FString Error;
		FVector2D SourcePixel = FVector2D::ZeroVector;
		FVector2D CopiedPixel = FVector2D::ZeroVector;
		FVector SourceWorld = FVector::ZeroVector;
		FVector PointOnExtrusion = FVector::ZeroVector;
		FVector PointOnRay = FVector::ZeroVector;
		double Depth = 0.0;
		double ClosestWorldDistance = 0.0;
		double ReprojectionErrorPixels = 0.0;
	};

	struct FSolidReconstructionResult
	{
		FString ComponentName;
		FString Action;
		FString SourcePolygonKey;
		FString CopiedPolygonKey;
		FString Error;
		FString Warning;
		FString ActorName;
		bool bSuccess = false;
		int32 CapWidth = 0;
		int32 CapHeight = 0;
		int32 FacesWidth = 0;
		int32 FacesHeight = 0;
		int32 SelectedFaceId = -1;
		FVector SourcePlanePoint = FVector::ZeroVector;
		FVector SourcePlaneNormal = FVector::UpVector;
		FVector OrientedNormal = FVector::UpVector;
		FVector2D SourceToCopiedVector2D = FVector2D::ZeroVector;
		FVector2D ProjectedNormal2D = FVector2D::ZeroVector;
		double ExtrusionDepth = 0.0;
		double MaxDepthSampleReprojectionErrorPixels = 0.0;
		double MeanCopiedReprojectionErrorPixels = 0.0;
		double MaxCopiedReprojectionErrorPixels = 0.0;
		TArray<FSolidDepthSample> DepthSamples;
		TArray<FVector2D> SourceLoop2D;
		TArray<FVector2D> CopiedTargetLoop2D;
		TArray<FVector2D> ReprojectedSourceLoop2D;
		TArray<FVector2D> ReprojectedCopiedLoop2D;
		TArray<FVector> SourceLoopWorld;
		TArray<FVector> CopiedLoopWorld;
		TArray<FVector> MeshVerticesWorld;
		TArray<int32> MeshTriangles;
		FVector MeshNormal = FVector::UpVector;
	};

	struct FComponentResult
	{
		FString ComponentName;
		FString Action;
		FString PolygonKey;
		FString Error;
		FString ActorName;
		int32 CapWidth = 0;
		int32 CapHeight = 0;
		int32 FacesWidth = 0;
		int32 FacesHeight = 0;
		int32 CapMaskPixels = 0;
		int32 MinOverlapPixels = 0;
		int32 SelectedFaceId = -1;
		FVector2D GreenLineVector2D = FVector2D::ZeroVector;
		// Every cap-connected green line, mapped to faces image space. The candidate-face
		// normal passes the parallel filter if it is parallel to ANY of these.
		TArray<FVector2D> GreenLineVectors2D;    // normalized direction per green
		TArray<FVector2D> GreenSegStarts;        // chord start in faces space (debug)
		TArray<FVector2D> GreenSegEnds;          // chord end in faces space (debug)
		bool bSuccess = false;
		TArray<FFaceCandidate> Candidates;
		FVector SelectedPlaneHit = FVector::ZeroVector;
		TArray<FVector> MeshVerticesWorld;
		TArray<int32> MeshTriangles;
		FVector MeshNormal = FVector::UpVector;
		FSolidReconstructionResult Solid;
	};

	struct FCommonInputs
	{
		FString CaptureJsonRel;
		FString FacesPngRel;
		FString FacesJsonRel;
		FString CaptureJsonPath;
		FString FacesPngPath;
		FString FacesJsonPath;
		FCameraInfo Camera;
		TArray<FFaceInfo> Faces;
		TMap<int32, int32> FaceIndexById;
		TMap<uint32, int32> FaceIdByColorKey;
		TArray<uint8> FacesRGBA;
		int32 FacesWidth = 0;
		int32 FacesHeight = 0;
	};

	struct FStep11MeshDiagnostics
	{
		FString Label;
		FString SourceType;
		int32 VertexCount = 0;
		int32 TriangleCount = 0;
		int32 EdgeCount = 0;
		int32 BoundaryEdgeCount = 0;
		int32 NonManifoldEdgeCount = 0;
		int32 InconsistentOrientationEdgeCount = 0;
		int32 InvalidTriangleCount = 0;
		int32 DegenerateTriangleCount = 0;
		int32 TinyTriangleCount = 0;
		int32 DuplicateTriangleCount = 0;
		double SurfaceArea = 0.0;
		double MinTriangleArea = 0.0;
		double MaxTriangleArea = 0.0;
		double SignedVolume = 0.0;
		double AbsVolume = 0.0;
		double SignedVolumeBeforeOrientationFix = 0.0;
		double SignedVolumeAfterOrientationFix = 0.0;
		double MinEdgeLength = 0.0;
		double MaxEdgeLength = 0.0;
		double MeanEdgeLength = 0.0;
		FBox Bounds = FBox(ForceInit);
		bool bOrientationReversedForBoolean = false;
	};

	static uint32 ColorKey(uint8 R, uint8 G, uint8 B)
	{
		return (uint32(R) << 16) | (uint32(G) << 8) | uint32(B);
	}

	static FString ResolveSavedPath(const FString& RelativeOrAbsolute)
	{
		if (RelativeOrAbsolute.IsEmpty())
		{
			return FString();
		}
		if (FPaths::IsRelative(RelativeOrAbsolute))
		{
			return FPaths::ProjectSavedDir() / RelativeOrAbsolute;
		}
		return RelativeOrAbsolute;
	}

	static FString StripTrailingFacesSuffixes(FString Stem)
	{
		while (Stem.EndsWith(TEXT("_faces")))
		{
			Stem.LeftChopInline(6);
		}
		return Stem;
	}

	static FString BuildCaptureRelPath(const FString& CaptureStem, const FString& Extension)
	{
		if (CaptureStem.IsEmpty())
		{
			return FString();
		}
		return TEXT("FromLZCaptures/") + StripTrailingFacesSuffixes(CaptureStem) + Extension;
	}

	static FString BuildFacesRelPath(const FString& CaptureStem, const FString& Extension)
	{
		if (CaptureStem.IsEmpty())
		{
			return FString();
		}
		return TEXT("FromLZCaptures/") + StripTrailingFacesSuffixes(CaptureStem) + TEXT("_faces") + Extension;
	}

	static void NormalizeFacesRelPath(FString& RelPath, const FString& Extension)
	{
		if (RelPath.IsEmpty())
		{
			return;
		}

		const FString Dir = FPaths::GetPath(RelPath);
		const FString Stem = StripTrailingFacesSuffixes(FPaths::GetBaseFilename(RelPath));
		const FString Filename = Stem + TEXT("_faces") + Extension;
		RelPath = Dir.IsEmpty() ? Filename : Dir / Filename;
	}

	static void NormalizeCaptureRelPath(FString& RelPath, const FString& Extension)
	{
		if (RelPath.IsEmpty())
		{
			return;
		}

		const FString Dir = FPaths::GetPath(RelPath);
		const FString Stem = StripTrailingFacesSuffixes(FPaths::GetBaseFilename(RelPath));
		const FString Filename = Stem + Extension;
		RelPath = Dir.IsEmpty() ? Filename : Dir / Filename;
	}

	static bool LoadJsonObject(const FString& Path, TSharedPtr<FJsonObject>& OutObject)
	{
		FString Text;
		if (!FFileHelper::LoadFileToString(Text, *Path))
		{
			return false;
		}

		const TSharedRef<TJsonReader<TCHAR>> Reader = TJsonReaderFactory<TCHAR>::Create(Text);
		return FJsonSerializer::Deserialize(Reader, OutObject) && OutObject.IsValid();
	}

	static bool SaveJsonObject(const TSharedRef<FJsonObject>& Object, const FString& Path)
	{
		FString Text;
		const TSharedRef<TJsonWriter<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>> Writer =
			TJsonWriterFactory<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>::Create(&Text);
		if (!FJsonSerializer::Serialize(Object, Writer))
		{
			return false;
		}
		return FFileHelper::SaveStringToFile(Text, *Path, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM);
	}

	static bool DecodePngToRGBA(const FString& Path, TArray<uint8>& OutPixels, int32& OutWidth, int32& OutHeight)
	{
		OutPixels.Reset();
		OutWidth = 0;
		OutHeight = 0;

		TArray<uint8> RawFileData;
		if (!FFileHelper::LoadFileToArray(RawFileData, *Path))
		{
			return false;
		}

		IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
		if (!ImageWrapper.IsValid() || !ImageWrapper->SetCompressed(RawFileData.GetData(), RawFileData.Num()))
		{
			return false;
		}
		if (!ImageWrapper->GetRaw(ERGBFormat::RGBA, 8, OutPixels))
		{
			return false;
		}

		OutWidth = ImageWrapper->GetWidth();
		OutHeight = ImageWrapper->GetHeight();
		return OutWidth > 0 && OutHeight > 0 && OutPixels.Num() >= OutWidth * OutHeight * 4;
	}

	static bool SaveRGBAToPng(const TArray<uint8>& RGBA, int32 Width, int32 Height, const FString& Path)
	{
		if (Width <= 0 || Height <= 0 || RGBA.Num() < Width * Height * 4)
		{
			return false;
		}

		IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> Wrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);
		if (!Wrapper.IsValid())
		{
			return false;
		}

		Wrapper->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
		const TArray64<uint8>& Compressed = Wrapper->GetCompressed();
		return FFileHelper::SaveArrayToFile(
			TArrayView<const uint8>(Compressed.GetData(), static_cast<int32>(Compressed.Num())),
			*Path);
	}

	static bool ParseVector2DArray(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, TArray<FVector2D>& Out)
	{
		Out.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Outer = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(Key, Outer))
		{
			return false;
		}

		for (const TSharedPtr<FJsonValue>& Value : *Outer)
		{
			if (!Value.IsValid() || Value->Type != EJson::Array)
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& Pair = Value->AsArray();
			if (Pair.Num() < 2)
			{
				continue;
			}
			Out.Emplace(Pair[0]->AsNumber(), Pair[1]->AsNumber());
		}
		return Out.Num() > 0;
	}

	// Parse an array of 2-point segments: [[[x0,y0],[x1,y1]], ...].
	static bool ParseSegment2DArray(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, TArray<FVector2D>& OutStarts, TArray<FVector2D>& OutEnds)
	{
		OutStarts.Reset();
		OutEnds.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Outer = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(Key, Outer))
		{
			return false;
		}
		for (const TSharedPtr<FJsonValue>& Value : *Outer)
		{
			if (!Value.IsValid() || Value->Type != EJson::Array)
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& Seg = Value->AsArray();
			if (Seg.Num() < 2 || !Seg[0].IsValid() || !Seg[1].IsValid() ||
				Seg[0]->Type != EJson::Array || Seg[1]->Type != EJson::Array)
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& A = Seg[0]->AsArray();
			const TArray<TSharedPtr<FJsonValue>>& B = Seg[1]->AsArray();
			if (A.Num() < 2 || B.Num() < 2)
			{
				continue;
			}
			OutStarts.Emplace(A[0]->AsNumber(), A[1]->AsNumber());
			OutEnds.Emplace(B[0]->AsNumber(), B[1]->AsNumber());
		}
		return OutStarts.Num() > 0;
	}

	static bool ParseVector2DField(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, FVector2D& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(Key, Values) || Values->Num() < 2)
		{
			return false;
		}
		Out = FVector2D((*Values)[0]->AsNumber(), (*Values)[1]->AsNumber());
		return true;
	}

	static bool ParseVectorArrayField(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, TArray<FVector>& Out)
	{
		Out.Reset();
		const TArray<TSharedPtr<FJsonValue>>* Outer = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(Key, Outer))
		{
			return false;
		}

		for (const TSharedPtr<FJsonValue>& Value : *Outer)
		{
			if (!Value.IsValid() || Value->Type != EJson::Array)
			{
				continue;
			}
			const TArray<TSharedPtr<FJsonValue>>& Triple = Value->AsArray();
			if (Triple.Num() < 3)
			{
				continue;
			}
			Out.Emplace(Triple[0]->AsNumber(), Triple[1]->AsNumber(), Triple[2]->AsNumber());
		}
		return Out.Num() > 0;
	}

	static bool ParseColorField(const TSharedPtr<FJsonObject>& Object, FColor& OutColor)
	{
		const TArray<TSharedPtr<FJsonValue>>* ColorArray = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(TEXT("color_rgb"), ColorArray) || ColorArray->Num() < 3)
		{
			return false;
		}

		OutColor = FColor(
			uint8(FMath::Clamp(FMath::RoundToInt((*ColorArray)[0]->AsNumber()), 0, 255)),
			uint8(FMath::Clamp(FMath::RoundToInt((*ColorArray)[1]->AsNumber()), 0, 255)),
			uint8(FMath::Clamp(FMath::RoundToInt((*ColorArray)[2]->AsNumber()), 0, 255)),
			255);
		return true;
	}

	static bool ParseVectorField(const TSharedPtr<FJsonObject>& Object, const TCHAR* Key, FVector& Out)
	{
		const TArray<TSharedPtr<FJsonValue>>* Values = nullptr;
		if (!Object.IsValid() || !Object->TryGetArrayField(Key, Values) || Values->Num() < 3)
		{
			return false;
		}
		Out = FVector((*Values)[0]->AsNumber(), (*Values)[1]->AsNumber(), (*Values)[2]->AsNumber());
		return true;
	}

	static bool LoadFacesJson(const FString& Path, TArray<FFaceInfo>& OutFaces)
	{
		OutFaces.Reset();

		TSharedPtr<FJsonObject> Root;
		if (!LoadJsonObject(Path, Root))
		{
			return false;
		}

		const TArray<TSharedPtr<FJsonValue>>* FacesArray = nullptr;
		if (!Root->TryGetArrayField(TEXT("faces"), FacesArray))
		{
			return false;
		}

		for (const TSharedPtr<FJsonValue>& FaceValue : *FacesArray)
		{
			const TSharedPtr<FJsonObject> FaceObject = FaceValue.IsValid() ? FaceValue->AsObject() : nullptr;
			if (!FaceObject.IsValid())
			{
				continue;
			}

			FFaceInfo Face;
			double IdNumber = -1.0;
			FaceObject->TryGetNumberField(TEXT("id"), IdNumber);
			Face.Id = FMath::RoundToInt(IdNumber);
			ParseColorField(FaceObject, Face.Color);
			ParseVectorField(FaceObject, TEXT("plane_point"), Face.PlanePoint);
			ParseVectorField(FaceObject, TEXT("normal_world"), Face.Normal);
			Face.Normal = Face.Normal.GetSafeNormal();
			ParseVector2DArray(FaceObject, TEXT("key_points_2d"), Face.KeyPoints2D);
			ParseVectorArrayField(FaceObject, TEXT("key_points_3d"), Face.KeyPoints3D);

			if (Face.Id >= 0 && Face.KeyPoints2D.Num() >= 3 && Face.KeyPoints3D.Num() == Face.KeyPoints2D.Num() && !Face.Normal.IsNearlyZero())
			{
				OutFaces.Add(MoveTemp(Face));
			}
		}

		return OutFaces.Num() > 0;
	}

	static bool LoadCameraJson(const FString& Path, FCameraInfo& OutCamera)
	{
		TSharedPtr<FJsonObject> Root;
		if (!LoadJsonObject(Path, Root))
		{
			return false;
		}

		const TSharedPtr<FJsonObject>* TransformObject = nullptr;
		if (!Root->TryGetObjectField(TEXT("camera_component_transform"), TransformObject) || !TransformObject || !TransformObject->IsValid())
		{
			return false;
		}

		double X = 0.0, Y = 0.0, Z = 0.0;
		double Pitch = 0.0, Yaw = 0.0, Roll = 0.0;
		(*TransformObject)->TryGetNumberField(TEXT("location_x"), X);
		(*TransformObject)->TryGetNumberField(TEXT("location_y"), Y);
		(*TransformObject)->TryGetNumberField(TEXT("location_z"), Z);
		(*TransformObject)->TryGetNumberField(TEXT("pitch"), Pitch);
		(*TransformObject)->TryGetNumberField(TEXT("yaw"), Yaw);
		(*TransformObject)->TryGetNumberField(TEXT("roll"), Roll);

		OutCamera.Location = FVector(X, Y, Z);
		const FRotator Rot(Pitch, Yaw, Roll);
		const FRotationMatrix RotMatrix(Rot);
		OutCamera.Forward = RotMatrix.GetScaledAxis(EAxis::X).GetSafeNormal();
		OutCamera.Right = RotMatrix.GetScaledAxis(EAxis::Y).GetSafeNormal();
		OutCamera.Up = RotMatrix.GetScaledAxis(EAxis::Z).GetSafeNormal();

		const TSharedPtr<FJsonObject>* ViewObject = nullptr;
		if (Root->TryGetObjectField(TEXT("camera_view"), ViewObject) && ViewObject && ViewObject->IsValid())
		{
			(*ViewObject)->TryGetNumberField(TEXT("fov"), OutCamera.Fov);
			(*ViewObject)->TryGetNumberField(TEXT("ortho_width"), OutCamera.OrthoWidth);
			FString ProjectionMode;
			if ((*ViewObject)->TryGetStringField(TEXT("projection_mode"), ProjectionMode))
			{
				OutCamera.bOrthographic = ProjectionMode.Contains(TEXT("Orthographic"));
			}
		}

		return !OutCamera.Forward.IsNearlyZero();
	}

	static bool LoadCaptureRef(const FString& PressDir, FCommonInputs& OutInputs)
	{
		TSharedPtr<FJsonObject> Root;
		if (!LoadJsonObject(PressDir / TEXT("capture_ref.json"), Root))
		{
			return false;
		}

		FString CaptureStem;
		Root->TryGetStringField(TEXT("capture_stem"), CaptureStem);
		CaptureStem = StripTrailingFacesSuffixes(CaptureStem);
		Root->TryGetStringField(TEXT("capture_json"), OutInputs.CaptureJsonRel);
		Root->TryGetStringField(TEXT("faces_png"), OutInputs.FacesPngRel);
		Root->TryGetStringField(TEXT("faces_json"), OutInputs.FacesJsonRel);

		if (OutInputs.CaptureJsonRel.IsEmpty() && !CaptureStem.IsEmpty())
		{
			OutInputs.CaptureJsonRel = BuildCaptureRelPath(CaptureStem, TEXT(".json"));
		}
		if (OutInputs.FacesPngRel.IsEmpty() && !CaptureStem.IsEmpty())
		{
			OutInputs.FacesPngRel = BuildFacesRelPath(CaptureStem, TEXT(".png"));
		}
		if (OutInputs.FacesJsonRel.IsEmpty() && !CaptureStem.IsEmpty())
		{
			OutInputs.FacesJsonRel = BuildFacesRelPath(CaptureStem, TEXT(".json"));
		}
		NormalizeCaptureRelPath(OutInputs.CaptureJsonRel, TEXT(".json"));
		NormalizeFacesRelPath(OutInputs.FacesPngRel, TEXT(".png"));
		NormalizeFacesRelPath(OutInputs.FacesJsonRel, TEXT(".json"));

		OutInputs.CaptureJsonPath = ResolveSavedPath(OutInputs.CaptureJsonRel);
		OutInputs.FacesPngPath = ResolveSavedPath(OutInputs.FacesPngRel);
		OutInputs.FacesJsonPath = ResolveSavedPath(OutInputs.FacesJsonRel);
		return !OutInputs.CaptureJsonPath.IsEmpty() && !OutInputs.FacesPngPath.IsEmpty() && !OutInputs.FacesJsonPath.IsEmpty();
	}

	static void BuildFaceLookups(FCommonInputs& Inputs)
	{
		Inputs.FaceIndexById.Reset();
		Inputs.FaceIdByColorKey.Reset();
		for (int32 i = 0; i < Inputs.Faces.Num(); ++i)
		{
			const FFaceInfo& Face = Inputs.Faces[i];
			Inputs.FaceIndexById.Add(Face.Id, i);
			Inputs.FaceIdByColorKey.Add(ColorKey(Face.Color.R, Face.Color.G, Face.Color.B), Face.Id);
		}
	}

	static double PolygonArea2D(const TArray<FVector2D>& Poly)
	{
		double Area = 0.0;
		for (int32 i = 0, j = Poly.Num() - 1; i < Poly.Num(); j = i++)
		{
			Area += double(Poly[j].X) * double(Poly[i].Y) - double(Poly[i].X) * double(Poly[j].Y);
		}
		return Area * 0.5;
	}

	static bool PointInPolygon(const TArray<FVector2D>& Poly, const FVector2D& P)
	{
		bool bInside = false;
		for (int32 i = 0, j = Poly.Num() - 1; i < Poly.Num(); j = i++)
		{
			const FVector2D& A = Poly[i];
			const FVector2D& B = Poly[j];
			const double Denom = double(B.Y) - double(A.Y);
			if (((A.Y > P.Y) != (B.Y > P.Y)) &&
				FMath::Abs(Denom) > 1e-12)
			{
				const double XIntersect = (double(B.X) - double(A.X)) * (double(P.Y) - double(A.Y)) / Denom + double(A.X);
				if (double(P.X) < XIntersect)
				{
					bInside = !bInside;
				}
			}
		}
		return bInside;
	}

	static void RasterizePolygonMask(const TArray<FVector2D>& Poly, int32 Width, int32 Height, TArray<uint8>& OutMask, int32& OutPixelCount)
	{
		OutMask.Init(0, Width * Height);
		OutPixelCount = 0;
		if (Poly.Num() < 3 || Width <= 0 || Height <= 0)
		{
			return;
		}

		double MinX = Width - 1;
		double MinY = Height - 1;
		double MaxX = 0;
		double MaxY = 0;
		for (const FVector2D& P : Poly)
		{
			MinX = FMath::Min(MinX, P.X);
			MinY = FMath::Min(MinY, P.Y);
			MaxX = FMath::Max(MaxX, P.X);
			MaxY = FMath::Max(MaxY, P.Y);
		}

		const int32 X0 = FMath::Clamp(FMath::FloorToInt(MinX), 0, Width - 1);
		const int32 Y0 = FMath::Clamp(FMath::FloorToInt(MinY), 0, Height - 1);
		const int32 X1 = FMath::Clamp(FMath::CeilToInt(MaxX), 0, Width - 1);
		const int32 Y1 = FMath::Clamp(FMath::CeilToInt(MaxY), 0, Height - 1);

		for (int32 y = Y0; y <= Y1; ++y)
		{
			for (int32 x = X0; x <= X1; ++x)
			{
				if (PointInPolygon(Poly, FVector2D(double(x) + 0.5, double(y) + 0.5)))
				{
					OutMask[y * Width + x] = 255;
					++OutPixelCount;
				}
			}
		}
	}

	static bool SaveMaskPng(const TArray<uint8>& Mask, int32 Width, int32 Height, const FString& Path)
	{
		TArray<uint8> RGBA;
		RGBA.SetNumUninitialized(Width * Height * 4);
		for (int32 i = 0; i < Width * Height; ++i)
		{
			const uint8 V = Mask[i] > 0 ? 0 : 255;
			const int32 Off = i * 4;
			RGBA[Off + 0] = V;
			RGBA[Off + 1] = V;
			RGBA[Off + 2] = V;
			RGBA[Off + 3] = 255;
		}
		return SaveRGBAToPng(RGBA, Width, Height, Path);
	}

	static uint8 BlendChannel(uint8 A, uint8 B, double T)
	{
		return uint8(FMath::Clamp(FMath::RoundToInt(double(A) * (1.0 - T) + double(B) * T), 0, 255));
	}

	static bool SaveOverlapPng(
		const TArray<uint8>& FacesRGBA, const TArray<uint8>& Mask,
		const TMap<uint32, int32>& FaceIdByColorKey, const TSet<int32>& CandidateFaceIds,
		const TSet<int32>& ParallelFaceIds,
		int32 SelectedFaceId, int32 Width, int32 Height, const FString& Path)
	{
		TArray<uint8> RGBA = FacesRGBA;
		if (RGBA.Num() < Width * Height * 4 || Mask.Num() < Width * Height)
		{
			return false;
		}

		for (int32 i = 0; i < Width * Height; ++i)
		{
			if (Mask[i] == 0)
			{
				continue;
			}

			const int32 Off = i * 4;
			const uint32 Key = ColorKey(RGBA[Off + 0], RGBA[Off + 1], RGBA[Off + 2]);
			const int32* FaceId = FaceIdByColorKey.Find(Key);
			FColor Overlay = FColor(255, 60, 60, 255);
			if (FaceId)
			{
				if (*FaceId == SelectedFaceId)
				{
					Overlay = FColor(40, 230, 80, 255);
				}
				else if (ParallelFaceIds.Contains(*FaceId))
				{
					Overlay = FColor(255, 220, 40, 255);
				}
				else if (CandidateFaceIds.Contains(*FaceId))
				{
					Overlay = FColor(255, 140, 40, 255);
				}
				else
				{
					continue;
				}
			}

			RGBA[Off + 0] = BlendChannel(RGBA[Off + 0], Overlay.R, 0.65);
			RGBA[Off + 1] = BlendChannel(RGBA[Off + 1], Overlay.G, 0.65);
			RGBA[Off + 2] = BlendChannel(RGBA[Off + 2], Overlay.B, 0.65);
			RGBA[Off + 3] = 255;
		}

		return SaveRGBAToPng(RGBA, Width, Height, Path);
	}

	static FVector CameraRayDirection(const FCameraInfo& Camera, int32 Width, int32 Height, const FVector2D& Pixel)
	{
		const double NdcX = 2.0 * ((Pixel.X + 0.5) / double(Width)) - 1.0;
		const double NdcY = 1.0 - 2.0 * ((Pixel.Y + 0.5) / double(Height));
		const double TanX = FMath::Tan(FMath::DegreesToRadians(Camera.Fov * 0.5));
		const double TanY = TanX * (double(Height) / double(Width));
		return (Camera.Forward + Camera.Right * (NdcX * TanX) + Camera.Up * (NdcY * TanY)).GetSafeNormal();
	}

	static FVector CameraOrthoRayOrigin(const FCameraInfo& Camera, int32 Width, int32 Height, const FVector2D& Pixel)
	{
		const double NdcX = 2.0 * ((Pixel.X + 0.5) / double(Width)) - 1.0;
		const double NdcY = 1.0 - 2.0 * ((Pixel.Y + 0.5) / double(Height));
		return Camera.Location
			+ Camera.Right * (NdcX * Camera.OrthoWidth * 0.5)
			+ Camera.Up * (NdcY * Camera.OrthoWidth * 0.5 * (double(Height) / double(Width)));
	}

	static double ResolveStep10OrthoWidth(const FCameraInfo& Camera, const FVector& AnchorWorld)
	{
		if (FMath::IsFinite(Camera.OrthoWidth) && Camera.OrthoWidth > 1e-6)
		{
			return Camera.OrthoWidth;
		}

		const double Depth = FVector::DotProduct(AnchorWorld - Camera.Location, Camera.Forward);
		const double TanX = FMath::Tan(FMath::DegreesToRadians(Camera.Fov * 0.5));
		if (Depth > 1e-6 && FMath::IsFinite(TanX) && FMath::Abs(TanX) > 1e-8)
		{
			return 2.0 * Depth * TanX;
		}
		return 0.0;
	}

	static FVector CameraOrthoRayOriginWithWidth(
		const FCameraInfo& Camera, int32 Width, int32 Height, const FVector2D& Pixel, double OrthoWidth)
	{
		const double NdcX = 2.0 * ((Pixel.X + 0.5) / double(Width)) - 1.0;
		const double NdcY = 1.0 - 2.0 * ((Pixel.Y + 0.5) / double(Height));
		return Camera.Location
			+ Camera.Right * (NdcX * OrthoWidth * 0.5)
			+ Camera.Up * (NdcY * OrthoWidth * 0.5 * (double(Height) / double(Width)));
	}

	static bool IntersectPixelWithPlaneOrthographic(
		const FCameraInfo& Camera, int32 Width, int32 Height, const FVector2D& Pixel,
		const FVector& PlanePoint, const FVector& PlaneNormal, double OrthoWidth,
		FVector& OutHit, double* OutDistance = nullptr)
	{
		if (Width <= 0 || Height <= 0 || OrthoWidth <= 1e-6)
		{
			return false;
		}

		const FVector RayOrigin = CameraOrthoRayOriginWithWidth(Camera, Width, Height, Pixel, OrthoWidth);
		const FVector RayDir = Camera.Forward.GetSafeNormal();
		const FVector Normal = PlaneNormal.GetSafeNormal();
		const double Denom = FVector::DotProduct(RayDir, Normal);
		if (FMath::Abs(Denom) < 1e-8)
		{
			return false;
		}

		const double T = FVector::DotProduct(PlanePoint - RayOrigin, Normal) / Denom;
		if (!FMath::IsFinite(T))
		{
			return false;
		}

		OutHit = RayOrigin + RayDir * T;
		if (OutDistance)
		{
			*OutDistance = FVector::Distance(Camera.Location, OutHit);
		}
		return FMath::IsFinite(OutHit.X) && FMath::IsFinite(OutHit.Y) && FMath::IsFinite(OutHit.Z);
	}

	static bool ProjectWorldToImageOrthographic(
		const FCameraInfo& Camera, int32 Width, int32 Height, const FVector& World,
		double OrthoWidth, FVector2D& OutPixel)
	{
		if (Width <= 0 || Height <= 0 || OrthoWidth <= 1e-6)
		{
			return false;
		}

		const FVector Delta = World - Camera.Location;
		const double HalfWidth = OrthoWidth * 0.5;
		const double HalfHeight = HalfWidth * (double(Height) / double(Width));
		if (FMath::Abs(HalfWidth) < 1e-8 || FMath::Abs(HalfHeight) < 1e-8)
		{
			return false;
		}

		const double NdcX = FVector::DotProduct(Delta, Camera.Right) / HalfWidth;
		const double NdcY = FVector::DotProduct(Delta, Camera.Up) / HalfHeight;
		OutPixel = FVector2D(
			((NdcX + 1.0) * 0.5 * double(Width)) - 0.5,
			((1.0 - NdcY) * 0.5 * double(Height)) - 0.5);
		return FMath::IsFinite(OutPixel.X) && FMath::IsFinite(OutPixel.Y);
	}

	static bool OrthographicPixelsPerWorldAlongDirection(
		const FCameraInfo& Camera, int32 Width, double OrthoWidth,
		const FVector& DirectionWorld, FVector2D& OutPixelsPerWorld)
	{
		if (Width <= 0 || OrthoWidth <= 1e-6)
		{
			return false;
		}

		const FVector Direction = DirectionWorld.GetSafeNormal();
		if (Direction.IsNearlyZero())
		{
			return false;
		}

		const double Scale = double(Width) / OrthoWidth;
		OutPixelsPerWorld = FVector2D(
			FVector::DotProduct(Direction, Camera.Right) * Scale,
			-FVector::DotProduct(Direction, Camera.Up) * Scale);
		return OutPixelsPerWorld.SizeSquared() > 1e-12 &&
			FMath::IsFinite(OutPixelsPerWorld.X) &&
			FMath::IsFinite(OutPixelsPerWorld.Y);
	}

	static bool ProjectWorldDirectionToImageOrthographic(
		const FCameraInfo& Camera, int32 Width, const FVector& DirectionWorld,
		double OrthoWidth, FVector2D& OutDirection)
	{
		FVector2D PixelsPerWorld;
		if (!OrthographicPixelsPerWorldAlongDirection(Camera, Width, OrthoWidth, DirectionWorld, PixelsPerWorld))
		{
			return false;
		}
		OutDirection = PixelsPerWorld.GetSafeNormal();
		return true;
	}

	static void CameraRayForPixel(
		const FCameraInfo& Camera, int32 Width, int32 Height, const FVector2D& Pixel,
		FVector& OutOrigin, FVector& OutDirection)
	{
		OutOrigin = Camera.bOrthographic ? CameraOrthoRayOrigin(Camera, Width, Height, Pixel) : Camera.Location;
		OutDirection = Camera.bOrthographic ? Camera.Forward : CameraRayDirection(Camera, Width, Height, Pixel);
		OutDirection = OutDirection.GetSafeNormal();
	}

	static bool IntersectPixelWithPlane(
		const FCameraInfo& Camera, int32 Width, int32 Height,
		const FVector2D& Pixel, const FVector& PlanePoint, const FVector& PlaneNormal,
		FVector& OutHit, double* OutDistance = nullptr)
	{
		FVector RayOrigin;
		FVector RayDir;
		CameraRayForPixel(Camera, Width, Height, Pixel, RayOrigin, RayDir);
		const FVector Normal = PlaneNormal.GetSafeNormal();
		const double Denom = FVector::DotProduct(RayDir, Normal);
		if (FMath::Abs(Denom) < 1e-8)
		{
			return false;
		}

		const double T = FVector::DotProduct(PlanePoint - RayOrigin, Normal) / Denom;
		if (!FMath::IsFinite(T))
		{
			return false;
		}
		OutHit = RayOrigin + RayDir * T;
		if (OutDistance)
		{
			*OutDistance = FVector::Distance(Camera.Location, OutHit);
		}
		return FMath::IsFinite(OutHit.X) && FMath::IsFinite(OutHit.Y) && FMath::IsFinite(OutHit.Z);
	}

	static bool ProjectWorldToImage(const FCameraInfo& Camera, int32 Width, int32 Height, const FVector& World, FVector2D& OutPixel)
	{
		if (Width <= 0 || Height <= 0)
		{
			return false;
		}

		const FVector Delta = World - Camera.Location;
		double NdcX = 0.0;
		double NdcY = 0.0;
		if (Camera.bOrthographic)
		{
			const double HalfWidth = Camera.OrthoWidth * 0.5;
			const double HalfHeight = HalfWidth * (double(Height) / double(Width));
			if (FMath::Abs(HalfWidth) < 1e-8 || FMath::Abs(HalfHeight) < 1e-8)
			{
				return false;
			}
			NdcX = FVector::DotProduct(Delta, Camera.Right) / HalfWidth;
			NdcY = FVector::DotProduct(Delta, Camera.Up) / HalfHeight;
		}
		else
		{
			const double Depth = FVector::DotProduct(Delta, Camera.Forward);
			if (Depth <= 1e-6)
			{
				return false;
			}

			const double TanX = FMath::Tan(FMath::DegreesToRadians(Camera.Fov * 0.5));
			const double TanY = TanX * (double(Height) / double(Width));
			if (FMath::Abs(TanX) < 1e-8 || FMath::Abs(TanY) < 1e-8)
			{
				return false;
			}

			NdcX = FVector::DotProduct(Delta, Camera.Right) / (Depth * TanX);
			NdcY = FVector::DotProduct(Delta, Camera.Up) / (Depth * TanY);
		}

		OutPixel = FVector2D(
			((NdcX + 1.0) * 0.5 * double(Width)) - 0.5,
			((1.0 - NdcY) * 0.5 * double(Height)) - 0.5);
		return FMath::IsFinite(OutPixel.X) && FMath::IsFinite(OutPixel.Y);
	}

	static bool IntersectMaskCentroidWithFacePlane(
		const FCameraInfo& Camera, int32 Width, int32 Height,
		const FVector2D& Pixel, const FFaceInfo& Face, FVector& OutHit, double& OutDistance)
	{
		const double OrthoWidth = ResolveStep10OrthoWidth(Camera, Face.PlanePoint);
		return IntersectPixelWithPlaneOrthographic(
			Camera, Width, Height, Pixel, Face.PlanePoint, Face.Normal, OrthoWidth, OutHit, &OutDistance);
	}

	static double FaceWorldExtent(const FFaceInfo& Face)
	{
		if (Face.KeyPoints3D.Num() == 0)
		{
			return 0.0;
		}

		FVector Min = Face.KeyPoints3D[0];
		FVector Max = Face.KeyPoints3D[0];
		for (const FVector& P : Face.KeyPoints3D)
		{
			Min.X = FMath::Min(Min.X, P.X);
			Min.Y = FMath::Min(Min.Y, P.Y);
			Min.Z = FMath::Min(Min.Z, P.Z);
			Max.X = FMath::Max(Max.X, P.X);
			Max.Y = FMath::Max(Max.Y, P.Y);
			Max.Z = FMath::Max(Max.Z, P.Z);
		}
		return (Max - Min).Size();
	}

	static bool ProjectFaceNormalToImage(
		const FCameraInfo& Camera, int32 Width, int32 Height,
		const FFaceInfo& Face, const FVector& AnchorWorld, FVector2D& OutDirection)
	{
		const double OrthoWidth = ResolveStep10OrthoWidth(Camera, AnchorWorld);
		return ProjectWorldDirectionToImageOrthographic(Camera, Width, Face.Normal, OrthoWidth, OutDirection);
	}

	static bool ProjectSignedWorldDirectionToImage(
		const FCameraInfo& Camera, int32 Width, int32 Height,
		const FVector& AnchorWorld, const FVector& DirectionWorld, double ProbeLength,
		FVector2D& OutDirection)
	{
		const double OrthoWidth = ResolveStep10OrthoWidth(Camera, AnchorWorld);
		return ProjectWorldDirectionToImageOrthographic(Camera, Width, DirectionWorld, OrthoWidth, OutDirection);
	}

	static void MapPolygonToFacesSpace(
		const TArray<FVector2D>& InPoly, double ScaleX, double ScaleY, TArray<FVector2D>& OutPoly)
	{
		OutPoly.Reset();
		OutPoly.Reserve(InPoly.Num());
		for (const FVector2D& P : InPoly)
		{
			OutPoly.Emplace(P.X * ScaleX, P.Y * ScaleY);
		}
	}

	static void RemoveClosingDuplicatePair(TArray<FVector2D>& Source, TArray<FVector2D>& Copied)
	{
		if (Source.Num() < 2 || Source.Num() != Copied.Num())
		{
			return;
		}
		if ((Source[0] - Source.Last()).SizeSquared() <= 1e-6 &&
			(Copied[0] - Copied.Last()).SizeSquared() <= 1e-6)
		{
			Source.RemoveAt(Source.Num() - 1);
			Copied.RemoveAt(Copied.Num() - 1);
		}
	}

	static bool BuildOppositeTranslatedPolygon(
		const TArray<FVector2D>& SourcePolygon,
		const TArray<FVector2D>& TranslatedPolygon,
		TArray<FVector2D>& OutOppositePolygon)
	{
		OutOppositePolygon.Reset();
		if (SourcePolygon.Num() != TranslatedPolygon.Num() || SourcePolygon.Num() < 3)
		{
			return false;
		}

		OutOppositePolygon.Reserve(SourcePolygon.Num());
		for (int32 i = 0; i < SourcePolygon.Num(); ++i)
		{
			const FVector2D Offset = TranslatedPolygon[i] - SourcePolygon[i];
			OutOppositePolygon.Add(SourcePolygon[i] - Offset);
		}
		return true;
	}

	static bool IsNearlyCollinear2D(const FVector2D& Prev, const FVector2D& Cur, const FVector2D& Next)
	{
		const FVector2D A = Cur - Prev;
		const FVector2D B = Next - Prev;
		const double BLen = B.Size();
		if (BLen < 1e-8)
		{
			return true;
		}

		const double PerpDist = FMath::Abs(FVector2D::CrossProduct(B, A)) / BLen;
		const double Dot = FVector2D::DotProduct(Cur - Prev, Cur - Next);
		return PerpDist <= SolidCollinearTolerancePixels && Dot <= SolidCollinearTolerancePixels * SolidCollinearTolerancePixels;
	}

	static double DistancePointToSegmentSquared2D(const FVector2D& P, const FVector2D& A, const FVector2D& B)
	{
		const FVector2D AB = B - A;
		const double LenSq = AB.SizeSquared();
		if (LenSq < 1e-12)
		{
			return (P - A).SizeSquared();
		}
		const double T = FMath::Clamp(FVector2D::DotProduct(P - A, AB) / LenSq, 0.0, 1.0);
		return (P - (A + AB * T)).SizeSquared();
	}

	static void MarkRdpSegment(const TArray<FVector2D>& Points, int32 Start, int32 End, double ToleranceSq, TArray<bool>& Keep)
	{
		TArray<TPair<int32, int32>> Stack;
		Stack.Emplace(Start, End);
		Keep[Start] = true;
		Keep[End] = true;

		while (Stack.Num() > 0)
		{
			const TPair<int32, int32> Range = Stack.Pop(EAllowShrinking::No);
			double BestDistSq = -1.0;
			int32 BestIndex = INDEX_NONE;
			for (int32 i = Range.Key + 1; i < Range.Value; ++i)
			{
				const double DistSq = DistancePointToSegmentSquared2D(Points[i], Points[Range.Key], Points[Range.Value]);
				if (DistSq > BestDistSq)
				{
					BestDistSq = DistSq;
					BestIndex = i;
				}
			}

			if (BestIndex != INDEX_NONE && BestDistSq > ToleranceSq)
			{
				Keep[BestIndex] = true;
				Stack.Emplace(Range.Key, BestIndex);
				Stack.Emplace(BestIndex, Range.Value);
			}
		}
	}

	static void SimplifyClosedLoopRdpPairs(TArray<FVector2D>& Source, TArray<FVector2D>& Copied)
	{
		const int32 N = Source.Num();
		if (N <= SolidTargetMaxLoopPoints || N != Copied.Num())
		{
			return;
		}

		int32 Split = 0;
		double BestDistSq = -1.0;
		for (int32 i = 1; i < N; ++i)
		{
			const double DistSq = (Source[i] - Source[0]).SizeSquared();
			if (DistSq > BestDistSq)
			{
				BestDistSq = DistSq;
				Split = i;
			}
		}
		if (Split <= 0 || Split >= N - 1)
		{
			return;
		}

		TArray<bool> Keep;
		Keep.Init(false, N);
		const double ToleranceSq = SolidRdpTolerancePixels * SolidRdpTolerancePixels;
		MarkRdpSegment(Source, 0, Split, ToleranceSq, Keep);
		MarkRdpSegment(Source, Split, N - 1, ToleranceSq, Keep);
		Keep[0] = true;
		Keep[Split] = true;

		TArray<FVector2D> NewSource;
		TArray<FVector2D> NewCopied;
		NewSource.Reserve(N);
		NewCopied.Reserve(N);
		for (int32 i = 0; i < N; ++i)
		{
			if (Keep[i])
			{
				NewSource.Add(Source[i]);
				NewCopied.Add(Copied[i]);
			}
		}

		if (NewSource.Num() >= 3)
		{
			Source = MoveTemp(NewSource);
			Copied = MoveTemp(NewCopied);
		}
	}

	static void DecimateLoopPairsToTarget(TArray<FVector2D>& Source, TArray<FVector2D>& Copied)
	{
		if (Source.Num() <= SolidTargetMaxLoopPoints || Source.Num() != Copied.Num())
		{
			return;
		}

		const int32 N = Source.Num();
		TArray<FVector2D> NewSource;
		TArray<FVector2D> NewCopied;
		NewSource.Reserve(SolidTargetMaxLoopPoints);
		NewCopied.Reserve(SolidTargetMaxLoopPoints);
		int32 LastIndex = INDEX_NONE;
		for (int32 k = 0; k < SolidTargetMaxLoopPoints; ++k)
		{
			const int32 Index = FMath::Clamp(FMath::RoundToInt(double(k) * double(N) / double(SolidTargetMaxLoopPoints)), 0, N - 1);
			if (Index == LastIndex)
			{
				continue;
			}
			NewSource.Add(Source[Index]);
			NewCopied.Add(Copied[Index]);
			LastIndex = Index;
		}
		if (NewSource.Num() >= 3)
		{
			Source = MoveTemp(NewSource);
			Copied = MoveTemp(NewCopied);
		}
	}

	static void SimplifyLoopPairs(TArray<FVector2D>& Source, TArray<FVector2D>& Copied)
	{
		if (Source.Num() != Copied.Num())
		{
			return;
		}

		RemoveClosingDuplicatePair(Source, Copied);

		for (int32 i = Source.Num() - 1; i >= 0 && Source.Num() > 3; --i)
		{
			const int32 Prev = (i + Source.Num() - 1) % Source.Num();
			if ((Source[i] - Source[Prev]).SizeSquared() <= 1e-6 &&
				(Copied[i] - Copied[Prev]).SizeSquared() <= 1e-6)
			{
				Source.RemoveAt(i);
				Copied.RemoveAt(i);
			}
		}

		bool bRemoved = true;
		int32 Safety = 0;
		while (bRemoved && Source.Num() > 3 && ++Safety < 64)
		{
			bRemoved = false;
			for (int32 i = 0; i < Source.Num() && Source.Num() > 3; ++i)
			{
				const int32 Prev = (i + Source.Num() - 1) % Source.Num();
				const int32 Next = (i + 1) % Source.Num();
				if (IsNearlyCollinear2D(Source[Prev], Source[i], Source[Next]) &&
					IsNearlyCollinear2D(Copied[Prev], Copied[i], Copied[Next]))
				{
					Source.RemoveAt(i);
					Copied.RemoveAt(i);
					bRemoved = true;
					--i;
				}
			}
		}

		SimplifyClosedLoopRdpPairs(Source, Copied);
		DecimateLoopPairsToTarget(Source, Copied);
	}

	// Orthographic closed-form extrusion depth:
	// source_to_copied_2d = depth * projected_normal_pixels_per_world.
	static bool SolveExtrusionDepthOrthographic(
		const FCameraInfo& Camera, int32 Width,
		const FVector& OrientedNormal,
		const FVector& AnchorWorld,
		const FVector2D& SourceToCopiedVector2D,
		double& OutDepth,
		FString& OutError)
	{
		const double OrthoWidth = ResolveStep10OrthoWidth(Camera, AnchorWorld);
		FVector2D PixelsPerWorld;
		if (!OrthographicPixelsPerWorldAlongDirection(Camera, Width, OrthoWidth, OrientedNormal, PixelsPerWorld))
		{
			OutError = TEXT("base face normal has near-zero orthographic image projection; extrusion depth is not observable");
			return false;
		}

		const double ScaleDenom = FVector2D::DotProduct(PixelsPerWorld, PixelsPerWorld);
		if (ScaleDenom < 1e-12)
		{
			OutError = TEXT("orthographic normal image scale is too small");
			return false;
		}

		const double Depth = FVector2D::DotProduct(PixelsPerWorld, SourceToCopiedVector2D) / ScaleDenom;
		if (!FMath::IsFinite(Depth) || Depth <= 1e-4)
		{
			OutError = TEXT("orthographic extrusion depth is not positive");
			return false;
		}

		OutDepth = Depth;
		return true;
	}

	static bool PointInTriangle2D(const FVector2D& P, const FVector2D& A, const FVector2D& B, const FVector2D& C)
	{
		const double D1 = (P.X - B.X) * (A.Y - B.Y) - (A.X - B.X) * (P.Y - B.Y);
		const double D2 = (P.X - C.X) * (B.Y - C.Y) - (B.X - C.X) * (P.Y - C.Y);
		const double D3 = (P.X - A.X) * (C.Y - A.Y) - (C.X - A.X) * (P.Y - A.Y);
		const bool bHasNeg = (D1 < -1e-8) || (D2 < -1e-8) || (D3 < -1e-8);
		const bool bHasPos = (D1 > 1e-8) || (D2 > 1e-8) || (D3 > 1e-8);
		return !(bHasNeg && bHasPos);
	}

	static bool TriangulatePolygon2D(const TArray<FVector2D>& Poly, TArray<int32>& OutTriangles)
	{
		OutTriangles.Reset();
		const int32 N = Poly.Num();
		if (N < 3)
		{
			return false;
		}
		if (N == 3)
		{
			OutTriangles = { 0, 1, 2 };
			return true;
		}

		TArray<int32> Indices;
		Indices.Reserve(N);
		for (int32 i = 0; i < N; ++i)
		{
			Indices.Add(i);
		}

		const double Orient = PolygonArea2D(Poly) >= 0.0 ? 1.0 : -1.0;
		int32 Safety = 0;
		while (Indices.Num() > 3 && ++Safety < N * N)
		{
			bool bClipped = false;
			for (int32 i = 0; i < Indices.Num(); ++i)
			{
				const int32 Prev = Indices[(i + Indices.Num() - 1) % Indices.Num()];
				const int32 Cur = Indices[i];
				const int32 Next = Indices[(i + 1) % Indices.Num()];

				const FVector2D A = Poly[Prev];
				const FVector2D B = Poly[Cur];
				const FVector2D C = Poly[Next];
				const double Cross = FVector2D::CrossProduct(B - A, C - B);
				if (Cross * Orient <= 1e-8)
				{
					continue;
				}

				bool bContainsOther = false;
				for (int32 TestIdx : Indices)
				{
					if (TestIdx == Prev || TestIdx == Cur || TestIdx == Next)
					{
						continue;
					}
					if (PointInTriangle2D(Poly[TestIdx], A, B, C))
					{
						bContainsOther = true;
						break;
					}
				}
				if (bContainsOther)
				{
					continue;
				}

				OutTriangles.Add(Prev);
				OutTriangles.Add(Cur);
				OutTriangles.Add(Next);
				Indices.RemoveAt(i);
				bClipped = true;
				break;
			}

			if (!bClipped)
			{
				OutTriangles.Reset();
				for (int32 i = 1; i + 1 < N; ++i)
				{
					OutTriangles.Add(0);
					OutTriangles.Add(i);
					OutTriangles.Add(i + 1);
				}
				return true;
			}
		}

		if (Indices.Num() == 3)
		{
			OutTriangles.Add(Indices[0]);
			OutTriangles.Add(Indices[1]);
			OutTriangles.Add(Indices[2]);
		}

		return OutTriangles.Num() >= 3;
	}

	static FVector ComputeTriangleNormal(const TArray<FVector>& Vertices, const TArray<int32>& Triangles)
	{
		FVector N = FVector::ZeroVector;
		for (int32 i = 0; i + 2 < Triangles.Num(); i += 3)
		{
			const FVector& A = Vertices[Triangles[i]];
			const FVector& B = Vertices[Triangles[i + 1]];
			const FVector& C = Vertices[Triangles[i + 2]];
			N += FVector::CrossProduct(B - A, C - A);
		}
		return N.GetSafeNormal();
	}

	static FVector AverageVector(const TArray<FVector>& Points)
	{
		FVector Sum = FVector::ZeroVector;
		for (const FVector& P : Points)
		{
			Sum += P;
		}
		return Points.Num() > 0 ? Sum / double(Points.Num()) : FVector::ZeroVector;
	}

	static void ScaleVerticesAlongAxis(TArray<FVector>& Vertices, const FVector& Axis, double Scale)
	{
		const FVector UnitAxis = Axis.GetSafeNormal();
		if (Vertices.Num() == 0 || UnitAxis.IsNearlyZero() || !FMath::IsFinite(Scale) || FMath::IsNearlyEqual(Scale, 1.0))
		{
			return;
		}

		const FVector Anchor = AverageVector(Vertices);
		const double DeltaScale = Scale - 1.0;
		for (FVector& Vertex : Vertices)
		{
			const double AxisDistance = FVector::DotProduct(Vertex - Anchor, UnitAxis);
			Vertex += UnitAxis * (AxisDistance * DeltaScale);
		}
	}

	static FVector2D AverageVector2DDelta(const TArray<FVector2D>& Source, const TArray<FVector2D>& Copied)
	{
		FVector2D Sum = FVector2D::ZeroVector;
		const int32 Count = FMath::Min(Source.Num(), Copied.Num());
		for (int32 i = 0; i < Count; ++i)
		{
			Sum += Copied[i] - Source[i];
		}
		return Count > 0 ? Sum / double(Count) : FVector2D::ZeroVector;
	}

	static void DrawPointRGBA(TArray<uint8>& RGBA, int32 Width, int32 Height, int32 X, int32 Y, const FColor& Color, int32 Radius)
	{
		for (int32 Dy = -Radius; Dy <= Radius; ++Dy)
		{
			for (int32 Dx = -Radius; Dx <= Radius; ++Dx)
			{
				const int32 PX = X + Dx;
				const int32 PY = Y + Dy;
				if (PX < 0 || PY < 0 || PX >= Width || PY >= Height)
				{
					continue;
				}
				const int32 Off = (PY * Width + PX) * 4;
				RGBA[Off + 0] = Color.R;
				RGBA[Off + 1] = Color.G;
				RGBA[Off + 2] = Color.B;
				RGBA[Off + 3] = 255;
			}
		}
	}

	static void DrawLineRGBA(TArray<uint8>& RGBA, int32 Width, int32 Height, const FVector2D& A, const FVector2D& B, const FColor& Color, int32 Radius)
	{
		const double Dx = B.X - A.X;
		const double Dy = B.Y - A.Y;
		const int32 Steps = FMath::Max(1, FMath::CeilToInt(FMath::Max(FMath::Abs(Dx), FMath::Abs(Dy))));
		for (int32 i = 0; i <= Steps; ++i)
		{
			const double T = double(i) / double(Steps);
			const int32 X = FMath::RoundToInt(A.X + Dx * T);
			const int32 Y = FMath::RoundToInt(A.Y + Dy * T);
			DrawPointRGBA(RGBA, Width, Height, X, Y, Color, Radius);
		}
	}

	static void DrawClosedPolylineRGBA(TArray<uint8>& RGBA, int32 Width, int32 Height, const TArray<FVector2D>& Points, const FColor& Color, int32 Radius)
	{
		if (Points.Num() < 2)
		{
			return;
		}
		for (int32 i = 0; i < Points.Num(); ++i)
		{
			DrawLineRGBA(RGBA, Width, Height, Points[i], Points[(i + 1) % Points.Num()], Color, Radius);
		}
	}

	static TSharedPtr<FJsonValue> JsonVector2D(const FVector2D& V)
	{
		TArray<TSharedPtr<FJsonValue>> Arr;
		Arr.Add(MakeShared<FJsonValueNumber>(V.X));
		Arr.Add(MakeShared<FJsonValueNumber>(V.Y));
		return MakeShared<FJsonValueArray>(Arr);
	}

	static TSharedPtr<FJsonValue> JsonVector(const FVector& V)
	{
		TArray<TSharedPtr<FJsonValue>> Arr;
		Arr.Add(MakeShared<FJsonValueNumber>(V.X));
		Arr.Add(MakeShared<FJsonValueNumber>(V.Y));
		Arr.Add(MakeShared<FJsonValueNumber>(V.Z));
		return MakeShared<FJsonValueArray>(Arr);
	}

	static TSharedPtr<FJsonValue> JsonIntTriple(int32 A, int32 B, int32 C)
	{
		TArray<TSharedPtr<FJsonValue>> Arr;
		Arr.Add(MakeShared<FJsonValueNumber>(A));
		Arr.Add(MakeShared<FJsonValueNumber>(B));
		Arr.Add(MakeShared<FJsonValueNumber>(C));
		return MakeShared<FJsonValueArray>(Arr);
	}

	static void SetVectorArrayField(TSharedRef<FJsonObject> Object, const TCHAR* Key, const TArray<FVector>& Values)
	{
		TArray<TSharedPtr<FJsonValue>> JsonValues;
		JsonValues.Reserve(Values.Num());
		for (const FVector& V : Values)
		{
			JsonValues.Add(JsonVector(V));
		}
		Object->SetArrayField(Key, JsonValues);
	}

	static void SetVector2DArrayField(TSharedRef<FJsonObject> Object, const TCHAR* Key, const TArray<FVector2D>& Values)
	{
		TArray<TSharedPtr<FJsonValue>> JsonValues;
		JsonValues.Reserve(Values.Num());
		for (const FVector2D& V : Values)
		{
			JsonValues.Add(JsonVector2D(V));
		}
		Object->SetArrayField(Key, JsonValues);
	}

	static void SetTriangleArrayField(TSharedRef<FJsonObject> Object, const TCHAR* Key, const TArray<int32>& Triangles)
	{
		TArray<TSharedPtr<FJsonValue>> JsonValues;
		for (int32 i = 0; i + 2 < Triangles.Num(); i += 3)
		{
			JsonValues.Add(JsonIntTriple(Triangles[i], Triangles[i + 1], Triangles[i + 2]));
		}
		Object->SetArrayField(Key, JsonValues);
	}

	static void SaveSolidResultJson(const FSolidReconstructionResult& Result, const FString& Path)
	{
		TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
		Root->SetStringField(TEXT("component"), Result.ComponentName);
		Root->SetStringField(TEXT("action"), Result.Action);
		Root->SetStringField(TEXT("source_polygon_key"), Result.SourcePolygonKey);
		Root->SetStringField(TEXT("copied_polygon_key"), Result.CopiedPolygonKey);
		Root->SetBoolField(TEXT("success"), Result.bSuccess);
		Root->SetStringField(TEXT("error"), Result.Error);
		Root->SetStringField(TEXT("warning"), Result.Warning);
		Root->SetStringField(TEXT("actor_name"), Result.ActorName);
		Root->SetNumberField(TEXT("cap_width"), Result.CapWidth);
		Root->SetNumberField(TEXT("cap_height"), Result.CapHeight);
		Root->SetNumberField(TEXT("faces_width"), Result.FacesWidth);
		Root->SetNumberField(TEXT("faces_height"), Result.FacesHeight);
		Root->SetNumberField(TEXT("selected_source_face_id"), Result.SelectedFaceId);
		Root->SetArrayField(TEXT("source_plane_point"), JsonVector(Result.SourcePlanePoint)->AsArray());
		Root->SetArrayField(TEXT("source_plane_normal"), JsonVector(Result.SourcePlaneNormal)->AsArray());
		Root->SetArrayField(TEXT("oriented_normal_source_to_copied"), JsonVector(Result.OrientedNormal)->AsArray());
		Root->SetArrayField(TEXT("source_to_copied_vector_2d"), JsonVector2D(Result.SourceToCopiedVector2D)->AsArray());
		Root->SetArrayField(TEXT("projected_oriented_normal_2d"), JsonVector2D(Result.ProjectedNormal2D)->AsArray());
		Root->SetNumberField(TEXT("extrusion_depth"), Result.ExtrusionDepth);
		Root->SetNumberField(TEXT("max_depth_sample_reprojection_error_pixels"), Result.MaxDepthSampleReprojectionErrorPixels);
		Root->SetNumberField(TEXT("mean_copied_reprojection_error_pixels"), Result.MeanCopiedReprojectionErrorPixels);
		Root->SetNumberField(TEXT("max_copied_reprojection_error_pixels"), Result.MaxCopiedReprojectionErrorPixels);

		TArray<TSharedPtr<FJsonValue>> SampleValues;
		for (const FSolidDepthSample& Sample : Result.DepthSamples)
		{
			TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
			Obj->SetNumberField(TEXT("index"), Sample.Index);
			Obj->SetBoolField(TEXT("valid"), Sample.bValid);
			Obj->SetStringField(TEXT("error"), Sample.Error);
			Obj->SetArrayField(TEXT("source_pixel_2d"), JsonVector2D(Sample.SourcePixel)->AsArray());
			Obj->SetArrayField(TEXT("copied_pixel_2d"), JsonVector2D(Sample.CopiedPixel)->AsArray());
			Obj->SetArrayField(TEXT("source_world"), JsonVector(Sample.SourceWorld)->AsArray());
			Obj->SetArrayField(TEXT("point_on_extrusion"), JsonVector(Sample.PointOnExtrusion)->AsArray());
			Obj->SetArrayField(TEXT("point_on_copied_ray"), JsonVector(Sample.PointOnRay)->AsArray());
			Obj->SetNumberField(TEXT("depth"), Sample.Depth);
			Obj->SetNumberField(TEXT("closest_world_distance"), Sample.ClosestWorldDistance);
			Obj->SetNumberField(TEXT("reprojection_error_pixels"), Sample.ReprojectionErrorPixels);
			SampleValues.Add(MakeShared<FJsonValueObject>(Obj));
		}
		Root->SetArrayField(TEXT("depth_samples"), SampleValues);

		SetVector2DArrayField(Root, TEXT("source_loop_2d"), Result.SourceLoop2D);
		SetVector2DArrayField(Root, TEXT("copied_target_loop_2d"), Result.CopiedTargetLoop2D);
		SetVector2DArrayField(Root, TEXT("reprojected_source_loop_2d"), Result.ReprojectedSourceLoop2D);
		SetVector2DArrayField(Root, TEXT("reprojected_copied_loop_2d"), Result.ReprojectedCopiedLoop2D);
		SetVectorArrayField(Root, TEXT("source_loop_world"), Result.SourceLoopWorld);
		SetVectorArrayField(Root, TEXT("copied_loop_world"), Result.CopiedLoopWorld);
		SetVectorArrayField(Root, TEXT("mesh_vertices_world"), Result.MeshVerticesWorld);
		SetTriangleArrayField(Root, TEXT("mesh_triangles"), Result.MeshTriangles);

		SaveJsonObject(Root, Path);
	}

	static bool SaveSolidProjectionCheckPng(
		const FSolidReconstructionResult& Result,
		const TArray<uint8>& FacesRGBA, int32 Width, int32 Height, const FString& Path)
	{
		if (FacesRGBA.Num() < Width * Height * 4 || !Result.bSuccess)
		{
			return false;
		}

		TArray<uint8> RGBA = FacesRGBA;
		DrawClosedPolylineRGBA(RGBA, Width, Height, Result.SourceLoop2D, FColor(0, 180, 255, 255), 1);
		DrawClosedPolylineRGBA(RGBA, Width, Height, Result.CopiedTargetLoop2D, FColor(255, 140, 0, 255), 1);
		DrawClosedPolylineRGBA(RGBA, Width, Height, Result.ReprojectedSourceLoop2D, FColor(0, 255, 80, 255), 2);
		DrawClosedPolylineRGBA(RGBA, Width, Height, Result.ReprojectedCopiedLoop2D, FColor(255, 0, 180, 255), 2);
		return SaveRGBAToPng(RGBA, Width, Height, Path);
	}

	static void DrawArrowRGBA(TArray<uint8>& RGBA, int32 Width, int32 Height, const FVector2D& A, const FVector2D& B, const FColor& Color, int32 Radius)
	{
		DrawLineRGBA(RGBA, Width, Height, A, B, Color, Radius);
		FVector2D Dir = B - A;
		if (Dir.SizeSquared() < 1e-6)
		{
			return;
		}
		Dir.Normalize();
		const FVector2D Perp(-Dir.Y, Dir.X);
		const double HeadLen = 20.0;
		const double HeadHalf = 10.0;
		const FVector2D Base = B - Dir * HeadLen;
		DrawLineRGBA(RGBA, Width, Height, B, Base + Perp * HeadHalf, Color, Radius);
		DrawLineRGBA(RGBA, Width, Height, B, Base - Perp * HeadHalf, Color, Radius);
	}

	// Debug overlay: the cap-connected green lines (green) and every candidate face's
	// projected normal (drawn from the candidate's mask centroid). Normal arrow color:
	// bright-green = selected source face, cyan = passed the normal-parallel filter,
	// magenta = candidate that failed the filter. The faint gray loop is the cap mask.
	static bool SaveNormalGreenCheckPng(
		const TArray<uint8>& FacesRGBA, int32 Width, int32 Height,
		const TArray<FVector2D>& CapLoopFaceSpace,
		const TArray<FVector2D>& GreenStarts, const TArray<FVector2D>& GreenEnds,
		const TArray<FFaceCandidate>& Candidates, int32 SelectedFaceId,
		const FString& Path)
	{
		if (FacesRGBA.Num() < Width * Height * 4)
		{
			return false;
		}
		TArray<uint8> RGBA = FacesRGBA;

		DrawClosedPolylineRGBA(RGBA, Width, Height, CapLoopFaceSpace, FColor(140, 140, 140, 255), 1);

		for (int32 i = 0; i < GreenStarts.Num(); ++i)
		{
			DrawArrowRGBA(RGBA, Width, Height, GreenStarts[i], GreenEnds[i], FColor(0, 190, 0, 255), 2);
		}

		const double NormalArrowLen = 110.0;
		for (const FFaceCandidate& C : Candidates)
		{
			if (!C.bHasProjectedNormal)
			{
				continue;
			}
			FColor Col = FColor(230, 0, 180, 255);              // candidate, failed filter
			if (C.FaceId == SelectedFaceId) { Col = FColor(40, 230, 80, 255); }   // selected
			else if (C.bNormalParallelPass) { Col = FColor(0, 200, 255, 255); }   // passed filter
			const FVector2D Tip = C.MaskCentroid + C.ProjectedNormal2D.GetSafeNormal() * NormalArrowLen;
			DrawArrowRGBA(RGBA, Width, Height, C.MaskCentroid, Tip, Col, 2);
		}

		return SaveRGBAToPng(RGBA, Width, Height, Path);
	}

	static void SaveComponentResultJson(const FComponentResult& Result, const FString& Path)
	{
		TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
		Root->SetStringField(TEXT("component"), Result.ComponentName);
		Root->SetStringField(TEXT("action"), Result.Action);
		Root->SetStringField(TEXT("polygon_key"), Result.PolygonKey);
		Root->SetBoolField(TEXT("success"), Result.bSuccess);
		Root->SetStringField(TEXT("error"), Result.Error);
		Root->SetStringField(TEXT("actor_name"), Result.ActorName);
		Root->SetNumberField(TEXT("cap_width"), Result.CapWidth);
		Root->SetNumberField(TEXT("cap_height"), Result.CapHeight);
		Root->SetNumberField(TEXT("faces_width"), Result.FacesWidth);
		Root->SetNumberField(TEXT("faces_height"), Result.FacesHeight);
		Root->SetNumberField(TEXT("cap_mask_pixels"), Result.CapMaskPixels);
		Root->SetNumberField(TEXT("min_overlap_pixels"), Result.MinOverlapPixels);
		Root->SetArrayField(TEXT("green_line_vector_2d"), JsonVector2D(Result.GreenLineVector2D)->AsArray());
		{
			TArray<TSharedPtr<FJsonValue>> GreenVectors;
			for (const FVector2D& V : Result.GreenLineVectors2D)
			{
				GreenVectors.Add(JsonVector2D(V));
			}
			Root->SetArrayField(TEXT("green_line_vectors_2d"), GreenVectors);
		}
		Root->SetNumberField(TEXT("normal_parallel_threshold_degrees"), NormalParallelThresholdDegrees);
		Root->SetNumberField(TEXT("selected_face_id"), Result.SelectedFaceId);
		Root->SetArrayField(TEXT("selected_plane_hit_3d"), JsonVector(Result.SelectedPlaneHit)->AsArray());

		TArray<TSharedPtr<FJsonValue>> CandidateValues;
		for (const FFaceCandidate& Candidate : Result.Candidates)
		{
			TSharedRef<FJsonObject> Obj = MakeShared<FJsonObject>();
			Obj->SetNumberField(TEXT("face_id"), Candidate.FaceId);
			Obj->SetNumberField(TEXT("overlap_pixels"), Candidate.OverlapPixels);
			Obj->SetNumberField(TEXT("overlap_ratio"), Candidate.OverlapRatio);
			Obj->SetArrayField(TEXT("mask_centroid_2d"), JsonVector2D(Candidate.MaskCentroid)->AsArray());
			Obj->SetBoolField(TEXT("has_plane_hit"), Candidate.bHasPlaneHit);
			Obj->SetArrayField(TEXT("plane_hit_3d"), JsonVector(Candidate.PlaneHit)->AsArray());
			Obj->SetNumberField(TEXT("distance_to_camera"), Candidate.DistanceToCamera);
			Obj->SetBoolField(TEXT("has_projected_normal"), Candidate.bHasProjectedNormal);
			Obj->SetArrayField(TEXT("projected_normal_2d"), JsonVector2D(Candidate.ProjectedNormal2D)->AsArray());
			Obj->SetNumberField(TEXT("normal_green_angle_degrees"), Candidate.NormalGreenAngleDegrees);
			Obj->SetBoolField(TEXT("normal_parallel_pass"), Candidate.bNormalParallelPass);
			CandidateValues.Add(MakeShared<FJsonValueObject>(Obj));
		}
		Root->SetArrayField(TEXT("candidates"), CandidateValues);

		TArray<TSharedPtr<FJsonValue>> VertexValues;
		for (const FVector& V : Result.MeshVerticesWorld)
		{
			VertexValues.Add(JsonVector(V));
		}
		Root->SetArrayField(TEXT("mesh_vertices_world"), VertexValues);

		TArray<TSharedPtr<FJsonValue>> TriangleValues;
		for (int32 i = 0; i + 2 < Result.MeshTriangles.Num(); i += 3)
		{
			TriangleValues.Add(JsonIntTriple(Result.MeshTriangles[i], Result.MeshTriangles[i + 1], Result.MeshTriangles[i + 2]));
		}
		Root->SetArrayField(TEXT("mesh_triangles"), TriangleValues);

		SaveJsonObject(Root, Path);
	}

	static FComponentResult MakeFailureResult(const FString& ComponentName, const FString& Error, const FString& OutputDir)
	{
		FComponentResult Result;
		Result.ComponentName = ComponentName;
		Result.Error = Error;
		SaveComponentResultJson(Result, OutputDir / TEXT("10_face_reconstruction.json"));
		return Result;
	}

	static bool LoadAction(const FString& Path, FString& OutAction)
	{
		TSharedPtr<FJsonObject> Root;
		if (!LoadJsonObject(Path, Root))
		{
			return false;
		}
		return Root->TryGetStringField(TEXT("action"), OutAction) && !OutAction.IsEmpty();
	}

	static void InitializeSolidResult(
		FSolidReconstructionResult& Solid, const FString& ComponentName, const FString& PressDir,
		const FCommonInputs& Inputs)
	{
		Solid.ComponentName = ComponentName;
		Solid.FacesWidth = Inputs.FacesWidth;
		Solid.FacesHeight = Inputs.FacesHeight;
		Solid.ActorName = FString::Printf(TEXT("FromLZ_ReconstructedSolid_%s_%s"), *FPaths::GetCleanFilename(PressDir), *ComponentName);
	}

	static void SaveSkippedSolidResult(
		FSolidReconstructionResult& Solid, const FString& Path, const FString& Error)
	{
		Solid.bSuccess = false;
		Solid.Error = Error;
		SaveSolidResultJson(Solid, Path);
	}

	static bool BuildSolidMeshTriangles(
		TArray<FVector2D>& SourceLoop2D,
		TArray<FVector2D>& CopiedTargetLoop2D,
		TArray<FVector>& SourceLoopWorld,
		TArray<FVector>& CopiedLoopWorld,
		const FVector& OrientedNormal,
		TArray<FVector>& OutVertices,
		TArray<int32>& OutTriangles,
		FVector& OutMeshNormal)
	{
		OutVertices.Reset();
		OutTriangles.Reset();
		OutMeshNormal = OrientedNormal.GetSafeNormal();

		TArray<int32> SourceTriangles;
		if (!TriangulatePolygon2D(SourceLoop2D, SourceTriangles))
		{
			return false;
		}

		FVector SourceTriNormal = ComputeTriangleNormal(SourceLoopWorld, SourceTriangles);
		if (!SourceTriNormal.IsNearlyZero() && FVector::DotProduct(SourceTriNormal, OrientedNormal) > 0.0)
		{
			Algo::Reverse(SourceLoop2D);
			Algo::Reverse(CopiedTargetLoop2D);
			Algo::Reverse(SourceLoopWorld);
			Algo::Reverse(CopiedLoopWorld);
			SourceTriangles.Reset();
			if (!TriangulatePolygon2D(SourceLoop2D, SourceTriangles))
			{
				return false;
			}
		}

		const int32 N = SourceLoopWorld.Num();
		if (N < 3 || CopiedLoopWorld.Num() != N)
		{
			return false;
		}

		OutVertices.Reserve(N * 2);
		for (const FVector& V : SourceLoopWorld)
		{
			OutVertices.Add(V);
		}
		for (const FVector& V : CopiedLoopWorld)
		{
			OutVertices.Add(V);
		}

		for (int32 i = 0; i + 2 < SourceTriangles.Num(); i += 3)
		{
			OutTriangles.Add(SourceTriangles[i]);
			OutTriangles.Add(SourceTriangles[i + 1]);
			OutTriangles.Add(SourceTriangles[i + 2]);
		}

		for (int32 i = 0; i + 2 < SourceTriangles.Num(); i += 3)
		{
			OutTriangles.Add(N + SourceTriangles[i]);
			OutTriangles.Add(N + SourceTriangles[i + 2]);
			OutTriangles.Add(N + SourceTriangles[i + 1]);
		}

		for (int32 i = 0; i < N; ++i)
		{
			const int32 J = (i + 1) % N;
			OutTriangles.Add(i);
			OutTriangles.Add(N + i);
			OutTriangles.Add(N + J);

			OutTriangles.Add(i);
			OutTriangles.Add(N + J);
			OutTriangles.Add(J);
		}

		return OutVertices.Num() >= 6 && OutTriangles.Num() >= 12;
	}

	static FSolidReconstructionResult BuildSolidReconstruction(
		const FString& ComponentName,
		const FString& Action,
		const FString& SourcePolygonKey,
		const FString& CopiedPolygonKey,
		const FString& PressDir,
		int32 CapWidth,
		int32 CapHeight,
		double ScaleX,
		double ScaleY,
		const TArray<FVector2D>& SourcePolygonCapSpace,
		const TArray<FVector2D>& CopiedPolygonCapSpace,
		const FFaceInfo& SelectedFace,
		const FCommonInputs& Inputs)
	{
		FSolidReconstructionResult Result;
		InitializeSolidResult(Result, ComponentName, PressDir, Inputs);
		Result.Action = Action;
		Result.SourcePolygonKey = SourcePolygonKey;
		Result.CopiedPolygonKey = CopiedPolygonKey;
		Result.CapWidth = CapWidth;
		Result.CapHeight = CapHeight;
		Result.SelectedFaceId = SelectedFace.Id;
		Result.SourcePlanePoint = SelectedFace.PlanePoint;
		Result.SourcePlaneNormal = SelectedFace.Normal.GetSafeNormal();
		Result.OrientedNormal = Result.SourcePlaneNormal;

		if (SourcePolygonCapSpace.Num() != CopiedPolygonCapSpace.Num())
		{
			Result.Error = FString::Printf(
				TEXT("source/copy polygon point counts differ: %d vs %d"),
				SourcePolygonCapSpace.Num(), CopiedPolygonCapSpace.Num());
			return Result;
		}
		if (SourcePolygonCapSpace.Num() < 3)
		{
			Result.Error = TEXT("source/copy polygons need at least three points");
			return Result;
		}

		MapPolygonToFacesSpace(SourcePolygonCapSpace, ScaleX, ScaleY, Result.SourceLoop2D);
		MapPolygonToFacesSpace(CopiedPolygonCapSpace, ScaleX, ScaleY, Result.CopiedTargetLoop2D);
		SimplifyLoopPairs(Result.SourceLoop2D, Result.CopiedTargetLoop2D);
		if (Result.SourceLoop2D.Num() != Result.CopiedTargetLoop2D.Num() || Result.SourceLoop2D.Num() < 3)
		{
			Result.Error = TEXT("source/copy polygons became invalid after duplicate/collinear cleanup");
			return Result;
		}

		Result.SourceToCopiedVector2D = AverageVector2DDelta(Result.SourceLoop2D, Result.CopiedTargetLoop2D);
		if (Result.SourceToCopiedVector2D.SizeSquared() < 1e-8)
		{
			Result.Error = TEXT("source-to-copied 2D offset is too short");
			return Result;
		}

		const double Step10OrthoWidth = ResolveStep10OrthoWidth(Inputs.Camera, SelectedFace.PlanePoint);
		if (Step10OrthoWidth <= 1e-6)
		{
			Result.Error = TEXT("failed to resolve orthographic width for Step 10 solid reconstruction");
			return Result;
		}

		Result.SourceLoopWorld.Reserve(Result.SourceLoop2D.Num());
		for (const FVector2D& P : Result.SourceLoop2D)
		{
			FVector Hit;
			if (!IntersectPixelWithPlaneOrthographic(
				Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight,
				P, SelectedFace.PlanePoint, SelectedFace.Normal, Step10OrthoWidth, Hit))
			{
				Result.Error = TEXT("failed to project a source cap point onto the selected source face plane");
				return Result;
			}
			Result.SourceLoopWorld.Add(Hit);
		}

		const FVector SourceAnchor = AverageVector(Result.SourceLoopWorld);
		const double ProbeLength = FMath::Clamp(FaceWorldExtent(SelectedFace) * 0.25, 10.0, 250.0);
		if (!ProjectSignedWorldDirectionToImage(
			Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight,
			SourceAnchor, Result.OrientedNormal, ProbeLength, Result.ProjectedNormal2D))
		{
			Result.Error = TEXT("source face normal projects to a near-zero image direction; cannot orient extrusion");
			return Result;
		}

		if (FVector2D::DotProduct(Result.ProjectedNormal2D, Result.SourceToCopiedVector2D.GetSafeNormal()) < 0.0)
		{
			Result.OrientedNormal *= -1.0;
			Result.ProjectedNormal2D *= -1.0;
		}

		Result.MaxDepthSampleReprojectionErrorPixels = FMath::Max(25.0, Result.SourceToCopiedVector2D.Size() * 0.75);
		if (!SolveExtrusionDepthOrthographic(
			Inputs.Camera, Inputs.FacesWidth,
			Result.OrientedNormal, SourceAnchor, Result.SourceToCopiedVector2D,
			Result.ExtrusionDepth, Result.Error))
		{
			return Result;
		}

		int32 ValidVertexCount = 0;
		Result.DepthSamples.Reserve(Result.SourceLoopWorld.Num());
		for (int32 i = 0; i < Result.SourceLoopWorld.Num(); ++i)
		{
			FSolidDepthSample Sample;
			Sample.Index = i;
			Sample.SourcePixel = Result.SourceLoop2D[i];
			Sample.CopiedPixel = Result.CopiedTargetLoop2D[i];
			Sample.SourceWorld = Result.SourceLoopWorld[i];
			Sample.Depth = Result.ExtrusionDepth;
			Sample.PointOnExtrusion = Result.SourceLoopWorld[i] + Result.OrientedNormal * Result.ExtrusionDepth;
			Sample.PointOnRay = Sample.PointOnExtrusion;
			Sample.ClosestWorldDistance = 0.0;

			FVector2D Reprojected;
			if (!ProjectWorldToImageOrthographic(
				Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight,
				Sample.PointOnExtrusion, Step10OrthoWidth, Reprojected))
			{
				Sample.Error = TEXT("extruded vertex failed to reproject");
				Result.DepthSamples.Add(Sample);
				continue;
			}

			Sample.ReprojectionErrorPixels = FVector2D::Distance(Reprojected, Sample.CopiedPixel);
			if (Sample.ReprojectionErrorPixels > Result.MaxDepthSampleReprojectionErrorPixels)
			{
				Sample.Error = TEXT("vertex reprojection error is too large");
			}
			else
			{
				Sample.bValid = true;
				++ValidVertexCount;
			}
			Result.DepthSamples.Add(Sample);
		}

		if (ValidVertexCount < MinSolidDepthSamples)
		{
			Result.Error = FString::Printf(
				TEXT("not enough vertices match copied cap after orthographic extrusion solve: %d valid, need %d"),
				ValidVertexCount, MinSolidDepthSamples);
			return Result;
		}
		Result.CopiedLoopWorld.Reserve(Result.SourceLoopWorld.Num());
		for (const FVector& P : Result.SourceLoopWorld)
		{
			Result.CopiedLoopWorld.Add(P + Result.OrientedNormal * Result.ExtrusionDepth);
		}

		Result.ReprojectedSourceLoop2D.Reserve(Result.SourceLoopWorld.Num());
		Result.ReprojectedCopiedLoop2D.Reserve(Result.CopiedLoopWorld.Num());
		double CopiedErrorSum = 0.0;
		for (int32 i = 0; i < Result.SourceLoopWorld.Num(); ++i)
		{
			FVector2D SourceProj;
			FVector2D CopiedProj;
			if (!ProjectWorldToImageOrthographic(Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight, Result.SourceLoopWorld[i], Step10OrthoWidth, SourceProj) ||
				!ProjectWorldToImageOrthographic(Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight, Result.CopiedLoopWorld[i], Step10OrthoWidth, CopiedProj))
			{
				Result.Error = TEXT("failed to reproject generated source/copied solid loop");
				return Result;
			}
			Result.ReprojectedSourceLoop2D.Add(SourceProj);
			Result.ReprojectedCopiedLoop2D.Add(CopiedProj);

			const double CopiedError = FVector2D::Distance(CopiedProj, Result.CopiedTargetLoop2D[i]);
			CopiedErrorSum += CopiedError;
			Result.MaxCopiedReprojectionErrorPixels = FMath::Max(Result.MaxCopiedReprojectionErrorPixels, CopiedError);
		}
		Result.MeanCopiedReprojectionErrorPixels = CopiedErrorSum / double(FMath::Max(1, Result.ReprojectedCopiedLoop2D.Num()));
		if (Result.MaxCopiedReprojectionErrorPixels > Result.MaxDepthSampleReprojectionErrorPixels)
		{
			Result.Warning = FString::Printf(
				TEXT("copied loop reprojection max error %.3f px exceeds depth-sample threshold %.3f px"),
				Result.MaxCopiedReprojectionErrorPixels, Result.MaxDepthSampleReprojectionErrorPixels);
		}

		if (!BuildSolidMeshTriangles(
			Result.SourceLoop2D, Result.CopiedTargetLoop2D,
			Result.SourceLoopWorld, Result.CopiedLoopWorld,
			Result.OrientedNormal, Result.MeshVerticesWorld, Result.MeshTriangles, Result.MeshNormal))
		{
			Result.Error = TEXT("failed to triangulate source cap or build solid side faces");
			return Result;
		}

		Result.ReprojectedSourceLoop2D.Reset();
		Result.ReprojectedCopiedLoop2D.Reset();
		Result.ReprojectedSourceLoop2D.Reserve(Result.SourceLoopWorld.Num());
		Result.ReprojectedCopiedLoop2D.Reserve(Result.CopiedLoopWorld.Num());
		Result.MeanCopiedReprojectionErrorPixels = 0.0;
		Result.MaxCopiedReprojectionErrorPixels = 0.0;
		double FinalCopiedErrorSum = 0.0;
		for (int32 i = 0; i < Result.SourceLoopWorld.Num(); ++i)
		{
			FVector2D SourceProj;
			FVector2D CopiedProj;
			if (!ProjectWorldToImageOrthographic(Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight, Result.SourceLoopWorld[i], Step10OrthoWidth, SourceProj) ||
				!ProjectWorldToImageOrthographic(Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight, Result.CopiedLoopWorld[i], Step10OrthoWidth, CopiedProj))
			{
				Result.Error = TEXT("failed to reproject final solid loop");
				Result.bSuccess = false;
				return Result;
			}
			Result.ReprojectedSourceLoop2D.Add(SourceProj);
			Result.ReprojectedCopiedLoop2D.Add(CopiedProj);
			const double CopiedError = FVector2D::Distance(CopiedProj, Result.CopiedTargetLoop2D[i]);
			FinalCopiedErrorSum += CopiedError;
			Result.MaxCopiedReprojectionErrorPixels = FMath::Max(Result.MaxCopiedReprojectionErrorPixels, CopiedError);
		}
		Result.MeanCopiedReprojectionErrorPixels = FinalCopiedErrorSum / double(FMath::Max(1, Result.ReprojectedCopiedLoop2D.Num()));
		Result.Warning.Reset();
		if (Result.MaxCopiedReprojectionErrorPixels > Result.MaxDepthSampleReprojectionErrorPixels)
		{
			Result.Warning = FString::Printf(
				TEXT("copied loop reprojection max error %.3f px exceeds depth-sample threshold %.3f px"),
				Result.MaxCopiedReprojectionErrorPixels, Result.MaxDepthSampleReprojectionErrorPixels);
		}

		Result.bSuccess = true;
		return Result;
	}

	static FComponentResult ProcessComponent(
		const FString& ComponentName,
		const FString& PressDir,
		const FString& ActionPressDir,
		const FCommonInputs& Inputs)
	{
		const FString ComponentDir = PressDir / ComponentName;
		FComponentResult Result;
		Result.ComponentName = ComponentName;
		Result.FacesWidth = Inputs.FacesWidth;
		Result.FacesHeight = Inputs.FacesHeight;
		Result.ActorName = FString::Printf(TEXT("FromLZ_ReconstructedFace_%s_%s"), *FPaths::GetCleanFilename(PressDir), *ComponentName);
		InitializeSolidResult(Result.Solid, ComponentName, PressDir, Inputs);

		const FString OutputJson = ComponentDir / TEXT("10_face_reconstruction.json");
		const FString SolidJson = ComponentDir / TEXT("10_solid_reconstruction.json");
		auto SaveFaceAndSkippedSolid = [&](const FString& SolidError)
		{
			Result.Solid.Action = Result.Action;
			Result.Solid.SelectedFaceId = Result.SelectedFaceId;
			Result.Solid.CapWidth = Result.CapWidth;
			Result.Solid.CapHeight = Result.CapHeight;
			Result.Solid.SourcePolygonKey = Result.PolygonKey;
			SaveComponentResultJson(Result, OutputJson);
			SaveSkippedSolidResult(Result.Solid, SolidJson, SolidError);
		};

		const FString ActionPath = ActionPressDir / ComponentName / TEXT("Action.json");
		if (!LoadAction(ActionPath, Result.Action))
		{
			Result.Error = FString::Printf(TEXT("Failed to read action from %s"), *ActionPath);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}
		Result.Solid.Action = Result.Action;

		if (Result.Action == TEXT("excavate"))
		{
			Result.PolygonKey = TEXT("cap_polygon");
			Result.Solid.SourcePolygonKey = TEXT("cap_polygon");
			Result.Solid.CopiedPolygonKey = TEXT("cap_polygon_translated_opposite");
		}
		else if (Result.Action == TEXT("attach"))
		{
			Result.PolygonKey = TEXT("cap_polygon_translated");
			Result.Solid.SourcePolygonKey = TEXT("cap_polygon_translated");
			Result.Solid.CopiedPolygonKey = TEXT("cap_polygon");
		}
		else
		{
			Result.Error = FString::Printf(TEXT("Unsupported action '%s'"), *Result.Action);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		TSharedPtr<FJsonObject> CapJson;
		const FString CapJsonPath = ComponentDir / TEXT("09_cap_extrusion.json");
		if (!LoadJsonObject(CapJsonPath, CapJson))
		{
			Result.Error = FString::Printf(TEXT("Failed to read %s"), *CapJsonPath);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		TArray<FVector2D> RawCapPolygon;
		TArray<FVector2D> RawTranslatedPolygon;
		const bool bHasCapPolygon = ParseVector2DArray(CapJson, TEXT("cap_polygon"), RawCapPolygon);
		const bool bHasTranslatedPolygon = ParseVector2DArray(CapJson, TEXT("cap_polygon_translated"), RawTranslatedPolygon);
		const TArray<FVector2D>* SourcePolyForFace = Result.PolygonKey == TEXT("cap_polygon") ? &RawCapPolygon : &RawTranslatedPolygon;
		if (!SourcePolyForFace || SourcePolyForFace->Num() < 3 ||
			(Result.PolygonKey == TEXT("cap_polygon") && !bHasCapPolygon) ||
			(Result.PolygonKey == TEXT("cap_polygon_translated") && !bHasTranslatedPolygon))
		{
			Result.Error = FString::Printf(TEXT("Missing or invalid polygon '%s' in %s"), *Result.PolygonKey, *CapJsonPath);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}
		const TArray<FVector2D>& CapPoly = *SourcePolyForFace;

		FVector2D SideVector = FVector2D::ZeroVector;
		if (!ParseVector2DField(CapJson, TEXT("side_vector"), SideVector))
		{
			Result.Error = FString::Printf(TEXT("Missing or invalid side_vector in %s"), *CapJsonPath);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		TArray<uint8> CapRGBA;
		const FString CapPngPath = ComponentDir / TEXT("09_cap_extrusion.png");
		if (!DecodePngToRGBA(CapPngPath, CapRGBA, Result.CapWidth, Result.CapHeight))
		{
			Result.Error = FString::Printf(TEXT("Failed to read cap image size from %s"), *CapPngPath);
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}
		Result.Solid.CapWidth = Result.CapWidth;
		Result.Solid.CapHeight = Result.CapHeight;

		const double ScaleX = double(Inputs.FacesWidth) / double(Result.CapWidth);
		const double ScaleY = double(Inputs.FacesHeight) / double(Result.CapHeight);
		const FVector2D ScaledSideVector(SideVector.X * ScaleX, SideVector.Y * ScaleY);
		if (ScaledSideVector.SizeSquared() < 1e-8)
		{
			Result.Error = TEXT("side_vector is too short after mapping to faces image space");
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}
		Result.GreenLineVector2D = ScaledSideVector.GetSafeNormal();

		// All green lines connected to this cap (mapped to faces image space). The
		// parallel filter below passes a face if its normal aligns with ANY of these.
		// Scaling is per-axis (ScaleX != ScaleY rotates the vector), so map then normalize.
		TArray<FVector2D> RawSideVectors;
		ParseVector2DArray(CapJson, TEXT("side_vectors"), RawSideVectors);
		for (const FVector2D& V : RawSideVectors)
		{
			const FVector2D Scaled(V.X * ScaleX, V.Y * ScaleY);
			if (Scaled.SizeSquared() >= 1e-8)
			{
				Result.GreenLineVectors2D.Add(Scaled.GetSafeNormal());
			}
		}
		// Fall back to the single side_vector when the array is absent (older cap json).
		if (Result.GreenLineVectors2D.Num() == 0)
		{
			Result.GreenLineVectors2D.Add(Result.GreenLineVector2D);
		}

		// Endpoint segments (faces space) for the debug overlay.
		TArray<FVector2D> RawSegStarts, RawSegEnds;
		if (ParseSegment2DArray(CapJson, TEXT("side_segments"), RawSegStarts, RawSegEnds))
		{
			for (int32 i = 0; i < RawSegStarts.Num(); ++i)
			{
				Result.GreenSegStarts.Emplace(RawSegStarts[i].X * ScaleX, RawSegStarts[i].Y * ScaleY);
				Result.GreenSegEnds.Emplace(RawSegEnds[i].X * ScaleX, RawSegEnds[i].Y * ScaleY);
			}
		}

		TArray<FVector2D> FaceSpacePoly;
		FaceSpacePoly.Reserve(CapPoly.Num());
		for (const FVector2D& P : CapPoly)
		{
			FaceSpacePoly.Emplace(P.X * ScaleX, P.Y * ScaleY);
		}

		TArray<uint8> Mask;
		RasterizePolygonMask(FaceSpacePoly, Inputs.FacesWidth, Inputs.FacesHeight, Mask, Result.CapMaskPixels);
		Result.MinOverlapPixels = FMath::Max(1, FMath::CeilToInt(double(Result.CapMaskPixels) * MinOverlapRatio));
		SaveMaskPng(Mask, Inputs.FacesWidth, Inputs.FacesHeight, ComponentDir / TEXT("10_cap_mask.png"));
		if (Result.CapMaskPixels <= 0)
		{
			Result.Error = TEXT("Cap mask is empty after mapping to faces image space");
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		TMap<int32, FOverlapAccum> AccumByFace;
		for (int32 y = 0; y < Inputs.FacesHeight; ++y)
		{
			for (int32 x = 0; x < Inputs.FacesWidth; ++x)
			{
				const int32 PixIdx = y * Inputs.FacesWidth + x;
				if (Mask[PixIdx] == 0)
				{
					continue;
				}
				const int32 Off = PixIdx * 4;
				const uint32 Key = ColorKey(Inputs.FacesRGBA[Off + 0], Inputs.FacesRGBA[Off + 1], Inputs.FacesRGBA[Off + 2]);
				if (const int32* FaceId = Inputs.FaceIdByColorKey.Find(Key))
				{
					FOverlapAccum& Acc = AccumByFace.FindOrAdd(*FaceId);
					Acc.Pixels += 1;
					Acc.SumX += double(x) + 0.5;
					Acc.SumY += double(y) + 0.5;
				}
			}
		}

		for (const TPair<int32, FOverlapAccum>& Pair : AccumByFace)
		{
			if (Pair.Value.Pixels < Result.MinOverlapPixels)
			{
				continue;
			}
			const int32* FaceIndex = Inputs.FaceIndexById.Find(Pair.Key);
			if (!FaceIndex)
			{
				continue;
			}

			const FFaceInfo& Face = Inputs.Faces[*FaceIndex];
			FFaceCandidate Candidate;
			Candidate.FaceId = Pair.Key;
			Candidate.OverlapPixels = Pair.Value.Pixels;
			Candidate.OverlapRatio = double(Pair.Value.Pixels) / double(Result.CapMaskPixels);
			Candidate.MaskCentroid = FVector2D(Pair.Value.SumX / double(Pair.Value.Pixels), Pair.Value.SumY / double(Pair.Value.Pixels));
			Candidate.bHasPlaneHit = IntersectMaskCentroidWithFacePlane(
				Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight,
				Candidate.MaskCentroid, Face, Candidate.PlaneHit, Candidate.DistanceToCamera);
			if (Candidate.bHasPlaneHit)
			{
				Candidate.bHasProjectedNormal = ProjectFaceNormalToImage(
					Inputs.Camera, Inputs.FacesWidth, Inputs.FacesHeight,
					Face, Candidate.PlaneHit, Candidate.ProjectedNormal2D);
				if (Candidate.bHasProjectedNormal)
				{
					// Test the projected normal against every cap-connected green line;
					// keep the smallest angle and orient the debug vector toward that green line.
					const FVector2D NormalDir = Candidate.ProjectedNormal2D.GetSafeNormal();
					double BestAngle = 180.0;
					FVector2D BestOrientedNormalDir = NormalDir;
					for (const FVector2D& Green : Result.GreenLineVectors2D)
					{
						const FVector2D GreenDir = Green.GetSafeNormal();
						if (GreenDir.IsNearlyZero())
						{
							continue;
						}
						const double SignedDot = FVector2D::DotProduct(NormalDir, GreenDir);
						const double Dot = FMath::Clamp(FMath::Abs(SignedDot), 0.0, 1.0);
						const double Angle = FMath::RadiansToDegrees(FMath::Acos(Dot));
						if (Angle < BestAngle)
						{
							BestAngle = Angle;
							BestOrientedNormalDir = SignedDot >= 0.0 ? NormalDir : -NormalDir;
						}
					}
					Candidate.ProjectedNormal2D = BestOrientedNormalDir;
					Candidate.NormalGreenAngleDegrees = BestAngle;
					Candidate.bNormalParallelPass = BestAngle <= NormalParallelThresholdDegrees;
				}
			}
			Result.Candidates.Add(Candidate);
		}

		Result.Candidates.Sort([](const FFaceCandidate& A, const FFaceCandidate& B)
		{
			return A.FaceId < B.FaceId;
		});

		double BestDistance = TNumericLimits<double>::Max();
		for (const FFaceCandidate& Candidate : Result.Candidates)
		{
			if (Candidate.bHasPlaneHit && Candidate.bNormalParallelPass && Candidate.DistanceToCamera < BestDistance)
			{
				BestDistance = Candidate.DistanceToCamera;
				Result.SelectedFaceId = Candidate.FaceId;
				Result.SelectedPlaneHit = Candidate.PlaneHit;
			}
		}

		TSet<int32> CandidateIds;
		for (const FFaceCandidate& Candidate : Result.Candidates)
		{
			CandidateIds.Add(Candidate.FaceId);
		}
		TSet<int32> ParallelFaceIds;
		for (const FFaceCandidate& Candidate : Result.Candidates)
		{
			if (Candidate.bNormalParallelPass)
			{
				ParallelFaceIds.Add(Candidate.FaceId);
			}
		}
		SaveOverlapPng(
			Inputs.FacesRGBA, Mask, Inputs.FaceIdByColorKey, CandidateIds, ParallelFaceIds, Result.SelectedFaceId,
			Inputs.FacesWidth, Inputs.FacesHeight, ComponentDir / TEXT("10_face_overlap.png"));

		// Visualize the cap-connected green lines vs each candidate face's projected normal.
		SaveNormalGreenCheckPng(
			Inputs.FacesRGBA, Inputs.FacesWidth, Inputs.FacesHeight,
			FaceSpacePoly, Result.GreenSegStarts, Result.GreenSegEnds,
			Result.Candidates, Result.SelectedFaceId,
			ComponentDir / TEXT("10_normal_green_check.png"));

		if (Result.SelectedFaceId < 0)
		{
			if (Result.Candidates.Num() == 0)
			{
				Result.Error = TEXT("No face survived the 5% mask-overlap threshold");
			}
			else if (ParallelFaceIds.Num() == 0)
			{
				Result.Error = TEXT("No 5% mask-overlap candidate passed the 10 degree normal-to-green-line parallel filter");
			}
			else
			{
				Result.Error = TEXT("No parallel face candidate had a valid camera-to-plane intersection");
			}
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because Step 10 did not select a source face: %s"), *Result.Error));
			return Result;
		}

		const int32* SelectedIndex = Inputs.FaceIndexById.Find(Result.SelectedFaceId);
		if (!SelectedIndex)
		{
			Result.Error = TEXT("Selected face id was not found in face table");
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		const FFaceInfo& SelectedFace = Inputs.Faces[*SelectedIndex];
		Result.MeshVerticesWorld = SelectedFace.KeyPoints3D;
		if (!TriangulatePolygon2D(SelectedFace.KeyPoints2D, Result.MeshTriangles))
		{
			Result.Error = TEXT("Failed to triangulate selected face");
			SaveFaceAndSkippedSolid(FString::Printf(TEXT("Solid skipped because face reconstruction failed: %s"), *Result.Error));
			return Result;
		}

		FVector Center = FVector::ZeroVector;
		for (const FVector& V : Result.MeshVerticesWorld)
		{
			Center += V;
		}
		Center /= double(FMath::Max(1, Result.MeshVerticesWorld.Num()));

		Result.MeshNormal = ComputeTriangleNormal(Result.MeshVerticesWorld, Result.MeshTriangles);
		if (Result.MeshNormal.IsNearlyZero())
		{
			Result.MeshNormal = SelectedFace.Normal;
		}
		if (FVector::DotProduct(Result.MeshNormal, Inputs.Camera.Location - Center) < 0.0)
		{
			for (int32 i = 0; i + 2 < Result.MeshTriangles.Num(); i += 3)
			{
				Swap(Result.MeshTriangles[i + 1], Result.MeshTriangles[i + 2]);
			}
			Result.MeshNormal *= -1.0;
		}

		Result.bSuccess = true;
		SaveComponentResultJson(Result, OutputJson);
		if (!bHasCapPolygon || !bHasTranslatedPolygon)
		{
			SaveSkippedSolidResult(
				Result.Solid, SolidJson,
				TEXT("Solid skipped because cap_polygon or cap_polygon_translated is missing from 09_cap_extrusion.json"));
			return Result;
		}

		TArray<FVector2D> OppositeTranslatedPolygon;
		const TArray<FVector2D>& SolidSourcePoly = Result.Action == TEXT("excavate") ? RawCapPolygon : RawTranslatedPolygon;
		const TArray<FVector2D>* SolidCopiedPoly = nullptr;
		if (Result.Action == TEXT("excavate"))
		{
			if (!BuildOppositeTranslatedPolygon(RawCapPolygon, RawTranslatedPolygon, OppositeTranslatedPolygon))
			{
				SaveSkippedSolidResult(
					Result.Solid, SolidJson,
					TEXT("Solid skipped because cap_polygon and cap_polygon_translated cannot form the inward excavation loop"));
				return Result;
			}
			SolidCopiedPoly = &OppositeTranslatedPolygon;
		}
		else
		{
			SolidCopiedPoly = &RawCapPolygon;
		}

		Result.Solid = BuildSolidReconstruction(
			ComponentName,
			Result.Action,
			Result.Solid.SourcePolygonKey,
			Result.Solid.CopiedPolygonKey,
			PressDir,
			Result.CapWidth,
			Result.CapHeight,
			ScaleX,
			ScaleY,
			SolidSourcePoly,
			*SolidCopiedPoly,
			SelectedFace,
			Inputs);
		SaveSolidResultJson(Result.Solid, SolidJson);
		SaveSolidProjectionCheckPng(
			Result.Solid, Inputs.FacesRGBA, Inputs.FacesWidth, Inputs.FacesHeight,
			ComponentDir / TEXT("10_solid_projection_check.png"));
		return Result;
	}

	static FString ObjSafeName(FString Name)
	{
		Name.TrimStartAndEndInline();
		for (TCHAR& Ch : Name)
		{
			if (!FChar::IsAlnum(Ch) && Ch != TCHAR('_') && Ch != TCHAR('-'))
			{
				Ch = TCHAR('_');
			}
		}
		return Name.IsEmpty() ? TEXT("Mesh") : Name;
	}

	static FName Step11PressTag(const FString& PressId)
	{
		return FName(*FString::Printf(TEXT("FromLZ_Press_%s"), *PressId));
	}

	static bool ActorHasAnyStep11RuntimeTag(const AActor* Actor)
	{
		return Actor && (
			Actor->ActorHasTag(Step11BooleanResultTag) ||
			Actor->ActorHasTag(Step11ActionAttachTag) ||
			Actor->ActorHasTag(Step11ActionExcavateCutterTag));
	}

	static bool ActorIsStep11Cutter(const AActor* Actor)
	{
		return Actor && Actor->ActorHasTag(Step11ActionExcavateCutterTag);
	}

	static void MergeStep11StringSet(TSet<FString>& Dest, const TSet<FString>& Source)
	{
		for (const FString& Value : Source)
		{
			Dest.Add(Value);
		}
	}

	static void AppendDoubleSidedTriangles(const TArray<int32>& Triangles, TArray<int32>& OutTriangles)
	{
		OutTriangles.Reset();
		OutTriangles.Reserve(Triangles.Num() * 2);
		for (int32 i = 0; i + 2 < Triangles.Num(); i += 3)
		{
			const int32 A = Triangles[i];
			const int32 B = Triangles[i + 1];
			const int32 C = Triangles[i + 2];
			OutTriangles.Add(A);
			OutTriangles.Add(B);
			OutTriangles.Add(C);
			OutTriangles.Add(A);
			OutTriangles.Add(C);
			OutTriangles.Add(B);
		}
	}

	static UMaterialInterface* GetReconstructionVertexColorMaterial()
	{
		static TWeakObjectPtr<UMaterialInterface> CachedMaterial;
		if (UMaterialInterface* Cached = CachedMaterial.Get())
		{
			return Cached;
		}

		const TCHAR* MaterialPaths[] =
		{
			TEXT("/Engine/EngineDebugMaterials/VertexColorViewMode_ColorOnly.VertexColorViewMode_ColorOnly"),
			TEXT("/Engine/EngineMaterials/VertexColorMaterial.VertexColorMaterial")
		};
		for (const TCHAR* Path : MaterialPaths)
		{
			if (UMaterialInterface* Material = LoadObject<UMaterialInterface>(nullptr, Path))
			{
				CachedMaterial = Material;
				return Material;
			}
		}

		if (GEngine && GEngine->VertexColorMaterial)
		{
			CachedMaterial = GEngine->VertexColorMaterial;
			return GEngine->VertexColorMaterial;
		}

		return UMaterial::GetDefaultMaterial(MD_Surface);
	}

	static UMaterialInterface* GetReconstructionCutterBaseMaterial()
	{
		static TWeakObjectPtr<UMaterialInterface> CachedMaterial;
		if (UMaterialInterface* Cached = CachedMaterial.Get())
		{
			return Cached;
		}

		const TCHAR* MaterialPaths[] =
		{
			TEXT("/Engine/EngineDebugMaterials/M_SimpleUnlitTranslucent.M_SimpleUnlitTranslucent"),
			TEXT("/Engine/EngineDebugMaterials/M_SimpleTranslucent.M_SimpleTranslucent"),
			TEXT("/Engine/EngineMaterials/Widget3DPassThrough_Translucent.Widget3DPassThrough_Translucent")
		};
		for (const TCHAR* Path : MaterialPaths)
		{
			if (UMaterialInterface* Material = LoadObject<UMaterialInterface>(nullptr, Path))
			{
				CachedMaterial = Material;
				return Material;
			}
		}

		return GetReconstructionVertexColorMaterial();
	}

	static UMaterialInterface* CreateCutterMaterial(UObject* Outer)
	{
		UMaterialInterface* BaseMaterial = GetReconstructionCutterBaseMaterial();
		UMaterialInstanceDynamic* DynamicMaterial = UMaterialInstanceDynamic::Create(BaseMaterial, Outer);
		if (!DynamicMaterial)
		{
			return BaseMaterial;
		}

		const FLinearColor CutterColor(0.0f, 0.47f, 1.0f, 0.25f);
		const FName ColorParameterNames[] =
		{
			TEXT("Color"),
			TEXT("BaseColor"),
			TEXT("Tint"),
			TEXT("TintColor"),
			TEXT("EmissiveColor")
		};
		for (const FName& ParameterName : ColorParameterNames)
		{
			DynamicMaterial->SetVectorParameterValue(ParameterName, CutterColor);
		}

		const FName OpacityParameterNames[] =
		{
			TEXT("Opacity"),
			TEXT("Alpha"),
			TEXT("Transparency")
		};
		for (const FName& ParameterName : OpacityParameterNames)
		{
			DynamicMaterial->SetScalarParameterValue(ParameterName, CutterColor.A);
		}
		return DynamicMaterial;
	}

	static FBox BuildWorldBounds(const TArray<FVector>& Vertices)
	{
		FBox Box(ForceInit);
		for (const FVector& V : Vertices)
		{
			Box += V;
		}
		return Box;
	}

	static FBox BuildDynamicMeshBounds(const UE::Geometry::FDynamicMesh3& Mesh)
	{
		FBox Box(ForceInit);
		for (int32 VertexId : Mesh.VertexIndicesItr())
		{
			const FVector3d V = Mesh.GetVertex(VertexId);
			Box += FVector(V.X, V.Y, V.Z);
		}
		return Box;
	}

	static FString Step11EdgeKey(int32 A, int32 B)
	{
		if (A > B)
		{
			Swap(A, B);
		}
		return FString::Printf(TEXT("%d_%d"), A, B);
	}

	static FString Step11DirectedEdgeKey(int32 A, int32 B)
	{
		return FString::Printf(TEXT("%d_%d"), A, B);
	}

	static FString Step11TriangleKey(int32 A, int32 B, int32 C)
	{
		TArray<int32> Indices;
		Indices.Add(A);
		Indices.Add(B);
		Indices.Add(C);
		Indices.Sort();
		return FString::Printf(TEXT("%d_%d_%d"), Indices[0], Indices[1], Indices[2]);
	}

	static void AddStep11EdgeDiagnostics(
		int32 A,
		int32 B,
		double EdgeLength,
		TMap<FString, int32>& UndirectedEdgeCounts,
		TMap<FString, int32>& DirectedEdgeCounts,
		TMap<FString, double>& EdgeLengthByKey)
	{
		const FString UndirectedKey = Step11EdgeKey(A, B);
		UndirectedEdgeCounts.FindOrAdd(UndirectedKey) += 1;
		DirectedEdgeCounts.FindOrAdd(Step11DirectedEdgeKey(A, B)) += 1;
		if (!EdgeLengthByKey.Contains(UndirectedKey))
		{
			EdgeLengthByKey.Add(UndirectedKey, EdgeLength);
		}
	}

	static FStep11MeshDiagnostics AnalyzeStep11DynamicMesh(
		const UE::Geometry::FDynamicMesh3& Mesh,
		const FString& Label,
		const FString& SourceType)
	{
		constexpr double DegenerateAreaTolerance = 1e-8;
		constexpr double TinyAreaTolerance = 1e-4;

		FStep11MeshDiagnostics Diagnostics;
		Diagnostics.Label = Label;
		Diagnostics.SourceType = SourceType;
		Diagnostics.VertexCount = Mesh.VertexCount();
		Diagnostics.TriangleCount = Mesh.TriangleCount();
		Diagnostics.Bounds = BuildDynamicMeshBounds(Mesh);

		TMap<FString, int32> UndirectedEdgeCounts;
		TMap<FString, int32> DirectedEdgeCounts;
		TMap<FString, double> EdgeLengthByKey;
		TSet<FString> TriangleKeys;

		double MinTriangleArea = TNumericLimits<double>::Max();
		double MaxTriangleArea = 0.0;
		for (int32 TriangleId : Mesh.TriangleIndicesItr())
		{
			const UE::Geometry::FIndex3i Tri = Mesh.GetTriangle(TriangleId);
			if (Tri.A == Tri.B || Tri.B == Tri.C || Tri.C == Tri.A)
			{
				++Diagnostics.InvalidTriangleCount;
				continue;
			}

			const FString TriKey = Step11TriangleKey(Tri.A, Tri.B, Tri.C);
			if (TriangleKeys.Contains(TriKey))
			{
				++Diagnostics.DuplicateTriangleCount;
			}
			else
			{
				TriangleKeys.Add(TriKey);
			}

			const FVector3d A3 = Mesh.GetVertex(Tri.A);
			const FVector3d B3 = Mesh.GetVertex(Tri.B);
			const FVector3d C3 = Mesh.GetVertex(Tri.C);
			const FVector A(A3.X, A3.Y, A3.Z);
			const FVector B(B3.X, B3.Y, B3.Z);
			const FVector C(C3.X, C3.Y, C3.Z);
			const double Area = 0.5 * FVector::CrossProduct(B - A, C - A).Size();
			Diagnostics.SurfaceArea += Area;
			MinTriangleArea = FMath::Min(MinTriangleArea, Area);
			MaxTriangleArea = FMath::Max(MaxTriangleArea, Area);
			if (Area <= DegenerateAreaTolerance)
			{
				++Diagnostics.DegenerateTriangleCount;
			}
			else if (Area <= TinyAreaTolerance)
			{
				++Diagnostics.TinyTriangleCount;
			}

			Diagnostics.SignedVolume += FVector::DotProduct(A, FVector::CrossProduct(B, C)) / 6.0;

			AddStep11EdgeDiagnostics(Tri.A, Tri.B, FVector::Distance(A, B), UndirectedEdgeCounts, DirectedEdgeCounts, EdgeLengthByKey);
			AddStep11EdgeDiagnostics(Tri.B, Tri.C, FVector::Distance(B, C), UndirectedEdgeCounts, DirectedEdgeCounts, EdgeLengthByKey);
			AddStep11EdgeDiagnostics(Tri.C, Tri.A, FVector::Distance(C, A), UndirectedEdgeCounts, DirectedEdgeCounts, EdgeLengthByKey);
		}

		Diagnostics.EdgeCount = UndirectedEdgeCounts.Num();
		for (const TPair<FString, int32>& Pair : UndirectedEdgeCounts)
		{
			if (Pair.Value == 1)
			{
				++Diagnostics.BoundaryEdgeCount;
			}
			else if (Pair.Value > 2)
			{
				++Diagnostics.NonManifoldEdgeCount;
			}

			TArray<FString> Parts;
			Pair.Key.ParseIntoArray(Parts, TEXT("_"), true);
			if (Parts.Num() == 2)
			{
				const int32 A = FCString::Atoi(*Parts[0]);
				const int32 B = FCString::Atoi(*Parts[1]);
				const int32 AB = DirectedEdgeCounts.FindRef(Step11DirectedEdgeKey(A, B));
				const int32 BA = DirectedEdgeCounts.FindRef(Step11DirectedEdgeKey(B, A));
				if (Pair.Value == 2 && !(AB == 1 && BA == 1))
				{
					++Diagnostics.InconsistentOrientationEdgeCount;
				}
			}
		}

		double EdgeLengthSum = 0.0;
		double MinEdgeLength = TNumericLimits<double>::Max();
		double MaxEdgeLength = 0.0;
		for (const TPair<FString, double>& Pair : EdgeLengthByKey)
		{
			EdgeLengthSum += Pair.Value;
			MinEdgeLength = FMath::Min(MinEdgeLength, Pair.Value);
			MaxEdgeLength = FMath::Max(MaxEdgeLength, Pair.Value);
		}

		Diagnostics.MinTriangleArea = MinTriangleArea == TNumericLimits<double>::Max() ? 0.0 : MinTriangleArea;
		Diagnostics.MaxTriangleArea = MaxTriangleArea;
		Diagnostics.AbsVolume = FMath::Abs(Diagnostics.SignedVolume);
		Diagnostics.SignedVolumeBeforeOrientationFix = Diagnostics.SignedVolume;
		Diagnostics.SignedVolumeAfterOrientationFix = Diagnostics.SignedVolume;
		Diagnostics.MinEdgeLength = MinEdgeLength == TNumericLimits<double>::Max() ? 0.0 : MinEdgeLength;
		Diagnostics.MaxEdgeLength = MaxEdgeLength;
		Diagnostics.MeanEdgeLength = EdgeLengthByKey.Num() > 0 ? EdgeLengthSum / double(EdgeLengthByKey.Num()) : 0.0;
		return Diagnostics;
	}

	static bool NormalizeStep11MeshOrientationForBoolean(
		UE::Geometry::FDynamicMesh3& Mesh,
		FStep11MeshDiagnostics& Diagnostics)
	{
		constexpr double SignedVolumeTolerance = 1e-6;
		const double SignedVolumeBefore = Diagnostics.SignedVolume;
		if (SignedVolumeBefore >= -SignedVolumeTolerance)
		{
			Diagnostics.SignedVolumeBeforeOrientationFix = SignedVolumeBefore;
			Diagnostics.SignedVolumeAfterOrientationFix = SignedVolumeBefore;
			Diagnostics.bOrientationReversedForBoolean = false;
			return false;
		}

		const FString Label = Diagnostics.Label;
		const FString SourceType = Diagnostics.SourceType;
		Mesh.ReverseOrientation(false);
		Diagnostics = AnalyzeStep11DynamicMesh(Mesh, Label, SourceType);
		Diagnostics.SignedVolumeBeforeOrientationFix = SignedVolumeBefore;
		Diagnostics.SignedVolumeAfterOrientationFix = Diagnostics.SignedVolume;
		Diagnostics.bOrientationReversedForBoolean = true;

		UE_LOG(
			LogTemp,
			Log,
			TEXT("Step11Diag: reversed mesh orientation for boolean label=%s source=%s signed_volume_before=%.6f signed_volume_after=%.6f."),
			*Diagnostics.Label,
			*Diagnostics.SourceType,
			SignedVolumeBefore,
			Diagnostics.SignedVolume);
		return true;
	}

	static int32 Step11SignedVolumeSign(double SignedVolume)
	{
		constexpr double SignedVolumeTolerance = 1e-6;
		if (SignedVolume > SignedVolumeTolerance)
		{
			return 1;
		}
		if (SignedVolume < -SignedVolumeTolerance)
		{
			return -1;
		}
		return 0;
	}

	static bool ReverseStep11MeshIfSignMismatch(UE::Geometry::FDynamicMesh3& Mesh, int32 DesiredSign)
	{
		if (DesiredSign == 0)
		{
			return false;
		}

		const FStep11MeshDiagnostics Diagnostics = AnalyzeStep11DynamicMesh(Mesh, TEXT("orientation_check"), TEXT("orientation_check"));
		const int32 CurrentSign = Step11SignedVolumeSign(Diagnostics.SignedVolume);
		if (CurrentSign != 0 && CurrentSign != DesiredSign)
		{
			Mesh.ReverseOrientation(false);
			return true;
		}
		return false;
	}

	struct FStep11BooleanAttemptResult
	{
		bool bComputeSuccess = false;
		bool bAccepted = false;
		bool bEmptyResult = false;
		bool bTargetReversedForPass = false;
		bool bCutterReversedForPass = false;
		bool bResultReversedToTargetSign = false;
		FString PassName;
		FString Status;
		FString RejectReason;
		double VolumeDelta = 0.0;
		double MinRequiredVolumeDelta = 0.0;
		FStep11MeshDiagnostics ResultDiagnostics;
		UE::Geometry::FDynamicMesh3 ResultMesh;
		TArray<int8> TriangleSourceMeshById;
	};

	static TSharedRef<FJsonObject> Step11BoxJson(const FBox& Box)
	{
		TSharedRef<FJsonObject> Object = MakeShared<FJsonObject>();
		Object->SetBoolField(TEXT("valid"), Box.IsValid != 0);
		if (Box.IsValid)
		{
			Object->SetArrayField(TEXT("min"), JsonVector(Box.Min)->AsArray());
			Object->SetArrayField(TEXT("max"), JsonVector(Box.Max)->AsArray());
			Object->SetArrayField(TEXT("center"), JsonVector(Box.GetCenter())->AsArray());
			Object->SetArrayField(TEXT("extent"), JsonVector(Box.GetExtent())->AsArray());
			Object->SetArrayField(TEXT("size"), JsonVector(Box.GetSize())->AsArray());
			Object->SetNumberField(TEXT("diagonal"), Box.GetSize().Size());
		}
		return Object;
	}

	static TSharedRef<FJsonObject> Step11MeshDiagnosticsJson(const FStep11MeshDiagnostics& Diagnostics)
	{
		TSharedRef<FJsonObject> Object = MakeShared<FJsonObject>();
		Object->SetStringField(TEXT("label"), Diagnostics.Label);
		Object->SetStringField(TEXT("source_type"), Diagnostics.SourceType);
		Object->SetNumberField(TEXT("vertex_count"), Diagnostics.VertexCount);
		Object->SetNumberField(TEXT("triangle_count"), Diagnostics.TriangleCount);
		Object->SetNumberField(TEXT("edge_count"), Diagnostics.EdgeCount);
		Object->SetNumberField(TEXT("boundary_edge_count"), Diagnostics.BoundaryEdgeCount);
		Object->SetNumberField(TEXT("non_manifold_edge_count"), Diagnostics.NonManifoldEdgeCount);
		Object->SetNumberField(TEXT("inconsistent_orientation_edge_count"), Diagnostics.InconsistentOrientationEdgeCount);
		Object->SetNumberField(TEXT("invalid_triangle_count"), Diagnostics.InvalidTriangleCount);
		Object->SetNumberField(TEXT("degenerate_triangle_count"), Diagnostics.DegenerateTriangleCount);
		Object->SetNumberField(TEXT("tiny_triangle_count"), Diagnostics.TinyTriangleCount);
		Object->SetNumberField(TEXT("duplicate_triangle_count"), Diagnostics.DuplicateTriangleCount);
		Object->SetNumberField(TEXT("surface_area"), Diagnostics.SurfaceArea);
		Object->SetNumberField(TEXT("min_triangle_area"), Diagnostics.MinTriangleArea);
		Object->SetNumberField(TEXT("max_triangle_area"), Diagnostics.MaxTriangleArea);
		Object->SetNumberField(TEXT("signed_volume"), Diagnostics.SignedVolume);
		Object->SetNumberField(TEXT("abs_volume"), Diagnostics.AbsVolume);
		Object->SetBoolField(TEXT("orientation_reversed_for_boolean"), Diagnostics.bOrientationReversedForBoolean);
		Object->SetNumberField(TEXT("signed_volume_before_orientation_fix"), Diagnostics.SignedVolumeBeforeOrientationFix);
		Object->SetNumberField(TEXT("signed_volume_after_orientation_fix"), Diagnostics.SignedVolumeAfterOrientationFix);
		Object->SetNumberField(TEXT("min_edge_length"), Diagnostics.MinEdgeLength);
		Object->SetNumberField(TEXT("max_edge_length"), Diagnostics.MaxEdgeLength);
		Object->SetNumberField(TEXT("mean_edge_length"), Diagnostics.MeanEdgeLength);
		Object->SetNumberField(TEXT("euler_characteristic"), Diagnostics.VertexCount - Diagnostics.EdgeCount + Diagnostics.TriangleCount);
		Object->SetBoolField(TEXT("closed_two_manifold"), Diagnostics.BoundaryEdgeCount == 0 && Diagnostics.NonManifoldEdgeCount == 0);
		Object->SetObjectField(TEXT("bounds"), Step11BoxJson(Diagnostics.Bounds));
		return Object;
	}

	static TSharedRef<FJsonObject> Step11BooleanAttemptJson(const FStep11BooleanAttemptResult& Attempt)
	{
		TSharedRef<FJsonObject> Object = MakeShared<FJsonObject>();
		Object->SetStringField(TEXT("pass_name"), Attempt.PassName);
		Object->SetStringField(TEXT("status"), Attempt.Status);
		Object->SetStringField(TEXT("reject_reason"), Attempt.RejectReason);
		Object->SetBoolField(TEXT("compute_success"), Attempt.bComputeSuccess);
		Object->SetBoolField(TEXT("accepted"), Attempt.bAccepted);
		Object->SetBoolField(TEXT("empty_result"), Attempt.bEmptyResult);
		Object->SetBoolField(TEXT("target_reversed_for_pass"), Attempt.bTargetReversedForPass);
		Object->SetBoolField(TEXT("cutter_reversed_for_pass"), Attempt.bCutterReversedForPass);
		Object->SetBoolField(TEXT("result_reversed_to_target_sign"), Attempt.bResultReversedToTargetSign);
		Object->SetNumberField(TEXT("volume_delta"), Attempt.VolumeDelta);
		Object->SetNumberField(TEXT("min_required_volume_delta"), Attempt.MinRequiredVolumeDelta);
		if (Attempt.bComputeSuccess && !Attempt.bEmptyResult)
		{
			Object->SetObjectField(TEXT("result"), Step11MeshDiagnosticsJson(Attempt.ResultDiagnostics));
		}
		return Object;
	}

	static void LogStep11MeshDiagnostics(const FStep11MeshDiagnostics& Diagnostics)
	{
		const FVector BoundsMin = Diagnostics.Bounds.IsValid ? Diagnostics.Bounds.Min : FVector::ZeroVector;
		const FVector BoundsMax = Diagnostics.Bounds.IsValid ? Diagnostics.Bounds.Max : FVector::ZeroVector;
		UE_LOG(
			LogTemp,
			Log,
			TEXT("Step11Diag: mesh label=%s source=%s vertices=%d triangles=%d edges=%d boundary=%d nonmanifold=%d inconsistent_orientation=%d invalid_triangles=%d degenerate=%d tiny=%d duplicate=%d area=%.6f signed_volume=%.6f abs_volume=%.6f orientation_reversed=%d signed_before=%.6f signed_after=%.6f min_edge=%.6f max_edge=%.6f mean_edge=%.6f bounds_min=(%.3f, %.3f, %.3f) bounds_max=(%.3f, %.3f, %.3f) closed_two_manifold=%d"),
			*Diagnostics.Label,
			*Diagnostics.SourceType,
			Diagnostics.VertexCount,
			Diagnostics.TriangleCount,
			Diagnostics.EdgeCount,
			Diagnostics.BoundaryEdgeCount,
			Diagnostics.NonManifoldEdgeCount,
			Diagnostics.InconsistentOrientationEdgeCount,
			Diagnostics.InvalidTriangleCount,
			Diagnostics.DegenerateTriangleCount,
			Diagnostics.TinyTriangleCount,
			Diagnostics.DuplicateTriangleCount,
			Diagnostics.SurfaceArea,
			Diagnostics.SignedVolume,
			Diagnostics.AbsVolume,
			Diagnostics.bOrientationReversedForBoolean ? 1 : 0,
			Diagnostics.SignedVolumeBeforeOrientationFix,
			Diagnostics.SignedVolumeAfterOrientationFix,
			Diagnostics.MinEdgeLength,
			Diagnostics.MaxEdgeLength,
			Diagnostics.MeanEdgeLength,
			BoundsMin.X,
			BoundsMin.Y,
			BoundsMin.Z,
			BoundsMax.X,
			BoundsMax.Y,
			BoundsMax.Z,
			Diagnostics.BoundaryEdgeCount == 0 && Diagnostics.NonManifoldEdgeCount == 0 ? 1 : 0);
	}

	static FString QuantizedVertexKey(const FVector& Position)
	{
		constexpr double WeldTolerance = 0.01;
		const int64 X = FMath::RoundToInt64(Position.X / WeldTolerance);
		const int64 Y = FMath::RoundToInt64(Position.Y / WeldTolerance);
		const int64 Z = FMath::RoundToInt64(Position.Z / WeldTolerance);
		return FString::Printf(TEXT("%lld_%lld_%lld"), X, Y, Z);
	}

	static bool BuildDynamicMeshFromStaticMeshComponent(UStaticMeshComponent* Component, UE::Geometry::FDynamicMesh3& OutMesh)
	{
		OutMesh = UE::Geometry::FDynamicMesh3();
		if (!Component || !Component->IsRegistered() || !Component->IsVisible())
		{
			return false;
		}

		UStaticMesh* StaticMesh = Component->GetStaticMesh();
		if (!StaticMesh || !StaticMesh->GetRenderData() || StaticMesh->GetRenderData()->LODResources.Num() == 0)
		{
			return false;
		}

		const FStaticMeshLODResources& LOD = StaticMesh->GetRenderData()->LODResources[0];
		const FPositionVertexBuffer& PositionBuffer = LOD.VertexBuffers.PositionVertexBuffer;
		const int32 NumRenderVertices = PositionBuffer.GetNumVertices();
		const FIndexArrayView Indices = LOD.IndexBuffer.GetArrayView();
		if (NumRenderVertices <= 0 || Indices.Num() < 3)
		{
			return false;
		}

		const FTransform& ComponentTransform = Component->GetComponentTransform();
		TMap<FString, int32> VertexIdByPosition;
		TArray<int32> RenderToDynamic;
		RenderToDynamic.Init(UE::Geometry::FDynamicMesh3::InvalidID, NumRenderVertices);
		for (int32 RenderVertexIndex = 0; RenderVertexIndex < NumRenderVertices; ++RenderVertexIndex)
		{
			const FVector3f LocalPosition = PositionBuffer.VertexPosition(RenderVertexIndex);
			const FVector WorldPosition = ComponentTransform.TransformPosition(
				FVector(double(LocalPosition.X), double(LocalPosition.Y), double(LocalPosition.Z)));
			const FString Key = QuantizedVertexKey(WorldPosition);
			if (const int32* ExistingId = VertexIdByPosition.Find(Key))
			{
				RenderToDynamic[RenderVertexIndex] = *ExistingId;
			}
			else
			{
				const int32 DynamicId = OutMesh.AppendVertex(FVector3d(WorldPosition.X, WorldPosition.Y, WorldPosition.Z));
				VertexIdByPosition.Add(Key, DynamicId);
				RenderToDynamic[RenderVertexIndex] = DynamicId;
			}
		}

		int32 AddedTriangles = 0;
		for (const FStaticMeshSection& Section : LOD.Sections)
		{
			for (uint32 TriIndex = 0; TriIndex < Section.NumTriangles; ++TriIndex)
			{
				const uint32 IndexBase = Section.FirstIndex + TriIndex * 3;
				if (IndexBase + 2 >= uint32(Indices.Num()))
				{
					continue;
				}

				const int32 A = int32(Indices[IndexBase]);
				const int32 B = int32(Indices[IndexBase + 1]);
				const int32 C = int32(Indices[IndexBase + 2]);
				if (!RenderToDynamic.IsValidIndex(A) || !RenderToDynamic.IsValidIndex(B) || !RenderToDynamic.IsValidIndex(C))
				{
					continue;
				}

				const int32 VA = RenderToDynamic[A];
				const int32 VB = RenderToDynamic[B];
				const int32 VC = RenderToDynamic[C];
				if (VA == VB || VB == VC || VC == VA)
				{
					continue;
				}

				if (OutMesh.AppendTriangle(VA, VB, VC) >= 0)
				{
					++AddedTriangles;
				}
			}
		}

		return AddedTriangles > 0;
	}

	static bool BuildDynamicMeshFromWorldMesh(
		const TArray<FVector>& Vertices,
		const TArray<int32>& Triangles,
		UE::Geometry::FDynamicMesh3& OutMesh)
	{
		OutMesh = UE::Geometry::FDynamicMesh3();
		if (Vertices.Num() < 3 || Triangles.Num() < 3)
		{
			return false;
		}

		TArray<int32> VertexIds;
		VertexIds.Reserve(Vertices.Num());
		for (const FVector& V : Vertices)
		{
			VertexIds.Add(OutMesh.AppendVertex(FVector3d(V.X, V.Y, V.Z)));
		}

		int32 AddedTriangles = 0;
		for (int32 i = 0; i + 2 < Triangles.Num(); i += 3)
		{
			const int32 A = Triangles[i];
			const int32 B = Triangles[i + 1];
			const int32 C = Triangles[i + 2];
			if (!VertexIds.IsValidIndex(A) || !VertexIds.IsValidIndex(B) || !VertexIds.IsValidIndex(C))
			{
				continue;
			}
			if (OutMesh.AppendTriangle(VertexIds[A], VertexIds[B], VertexIds[C]) >= 0)
			{
				++AddedTriangles;
			}
		}
		return AddedTriangles > 0;
	}

	static bool BuildDynamicMeshFromProceduralMeshComponent(UProceduralMeshComponent* Component, UE::Geometry::FDynamicMesh3& OutMesh)
	{
		OutMesh = UE::Geometry::FDynamicMesh3();
		if (!Component || !Component->IsRegistered() || !Component->IsVisible())
		{
			return false;
		}

		const FTransform& ComponentTransform = Component->GetComponentTransform();
		TMap<FString, int32> VertexIdByPosition;
		int32 AddedTriangles = 0;
		for (int32 SectionIndex = 0; SectionIndex < Component->GetNumSections(); ++SectionIndex)
		{
			if (!Component->IsMeshSectionVisible(SectionIndex))
			{
				continue;
			}

			const FProcMeshSection* Section = Component->GetProcMeshSection(SectionIndex);
			if (!Section || Section->ProcVertexBuffer.Num() < 3 || Section->ProcIndexBuffer.Num() < 3)
			{
				continue;
			}

			TArray<int32> SectionVertexIds;
			SectionVertexIds.Reserve(Section->ProcVertexBuffer.Num());
			for (const FProcMeshVertex& Vertex : Section->ProcVertexBuffer)
			{
				const FVector WorldPosition = ComponentTransform.TransformPosition(Vertex.Position);
				const FString Key = QuantizedVertexKey(WorldPosition);
				if (const int32* ExistingId = VertexIdByPosition.Find(Key))
				{
					SectionVertexIds.Add(*ExistingId);
				}
				else
				{
					const int32 DynamicId = OutMesh.AppendVertex(FVector3d(WorldPosition.X, WorldPosition.Y, WorldPosition.Z));
					VertexIdByPosition.Add(Key, DynamicId);
					SectionVertexIds.Add(DynamicId);
				}
			}

			for (int32 Index = 0; Index + 2 < Section->ProcIndexBuffer.Num(); Index += 3)
			{
				const int32 A = int32(Section->ProcIndexBuffer[Index]);
				const int32 B = int32(Section->ProcIndexBuffer[Index + 1]);
				const int32 C = int32(Section->ProcIndexBuffer[Index + 2]);
				if (!SectionVertexIds.IsValidIndex(A) || !SectionVertexIds.IsValidIndex(B) || !SectionVertexIds.IsValidIndex(C))
				{
					continue;
				}

				const int32 VA = SectionVertexIds[A];
				const int32 VB = SectionVertexIds[B];
				const int32 VC = SectionVertexIds[C];
				if (VA == VB || VB == VC || VC == VA)
				{
					continue;
				}

				if (OutMesh.AppendTriangle(VA, VB, VC) >= 0)
				{
					++AddedTriangles;
				}
			}
		}

		return AddedTriangles > 0;
	}

	static bool ConvertDynamicMeshToProceduralArrays(
		const UE::Geometry::FDynamicMesh3& Mesh,
		TArray<FVector>& OutVertices,
		TArray<int32>& OutTriangles,
		TArray<FVector>& OutNormals)
	{
		OutVertices.Reset();
		OutTriangles.Reset();
		OutNormals.Reset();
		if (Mesh.VertexCount() <= 0 || Mesh.TriangleCount() <= 0)
		{
			return false;
		}

		TMap<int32, int32> CompactVertexIndex;
		CompactVertexIndex.Reserve(Mesh.VertexCount());
		OutVertices.Reserve(Mesh.VertexCount());
		for (int32 VertexId : Mesh.VertexIndicesItr())
		{
			const FVector3d V = Mesh.GetVertex(VertexId);
			CompactVertexIndex.Add(VertexId, OutVertices.Num());
			OutVertices.Emplace(V.X, V.Y, V.Z);
		}

		OutTriangles.Reserve(Mesh.TriangleCount() * 3);
		OutNormals.Init(FVector::ZeroVector, OutVertices.Num());
		for (int32 TriangleId : Mesh.TriangleIndicesItr())
		{
			const UE::Geometry::FIndex3i Tri = Mesh.GetTriangle(TriangleId);
			const int32* A = CompactVertexIndex.Find(Tri.A);
			const int32* B = CompactVertexIndex.Find(Tri.B);
			const int32* C = CompactVertexIndex.Find(Tri.C);
			if (!A || !B || !C)
			{
				continue;
			}

			OutTriangles.Add(*A);
			OutTriangles.Add(*B);
			OutTriangles.Add(*C);
			const FVector& VA = OutVertices[*A];
			const FVector& VB = OutVertices[*B];
			const FVector& VC = OutVertices[*C];
			const FVector TriNormal = FVector::CrossProduct(VB - VA, VC - VA).GetSafeNormal();
			OutNormals[*A] += TriNormal;
			OutNormals[*B] += TriNormal;
			OutNormals[*C] += TriNormal;
		}

		for (FVector& Normal : OutNormals)
		{
			Normal = Normal.GetSafeNormal();
			if (Normal.IsNearlyZero())
			{
				Normal = FVector::UpVector;
			}
		}
		return OutVertices.Num() >= 3 && OutTriangles.Num() >= 3;
	}

	struct FStep11ProcMeshSectionData
	{
		TArray<FVector> Vertices;
		TArray<int32> Triangles;
		TArray<FVector> Normals;
		TArray<FVector2D> UV0;
		TArray<FColor> Colors;
		TArray<FProcMeshTangent> Tangents;

		bool HasGeometry() const
		{
			return Vertices.Num() >= 3 && Triangles.Num() >= 3;
		}
	};

	static FVector Step11StableTangent(const FVector& Normal)
	{
		FVector Tangent = FVector::CrossProduct(FVector::UpVector, Normal);
		if (Tangent.IsNearlyZero())
		{
			Tangent = FVector::CrossProduct(FVector::RightVector, Normal);
		}
		Tangent.Normalize();
		return Tangent.IsNearlyZero() ? FVector::ForwardVector : Tangent;
	}

	static FVector2D Step11PlanarUV(const FVector& WorldPosition, const FVector& Normal)
	{
		constexpr double UvScaleCm = 100.0;
		const FVector AbsNormal(FMath::Abs(Normal.X), FMath::Abs(Normal.Y), FMath::Abs(Normal.Z));
		if (AbsNormal.Z >= AbsNormal.X && AbsNormal.Z >= AbsNormal.Y)
		{
			return FVector2D(WorldPosition.X / UvScaleCm, WorldPosition.Y / UvScaleCm);
		}
		if (AbsNormal.Y >= AbsNormal.X)
		{
			return FVector2D(WorldPosition.X / UvScaleCm, WorldPosition.Z / UvScaleCm);
		}
		return FVector2D(WorldPosition.Y / UvScaleCm, WorldPosition.Z / UvScaleCm);
	}

	static void AppendStep11FlatTriangle(
		FStep11ProcMeshSectionData& Section,
		const FVector& AWorld,
		const FVector& BWorld,
		const FVector& CWorld,
		const FVector& Origin,
		const FColor& Color)
	{
		FVector Normal = FVector::CrossProduct(BWorld - AWorld, CWorld - AWorld).GetSafeNormal();
		if (Normal.IsNearlyZero())
		{
			Normal = FVector::UpVector;
		}
		const FVector Tangent = Step11StableTangent(Normal);
		const int32 BaseIndex = Section.Vertices.Num();
		Section.Vertices.Add(AWorld - Origin);
		Section.Vertices.Add(BWorld - Origin);
		Section.Vertices.Add(CWorld - Origin);
		Section.Triangles.Add(BaseIndex);
		Section.Triangles.Add(BaseIndex + 1);
		Section.Triangles.Add(BaseIndex + 2);
		Section.Normals.Add(Normal);
		Section.Normals.Add(Normal);
		Section.Normals.Add(Normal);
		Section.UV0.Add(Step11PlanarUV(AWorld, Normal));
		Section.UV0.Add(Step11PlanarUV(BWorld, Normal));
		Section.UV0.Add(Step11PlanarUV(CWorld, Normal));
		Section.Colors.Add(Color);
		Section.Colors.Add(Color);
		Section.Colors.Add(Color);
		Section.Tangents.Add(FProcMeshTangent(Tangent, false));
		Section.Tangents.Add(FProcMeshTangent(Tangent, false));
		Section.Tangents.Add(FProcMeshTangent(Tangent, false));
	}

	static bool ConvertDynamicMeshToFlatProceduralSections(
		const UE::Geometry::FDynamicMesh3& Mesh,
		const FVector& Origin,
		const TArray<int8>* TriangleSourceMeshById,
		FStep11ProcMeshSectionData& OutSourceSection,
		FStep11ProcMeshSectionData& OutCapSection)
	{
		OutSourceSection = FStep11ProcMeshSectionData();
		OutCapSection = FStep11ProcMeshSectionData();
		if (Mesh.TriangleCount() <= 0)
		{
			return false;
		}

		for (int32 TriangleId : Mesh.TriangleIndicesItr())
		{
			const UE::Geometry::FIndex3i Tri = Mesh.GetTriangle(TriangleId);
			if (!Mesh.IsVertex(Tri.A) || !Mesh.IsVertex(Tri.B) || !Mesh.IsVertex(Tri.C))
			{
				continue;
			}

			const FVector3d A3 = Mesh.GetVertex(Tri.A);
			const FVector3d B3 = Mesh.GetVertex(Tri.B);
			const FVector3d C3 = Mesh.GetVertex(Tri.C);
			const FVector A(A3.X, A3.Y, A3.Z);
			const FVector B(B3.X, B3.Y, B3.Z);
			const FVector C(C3.X, C3.Y, C3.Z);
			if (FVector::CrossProduct(B - A, C - A).IsNearlyZero())
			{
				continue;
			}

			const bool bFromCutter = TriangleSourceMeshById &&
				TriangleSourceMeshById->IsValidIndex(TriangleId) &&
				(*TriangleSourceMeshById)[TriangleId] == 1;
			AppendStep11FlatTriangle(
				bFromCutter ? OutCapSection : OutSourceSection,
				A,
				B,
				C,
				Origin,
				bFromCutter ? FColor(180, 180, 180, 255) : FColor::White);
		}

		return OutSourceSection.HasGeometry() || OutCapSection.HasGeometry();
	}

	static FStep11BooleanAttemptResult RunStep11BooleanDifferencePass(
		const UE::Geometry::FDynamicMesh3& CurrentMesh,
		const UE::Geometry::FDynamicMesh3& CutterMesh,
		const FStep11MeshDiagnostics& TargetBeforeDiagnostics,
		int32 TargetRenderSign,
		bool bReverseTargetForPass,
		const FString& PassName,
		const FString& ResultLabel)
	{
		FStep11BooleanAttemptResult Attempt;
		Attempt.PassName = PassName;
		Attempt.MinRequiredVolumeDelta = FMath::Max(1.0, TargetBeforeDiagnostics.AbsVolume * 1e-4);
		if (TargetBeforeDiagnostics.AbsVolume <= 1e-6 || TargetRenderSign == 0)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("target_zero_volume");
			return Attempt;
		}

		UE::Geometry::FDynamicMesh3 TargetWork = CurrentMesh;
		UE::Geometry::FDynamicMesh3 CutterWork = CutterMesh;
		const int32 WorkingTargetSign = bReverseTargetForPass ? -TargetRenderSign : TargetRenderSign;
		if (bReverseTargetForPass)
		{
			TargetWork.ReverseOrientation(false);
			Attempt.bTargetReversedForPass = true;
		}
		Attempt.bCutterReversedForPass = ReverseStep11MeshIfSignMismatch(CutterWork, WorkingTargetSign);

		UE::Geometry::FMeshBoolean MeshBoolean(
			&TargetWork,
			&CutterWork,
			&Attempt.ResultMesh,
			UE::Geometry::FMeshBoolean::EBooleanOp::Difference);
		MeshBoolean.SnapTolerance = Step11BooleanSnapToleranceCm;
		MeshBoolean.DegenerateEdgeTolFactor = 2.0;
		MeshBoolean.bCollapseDegenerateEdgesOnCut = true;
		MeshBoolean.bPutResultInInputSpace = true;
		MeshBoolean.bWeldSharedEdges = true;
		MeshBoolean.bTrackAllNewEdges = true;
		MeshBoolean.bSimplifyAlongNewEdges = true;
		MeshBoolean.SimplificationAngleTolerance = 0.25;
		MeshBoolean.TryToImproveTriQualityThreshold = 0.2;
		MeshBoolean.bPreserveTriangleGroups = false;
		MeshBoolean.bPreserveVertexUVs = false;
		MeshBoolean.bPreserveOverlayUVs = false;
		MeshBoolean.bPreserveVertexNormals = false;
		MeshBoolean.TrackPerTriangleSourceMesh.Emplace();

		Attempt.bComputeSuccess = MeshBoolean.Compute();
		if (!Attempt.bComputeSuccess)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("compute_failed");
			return Attempt;
		}

		if (Attempt.ResultMesh.TriangleCount() <= 0)
		{
			Attempt.bAccepted = true;
			Attempt.bEmptyResult = true;
			Attempt.Status = TEXT("success_empty_result");
			Attempt.VolumeDelta = TargetBeforeDiagnostics.AbsVolume;
			return Attempt;
		}

		if (MeshBoolean.TrackPerTriangleSourceMesh.IsSet())
		{
			Attempt.TriangleSourceMeshById = MoveTemp(MeshBoolean.TrackPerTriangleSourceMesh.GetValue());
		}

		Attempt.ResultDiagnostics = AnalyzeStep11DynamicMesh(Attempt.ResultMesh, ResultLabel, TEXT("boolean_result"));
		const int32 ResultSign = Step11SignedVolumeSign(Attempt.ResultDiagnostics.SignedVolume);
		if (ResultSign != 0 && ResultSign != TargetRenderSign)
		{
			Attempt.ResultMesh.ReverseOrientation(false);
			Attempt.bResultReversedToTargetSign = true;
			Attempt.ResultDiagnostics = AnalyzeStep11DynamicMesh(Attempt.ResultMesh, ResultLabel, TEXT("boolean_result"));
		}

		Attempt.VolumeDelta = TargetBeforeDiagnostics.AbsVolume - Attempt.ResultDiagnostics.AbsVolume;
		if (Attempt.ResultDiagnostics.BoundaryEdgeCount > 0)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("open_boundary");
			return Attempt;
		}
		if (Attempt.ResultDiagnostics.NonManifoldEdgeCount > 0)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("nonmanifold_result");
			return Attempt;
		}
		if (Attempt.VolumeDelta < Attempt.MinRequiredVolumeDelta)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("no_volume_removed");
			return Attempt;
		}
		if (Attempt.ResultDiagnostics.MinEdgeLength > 0.0 &&
			Attempt.ResultDiagnostics.MinEdgeLength < Step11BooleanMinRenderableEdgeCm)
		{
			Attempt.Status = TEXT("rejected");
			Attempt.RejectReason = TEXT("tiny_edges_after_boolean");
			return Attempt;
		}

		Attempt.bAccepted = true;
		Attempt.Status = TEXT("accepted");
		return Attempt;
	}

	static bool ApplyCuttersToDynamicMesh(
		UE::Geometry::FDynamicMesh3& CurrentMesh,
		const FString& TargetActorName,
		const FString& TargetComponentName,
		const FString& TargetSourceType,
		const TArray<const FReconstructedMesh*>& Cutters,
		const TArray<UE::Geometry::FDynamicMesh3>& CutterMeshes,
		const TArray<FBox>& CutterBounds,
		const TArray<FStep11MeshDiagnostics>& CutterDiagnostics,
		TArray<TSharedPtr<FJsonValue>>* OutOperationDiagnostics,
		TArray<int8>& OutFinalTriangleSourceMeshById,
		TSet<FString>& OutAcceptedCutterActorNames,
		bool& bOutEmptyResult,
		int32& FailedBooleanCount)
	{
		bOutEmptyResult = false;
		bool bModified = false;
		for (int32 CutterIndex = 0; CutterIndex < CutterMeshes.Num(); ++CutterIndex)
		{
			const FString TargetLabel = FString::Printf(TEXT("%s/%s"), *TargetActorName, *TargetComponentName);
			FStep11MeshDiagnostics TargetBeforeDiagnostics = AnalyzeStep11DynamicMesh(CurrentMesh, TargetLabel, TargetSourceType);
			const FBox TargetBounds = TargetBeforeDiagnostics.Bounds;
			const FBox ExpandedTargetBounds = TargetBounds.ExpandBy(Step11BooleanBoundsExpandCm);
			const FBox ExpandedCutterBounds = CutterBounds[CutterIndex].ExpandBy(Step11BooleanBoundsExpandCm);
			const bool bBoundsIntersect = TargetBounds.Intersect(CutterBounds[CutterIndex]);
			const bool bExpandedBoundsIntersect = ExpandedTargetBounds.Intersect(ExpandedCutterBounds);
			const double CenterDistance = (TargetBounds.IsValid && CutterBounds[CutterIndex].IsValid)
				? FVector::Distance(TargetBounds.GetCenter(), CutterBounds[CutterIndex].GetCenter())
				: 0.0;
			const int32 TargetRenderSign = Step11SignedVolumeSign(TargetBeforeDiagnostics.SignedVolume);

			TSharedRef<FJsonObject> OperationJson = MakeShared<FJsonObject>();
			OperationJson->SetStringField(TEXT("target_actor"), TargetActorName);
			OperationJson->SetStringField(TEXT("target_component"), TargetComponentName);
			OperationJson->SetStringField(TEXT("target_source_type"), TargetSourceType);
			OperationJson->SetStringField(TEXT("cutter_actor"), Cutters[CutterIndex]->ActorName);
			OperationJson->SetNumberField(TEXT("cutter_index"), CutterIndex);
			OperationJson->SetBoolField(TEXT("bounds_intersect"), bBoundsIntersect);
			OperationJson->SetBoolField(TEXT("expanded_bounds_intersect"), bExpandedBoundsIntersect);
			OperationJson->SetNumberField(TEXT("center_distance"), CenterDistance);
			OperationJson->SetObjectField(TEXT("target_bounds"), Step11BoxJson(TargetBounds));
			OperationJson->SetObjectField(TEXT("cutter_bounds"), Step11BoxJson(CutterBounds[CutterIndex]));
			OperationJson->SetObjectField(TEXT("target_before"), Step11MeshDiagnosticsJson(TargetBeforeDiagnostics));
			OperationJson->SetNumberField(TEXT("target_render_sign"), TargetRenderSign);
			OperationJson->SetNumberField(TEXT("snap_tolerance_cm"), Step11BooleanSnapToleranceCm);
			OperationJson->SetNumberField(TEXT("min_renderable_edge_cm"), Step11BooleanMinRenderableEdgeCm);
			if (CutterDiagnostics.IsValidIndex(CutterIndex))
			{
				OperationJson->SetObjectField(TEXT("cutter_diagnostics"), Step11MeshDiagnosticsJson(CutterDiagnostics[CutterIndex]));
			}

			UE_LOG(
				LogTemp,
				Log,
				TEXT("Step11Diag: pair target=%s/%s source=%s cutter=%s bounds_intersect=%d expanded_bounds_intersect=%d center_distance=%.6f target_triangles=%d cutter_triangles=%d"),
				*TargetActorName,
				*TargetComponentName,
				*TargetSourceType,
				*Cutters[CutterIndex]->ActorName,
				bBoundsIntersect ? 1 : 0,
				bExpandedBoundsIntersect ? 1 : 0,
				CenterDistance,
				TargetBeforeDiagnostics.TriangleCount,
				CutterDiagnostics.IsValidIndex(CutterIndex) ? CutterDiagnostics[CutterIndex].TriangleCount : CutterMeshes[CutterIndex].TriangleCount());

			if (!bExpandedBoundsIntersect)
			{
				OperationJson->SetStringField(TEXT("status"), TEXT("skipped_bounds_no_intersection"));
				if (OutOperationDiagnostics)
				{
					OutOperationDiagnostics->Add(MakeShared<FJsonValueObject>(OperationJson));
				}
				continue;
			}

			TArray<TSharedPtr<FJsonValue>> AttemptDiagnostics;
			FStep11BooleanAttemptResult PrimaryAttempt = RunStep11BooleanDifferencePass(
				CurrentMesh,
				CutterMeshes[CutterIndex],
				TargetBeforeDiagnostics,
				TargetRenderSign,
				false,
				TEXT("primary_target_render_orientation"),
				FString::Printf(TEXT("%s/%s result_after_%s_primary"), *TargetActorName, *TargetComponentName, *Cutters[CutterIndex]->ActorName));
			AttemptDiagnostics.Add(MakeShared<FJsonValueObject>(Step11BooleanAttemptJson(PrimaryAttempt)));
			FStep11BooleanAttemptResult AcceptedAttempt = MoveTemp(PrimaryAttempt);
			if (!AcceptedAttempt.bAccepted)
			{
				FStep11BooleanAttemptResult FallbackAttempt = RunStep11BooleanDifferencePass(
					CurrentMesh,
					CutterMeshes[CutterIndex],
					TargetBeforeDiagnostics,
					TargetRenderSign,
					true,
					TEXT("fallback_reversed_pair_orientation"),
					FString::Printf(TEXT("%s/%s result_after_%s_fallback"), *TargetActorName, *TargetComponentName, *Cutters[CutterIndex]->ActorName));
				AttemptDiagnostics.Add(MakeShared<FJsonValueObject>(Step11BooleanAttemptJson(FallbackAttempt)));
				if (FallbackAttempt.bAccepted)
				{
					AcceptedAttempt = MoveTemp(FallbackAttempt);
				}
			}

			OperationJson->SetArrayField(TEXT("attempts"), AttemptDiagnostics);
			if (AcceptedAttempt.bAccepted && !AcceptedAttempt.bEmptyResult)
			{
				OperationJson->SetStringField(TEXT("status"), TEXT("accepted_non_empty"));
				OperationJson->SetObjectField(TEXT("accepted_result"), Step11MeshDiagnosticsJson(AcceptedAttempt.ResultDiagnostics));
				OperationJson->SetStringField(TEXT("accepted_pass"), AcceptedAttempt.PassName);
				OperationJson->SetNumberField(TEXT("volume_delta"), AcceptedAttempt.VolumeDelta);
				OperationJson->SetNumberField(TEXT("min_required_volume_delta"), AcceptedAttempt.MinRequiredVolumeDelta);
				LogStep11MeshDiagnostics(AcceptedAttempt.ResultDiagnostics);
				UE_LOG(
					LogTemp,
					Log,
					TEXT("Step11Diag: boolean accepted target=%s/%s cutter=%s pass=%s result_triangles=%d volume_delta=%.6f result_boundary=%d result_nonmanifold=%d"),
					*TargetActorName,
					*TargetComponentName,
					*Cutters[CutterIndex]->ActorName,
					*AcceptedAttempt.PassName,
					AcceptedAttempt.ResultDiagnostics.TriangleCount,
					AcceptedAttempt.VolumeDelta,
					AcceptedAttempt.ResultDiagnostics.BoundaryEdgeCount,
					AcceptedAttempt.ResultDiagnostics.NonManifoldEdgeCount);
				CurrentMesh = MoveTemp(AcceptedAttempt.ResultMesh);
				OutFinalTriangleSourceMeshById = MoveTemp(AcceptedAttempt.TriangleSourceMeshById);
				OutAcceptedCutterActorNames.Add(Cutters[CutterIndex]->ActorName);
				bModified = true;
			}
			else if (AcceptedAttempt.bAccepted)
			{
				OperationJson->SetStringField(TEXT("status"), TEXT("accepted_empty_result"));
				OperationJson->SetStringField(TEXT("accepted_pass"), AcceptedAttempt.PassName);
				UE_LOG(
					LogTemp,
					Log,
					TEXT("Step11Diag: boolean accepted empty result target=%s/%s cutter=%s pass=%s."),
					*TargetActorName,
					*TargetComponentName,
					*Cutters[CutterIndex]->ActorName,
					*AcceptedAttempt.PassName);
				OutAcceptedCutterActorNames.Add(Cutters[CutterIndex]->ActorName);
				bModified = true;
				bOutEmptyResult = true;
				if (OutOperationDiagnostics)
				{
					OutOperationDiagnostics->Add(MakeShared<FJsonValueObject>(OperationJson));
				}
				break;
			}
			else
			{
				++FailedBooleanCount;
				OperationJson->SetStringField(TEXT("status"), TEXT("rejected"));
				OperationJson->SetStringField(TEXT("reject_reason"), AcceptedAttempt.RejectReason.IsEmpty() ? TEXT("compute_failed") : AcceptedAttempt.RejectReason);
				UE_LOG(
					LogTemp,
					Warning,
					TEXT("Step11: boolean subtract rejected for actor=%s component=%s cutter=%s reason=%s."),
					*TargetActorName,
					*TargetComponentName,
					*Cutters[CutterIndex]->ActorName,
					*OperationJson->GetStringField(TEXT("reject_reason")));
			}

			if (OutOperationDiagnostics)
			{
				OutOperationDiagnostics->Add(MakeShared<FJsonValueObject>(OperationJson));
			}
		}

		return bModified;
	}

	static bool UpdateBooleanResultComponent(UProceduralMeshComponent* Component, const UE::Geometry::FDynamicMesh3& Mesh)
	{
		if (!Component)
		{
			return false;
		}

		TArray<FVector> Vertices;
		TArray<int32> Triangles;
		TArray<FVector> Normals;
		if (!ConvertDynamicMeshToProceduralArrays(Mesh, Vertices, Triangles, Normals))
		{
			return false;
		}

		UMaterialInterface* Material = Component->GetMaterial(0);
		const ECollisionEnabled::Type CollisionEnabled = Component->GetCollisionEnabled();
		Component->ClearAllMeshSections();
		Component->SetWorldTransform(FTransform::Identity, false, nullptr, ETeleportType::TeleportPhysics);

		TArray<FVector2D> UV0;
		UV0.Init(FVector2D::ZeroVector, Vertices.Num());
		TArray<FColor> Colors;
		Colors.Init(FColor::White, Vertices.Num());
		TArray<FProcMeshTangent> Tangents;
		Component->CreateMeshSection(0, Vertices, Triangles, Normals, UV0, Colors, Tangents, CollisionEnabled != ECollisionEnabled::NoCollision);
		Component->SetMaterial(0, Material);
		return true;
	}

	static UProceduralMeshComponent* CreateBooleanResultComponent(
		AActor* Owner,
		UStaticMeshComponent* SourceComponent,
		const UE::Geometry::FDynamicMesh3& Mesh)
	{
		if (!Owner || !SourceComponent)
		{
			return nullptr;
		}

		TArray<FVector> Vertices;
		TArray<int32> Triangles;
		TArray<FVector> Normals;
		if (!ConvertDynamicMeshToProceduralArrays(Mesh, Vertices, Triangles, Normals))
		{
			return nullptr;
		}

		const FName ComponentName = MakeUniqueObjectName(Owner, UProceduralMeshComponent::StaticClass(), TEXT("FromLZ_Step11BooleanMesh"));
		UProceduralMeshComponent* ResultComponent = NewObject<UProceduralMeshComponent>(Owner, ComponentName);
		if (!ResultComponent)
		{
			return nullptr;
		}

		ResultComponent->ComponentTags.AddUnique(Step11BooleanResultTag);
		ResultComponent->SetMobility(EComponentMobility::Movable);
		ResultComponent->SetCollisionEnabled(SourceComponent->GetCollisionEnabled());
		ResultComponent->bUseAsyncCooking = false;
		ResultComponent->SetCastShadow(SourceComponent->CastShadow);
		Owner->AddInstanceComponent(ResultComponent);

		TArray<FVector2D> UV0;
		UV0.Init(FVector2D::ZeroVector, Vertices.Num());
		TArray<FColor> Colors;
		Colors.Init(FColor::White, Vertices.Num());
		TArray<FProcMeshTangent> Tangents;
		ResultComponent->CreateMeshSection(0, Vertices, Triangles, Normals, UV0, Colors, Tangents, SourceComponent->GetCollisionEnabled() != ECollisionEnabled::NoCollision);
		ResultComponent->SetMaterial(0, SourceComponent->GetMaterial(0));
		ResultComponent->RegisterComponent();
		ResultComponent->SetWorldTransform(FTransform::Identity);
		return ResultComponent;
	}

	static AActor* CreateBooleanResultActor(
		UWorld* World,
		UPrimitiveComponent* SourceComponent,
		const FString& SourceActorName,
		const FString& SourceComponentName,
		const FString& PressId,
		const UE::Geometry::FDynamicMesh3& Mesh,
		const TArray<int8>& TriangleSourceMeshById)
	{
		if (!World || !SourceComponent || Mesh.TriangleCount() <= 0)
		{
			return nullptr;
		}

		const FBox Bounds = BuildDynamicMeshBounds(Mesh);
		const FVector Origin = Bounds.IsValid ? Bounds.GetCenter() : FVector::ZeroVector;
		FStep11ProcMeshSectionData SourceSection;
		FStep11ProcMeshSectionData CapSection;
		const TArray<int8>* SourceMapPtr = TriangleSourceMeshById.Num() > 0 ? &TriangleSourceMeshById : nullptr;
		if (!ConvertDynamicMeshToFlatProceduralSections(Mesh, Origin, SourceMapPtr, SourceSection, CapSection))
		{
			return nullptr;
		}

		const FString BaseName = ObjSafeName(FString::Printf(TEXT("FromLZ_BooleanResult_%s_%s"), *SourceActorName, *SourceComponentName));
		FActorSpawnParameters Params;
		Params.Name = MakeUniqueObjectName(World->GetCurrentLevel(), AActor::StaticClass(), FName(*BaseName));
		AActor* ResultActor = World->SpawnActor<AActor>(AActor::StaticClass(), FTransform(Origin), Params);
		if (!ResultActor)
		{
			return nullptr;
		}

		ResultActor->Tags.AddUnique(Step11BooleanResultTag);
		ResultActor->Tags.AddUnique(Step11PressTag(PressId));
#if WITH_EDITOR
		ResultActor->SetActorLabel(BaseName);
#endif

		UProceduralMeshComponent* MeshComponent = NewObject<UProceduralMeshComponent>(ResultActor, TEXT("FromLZ_Step11BooleanMesh"));
		if (!MeshComponent)
		{
			ResultActor->Destroy();
			return nullptr;
		}

		MeshComponent->ComponentTags.AddUnique(Step11BooleanResultTag);
		MeshComponent->ComponentTags.AddUnique(Step11PressTag(PressId));
		MeshComponent->SetMobility(EComponentMobility::Movable);
		MeshComponent->SetCollisionEnabled(SourceComponent->GetCollisionEnabled());
		MeshComponent->bUseAsyncCooking = false;
		MeshComponent->SetCastShadow(SourceComponent->CastShadow);
		ResultActor->SetRootComponent(MeshComponent);
		ResultActor->AddInstanceComponent(MeshComponent);
		MeshComponent->RegisterComponent();

		if (SourceSection.HasGeometry())
		{
			MeshComponent->CreateMeshSection(
				0,
				SourceSection.Vertices,
				SourceSection.Triangles,
				SourceSection.Normals,
				SourceSection.UV0,
				SourceSection.Colors,
				SourceSection.Tangents,
				SourceComponent->GetCollisionEnabled() != ECollisionEnabled::NoCollision);
			MeshComponent->SetMaterial(0, SourceComponent->GetMaterial(0));
		}
		if (CapSection.HasGeometry())
		{
			const int32 SectionIndex = SourceSection.HasGeometry() ? 1 : 0;
			MeshComponent->CreateMeshSection(
				SectionIndex,
				CapSection.Vertices,
				CapSection.Triangles,
				CapSection.Normals,
				CapSection.UV0,
				CapSection.Colors,
				CapSection.Tangents,
				SourceComponent->GetCollisionEnabled() != ECollisionEnabled::NoCollision);
			MeshComponent->SetMaterial(SectionIndex, GetReconstructionVertexColorMaterial());
		}

		ResultActor->SetActorLocation(Origin, false, nullptr, ETeleportType::TeleportPhysics);
		return ResultActor;
	}

	static void HideStep11SourceComponentForPress(UPrimitiveComponent* Component, const FString& PressId)
	{
		if (!Component)
		{
			return;
		}
		Component->ComponentTags.AddUnique(Step11HiddenSourceTag);
		Component->ComponentTags.AddUnique(Step11PressTag(PressId));
		Component->SetVisibility(false, true);
		Component->SetHiddenInGame(true, true);
	}

	static void RestoreStep11BooleanResults(UWorld* World, const FString& PressId)
	{
		if (!World)
		{
			return;
		}

		if (PressId.IsEmpty())
		{
			UE_LOG(LogTemp, Warning, TEXT("Step11: no active press id; skipped press-scoped restore."));
			return;
		}

		const FName PressTag = Step11PressTag(PressId);
		const FString UndoDiagnosticPath = FPaths::ProjectSavedDir() / TEXT("2DDebug") / PressId / TEXT("11_undo_diagnostics.json");
		TSharedRef<FJsonObject> UndoRoot = MakeShared<FJsonObject>();
		UndoRoot->SetNumberField(TEXT("diagnostic_version"), 1);
		UndoRoot->SetStringField(TEXT("press_id"), PressId);
		UndoRoot->SetStringField(TEXT("active_undo_press_id"), GActiveUndoPressId);

		TArray<AActor*> ActorsToDestroy;
		TArray<UProceduralMeshComponent*> GeneratedComponents;
		TArray<UPrimitiveComponent*> HiddenSourceComponents;
		int32 SkippedOtherPressActorCount = 0;
		int32 SkippedOtherPressComponentCount = 0;
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (!Actor)
			{
				continue;
			}

			const bool bCurrentPressActor = Actor->ActorHasTag(PressTag);
			if (bCurrentPressActor && (
				Actor->ActorHasTag(Step11BooleanResultTag) ||
				Actor->ActorHasTag(Step11ActionAttachTag) ||
				Actor->ActorHasTag(Step11ActionExcavateCutterTag)))
			{
				ActorsToDestroy.Add(Actor);
				continue;
			}
			if (!bCurrentPressActor && ActorHasAnyStep11RuntimeTag(Actor))
			{
				++SkippedOtherPressActorCount;
			}

			TArray<UProceduralMeshComponent*> ProceduralComponents;
			Actor->GetComponents<UProceduralMeshComponent>(ProceduralComponents);
			for (UProceduralMeshComponent* Component : ProceduralComponents)
			{
				if (Component &&
					Component->ComponentTags.Contains(Step11BooleanResultTag) &&
					Component->ComponentTags.Contains(PressTag))
				{
					GeneratedComponents.Add(Component);
				}
				else if (Component && Component->ComponentTags.Contains(Step11BooleanResultTag))
				{
					++SkippedOtherPressComponentCount;
				}
			}

			TArray<UPrimitiveComponent*> PrimitiveComponents;
			Actor->GetComponents<UPrimitiveComponent>(PrimitiveComponents);
			for (UPrimitiveComponent* Component : PrimitiveComponents)
			{
				if (Component &&
					Component->ComponentTags.Contains(Step11HiddenSourceTag) &&
					Component->ComponentTags.Contains(PressTag))
				{
					HiddenSourceComponents.Add(Component);
				}
				else if (Component && Component->ComponentTags.Contains(Step11HiddenSourceTag))
				{
					++SkippedOtherPressComponentCount;
				}
			}
		}

		for (UProceduralMeshComponent* Component : GeneratedComponents)
		{
			if (Component)
			{
				Component->DestroyComponent();
			}
		}
		for (AActor* Actor : ActorsToDestroy)
		{
			if (Actor)
			{
				Actor->Destroy();
			}
		}
		for (UPrimitiveComponent* Component : HiddenSourceComponents)
		{
			if (Component)
			{
				Component->SetVisibility(true, true);
				Component->SetHiddenInGame(false, true);
				Component->ComponentTags.Remove(Step11HiddenSourceTag);
				Component->ComponentTags.Remove(PressTag);
			}
		}
		UndoRoot->SetNumberField(TEXT("destroyed_actor_count"), ActorsToDestroy.Num());
		UndoRoot->SetNumberField(TEXT("destroyed_boolean_component_count"), GeneratedComponents.Num());
		UndoRoot->SetNumberField(TEXT("restored_hidden_source_count"), HiddenSourceComponents.Num());
		UndoRoot->SetNumberField(TEXT("skipped_other_press_actor_count"), SkippedOtherPressActorCount);
		UndoRoot->SetNumberField(TEXT("skipped_other_press_component_count"), SkippedOtherPressComponentCount);
		SaveJsonObject(UndoRoot, UndoDiagnosticPath);
		UE_LOG(
			LogTemp,
			Log,
			TEXT("Step11: press-scoped restore press=%s restored %d hidden source component(s), removed %d boolean result component(s), destroyed %d current press actor(s), skipped_other_press_actors=%d skipped_other_press_components=%d."),
			*PressId,
			HiddenSourceComponents.Num(),
			GeneratedComponents.Num(),
			ActorsToDestroy.Num(),
			SkippedOtherPressActorCount,
			SkippedOtherPressComponentCount);
	}

	static void ApplyStep11BooleanOperations(
		UWorld* World,
		const TArray<FReconstructedMesh>& Meshes,
		const FString& PressId,
		const FString& DiagnosticJsonPath,
		TSet<FString>& OutAcceptedCutterActorNames)
	{
		if (!World)
		{
			return;
		}

		OutAcceptedCutterActorNames.Reset();
		TSharedRef<FJsonObject> DiagnosticsRoot = MakeShared<FJsonObject>();
		DiagnosticsRoot->SetNumberField(TEXT("diagnostic_version"), 2);
		DiagnosticsRoot->SetStringField(TEXT("press_id"), PressId);
		DiagnosticsRoot->SetStringField(TEXT("active_undo_press_id"), GActiveUndoPressId);
		DiagnosticsRoot->SetNumberField(TEXT("excavation_cutter_normal_scale"), ExcavationCutterNormalScale);
		DiagnosticsRoot->SetNumberField(TEXT("snap_tolerance_cm"), Step11BooleanSnapToleranceCm);
		DiagnosticsRoot->SetNumberField(TEXT("min_renderable_edge_cm"), Step11BooleanMinRenderableEdgeCm);

		TArray<TSharedPtr<FJsonValue>> InputMeshDiagnostics;
		TArray<TSharedPtr<FJsonValue>> CutterDiagnosticsJson;
		TArray<TSharedPtr<FJsonValue>> TargetDiagnosticsJson;
		TArray<TSharedPtr<FJsonValue>> BuildFailureDiagnostics;

		TArray<const FReconstructedMesh*> Cutters;
		TArray<UE::Geometry::FDynamicMesh3> CutterMeshes;
		TArray<FBox> CutterBounds;
		TArray<FStep11MeshDiagnostics> CutterDiagnostics;
		for (const FReconstructedMesh& Mesh : Meshes)
		{
			TSharedRef<FJsonObject> InputObject = MakeShared<FJsonObject>();
			InputObject->SetStringField(TEXT("actor_name"), Mesh.ActorName);
			InputObject->SetStringField(TEXT("tag"), Mesh.Tag.ToString());
			InputObject->SetBoolField(TEXT("is_excavate_cutter"), Mesh.bIsExcavateCutter);
			InputObject->SetNumberField(TEXT("input_world_vertex_count"), Mesh.VerticesWorld.Num());
			InputObject->SetNumberField(TEXT("input_triangle_index_count"), Mesh.Triangles.Num());
			InputObject->SetNumberField(TEXT("input_triangle_count"), Mesh.Triangles.Num() / 3);
			InputObject->SetArrayField(TEXT("normal"), JsonVector(Mesh.Normal)->AsArray());
			InputObject->SetObjectField(TEXT("input_bounds"), Step11BoxJson(BuildWorldBounds(Mesh.VerticesWorld)));
			InputMeshDiagnostics.Add(MakeShared<FJsonValueObject>(InputObject));

			if (!Mesh.bIsExcavateCutter || Mesh.VerticesWorld.Num() < 3 || Mesh.Triangles.Num() < 3)
			{
				UE_LOG(
					LogTemp,
					Log,
					TEXT("Step11Diag: reconstruction mesh skipped as cutter actor=%s excavate=%d vertices=%d triangle_indices=%d."),
					*Mesh.ActorName,
					Mesh.bIsExcavateCutter ? 1 : 0,
					Mesh.VerticesWorld.Num(),
					Mesh.Triangles.Num());
				continue;
			}

			UE::Geometry::FDynamicMesh3 CutterMesh;
			if (!BuildDynamicMeshFromWorldMesh(Mesh.VerticesWorld, Mesh.Triangles, CutterMesh))
			{
				UE_LOG(LogTemp, Warning, TEXT("Step11: failed to build cutter dynamic mesh for %s."), *Mesh.ActorName);
				TSharedRef<FJsonObject> FailureObject = MakeShared<FJsonObject>();
				FailureObject->SetStringField(TEXT("kind"), TEXT("cutter_build_failed"));
				FailureObject->SetStringField(TEXT("actor_name"), Mesh.ActorName);
				FailureObject->SetNumberField(TEXT("input_world_vertex_count"), Mesh.VerticesWorld.Num());
				FailureObject->SetNumberField(TEXT("input_triangle_index_count"), Mesh.Triangles.Num());
				BuildFailureDiagnostics.Add(MakeShared<FJsonValueObject>(FailureObject));
				continue;
			}

			Cutters.Add(&Mesh);
			CutterBounds.Add(BuildWorldBounds(Mesh.VerticesWorld));
			FStep11MeshDiagnostics Diagnostics = AnalyzeStep11DynamicMesh(CutterMesh, Mesh.ActorName, TEXT("excavate_cutter"));
			CutterDiagnostics.Add(Diagnostics);
			LogStep11MeshDiagnostics(Diagnostics);
			CutterDiagnosticsJson.Add(MakeShared<FJsonValueObject>(Step11MeshDiagnosticsJson(Diagnostics)));
			CutterMeshes.Add(MoveTemp(CutterMesh));
		}

		DiagnosticsRoot->SetArrayField(TEXT("input_reconstruction_meshes"), InputMeshDiagnostics);
		DiagnosticsRoot->SetArrayField(TEXT("cutters"), CutterDiagnosticsJson);
		DiagnosticsRoot->SetArrayField(TEXT("build_failures"), BuildFailureDiagnostics);

		if (Cutters.Num() == 0)
		{
			TSharedRef<FJsonObject> Summary = MakeShared<FJsonObject>();
			Summary->SetNumberField(TEXT("excavate_cutter_count"), 0);
			Summary->SetNumberField(TEXT("updated_static_component_count"), 0);
			Summary->SetNumberField(TEXT("updated_procedural_component_count"), 0);
			Summary->SetNumberField(TEXT("created_boolean_result_actor_count"), 0);
			Summary->SetNumberField(TEXT("failed_boolean_count"), 0);
			Summary->SetStringField(TEXT("status"), TEXT("no_excavate_cutters"));
			DiagnosticsRoot->SetObjectField(TEXT("summary"), Summary);
			DiagnosticsRoot->SetArrayField(TEXT("targets"), TargetDiagnosticsJson);
			if (!DiagnosticJsonPath.IsEmpty() && SaveJsonObject(DiagnosticsRoot, DiagnosticJsonPath))
			{
				UE_LOG(LogTemp, Log, TEXT("Step11Diag: saved diagnostics %s"), *DiagnosticJsonPath);
			}
			return;
		}

		TArray<UProceduralMeshComponent*> ProceduralTargetComponents;
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (!Actor || Actor->IsHidden() ||
				Actor->ActorHasTag(ReconstructedFaceTag) ||
				ActorIsStep11Cutter(Actor))
			{
				continue;
			}

			TArray<UProceduralMeshComponent*> ProceduralComponents;
			Actor->GetComponents<UProceduralMeshComponent>(ProceduralComponents);
			for (UProceduralMeshComponent* Component : ProceduralComponents)
			{
				if (!Component || !Component->IsRegistered() || !Component->IsVisible())
				{
					continue;
				}
				if (Component->ComponentTags.Contains(Step11BooleanResultTag) ||
					Actor->ActorHasTag(Step11ActionAttachTag))
				{
					ProceduralTargetComponents.Add(Component);
				}
			}
		}

		int32 UpdatedStaticComponentCount = 0;
		int32 UpdatedProceduralComponentCount = 0;
		int32 CreatedBooleanResultActorCount = 0;
		int32 FailedBooleanCount = 0;
		for (UProceduralMeshComponent* Component : ProceduralTargetComponents)
		{
			AActor* Actor = Component ? Component->GetOwner() : nullptr;
			if (!Actor || !Component->IsRegistered() || !Component->IsVisible())
			{
				continue;
			}

			UE::Geometry::FDynamicMesh3 CurrentMesh;
			if (!BuildDynamicMeshFromProceduralMeshComponent(Component, CurrentMesh))
			{
				TSharedRef<FJsonObject> TargetObject = MakeShared<FJsonObject>();
				TargetObject->SetStringField(TEXT("target_actor"), Actor->GetName());
				TargetObject->SetStringField(TEXT("target_component"), Component->GetName());
				TargetObject->SetStringField(TEXT("source_type"), Component->ComponentTags.Contains(Step11BooleanResultTag) ? TEXT("prior_boolean_result") : TEXT("attach_procedural"));
				TargetObject->SetStringField(TEXT("status"), TEXT("build_dynamic_mesh_failed"));
				TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
				UE_LOG(
					LogTemp,
					Warning,
					TEXT("Step11Diag: failed to build procedural target mesh actor=%s component=%s."),
					*Actor->GetName(),
					*Component->GetName());
				continue;
			}

			const FString SourceType = Component->ComponentTags.Contains(Step11BooleanResultTag) ? TEXT("prior_boolean_result") : TEXT("attach_procedural");
			FStep11MeshDiagnostics InitialDiagnostics = AnalyzeStep11DynamicMesh(
				CurrentMesh,
				FString::Printf(TEXT("%s/%s"), *Actor->GetName(), *Component->GetName()),
				SourceType);
			LogStep11MeshDiagnostics(InitialDiagnostics);
			TArray<TSharedPtr<FJsonValue>> OperationDiagnostics;
			TArray<int8> FinalTriangleSourceMeshById;
			TSet<FString> TargetAcceptedCutters;

			bool bEmptyResult = false;
			const bool bModified = ApplyCuttersToDynamicMesh(
				CurrentMesh,
				Actor->GetName(),
				Component->GetName(),
				SourceType,
				Cutters,
				CutterMeshes,
				CutterBounds,
				CutterDiagnostics,
				&OperationDiagnostics,
				FinalTriangleSourceMeshById,
				TargetAcceptedCutters,
				bEmptyResult,
				FailedBooleanCount);

			TSharedRef<FJsonObject> TargetObject = MakeShared<FJsonObject>();
			TargetObject->SetStringField(TEXT("target_actor"), Actor->GetName());
			TargetObject->SetStringField(TEXT("target_component"), Component->GetName());
			TargetObject->SetStringField(TEXT("source_type"), SourceType);
			TargetObject->SetObjectField(TEXT("initial"), Step11MeshDiagnosticsJson(InitialDiagnostics));
			TargetObject->SetBoolField(TEXT("modified"), bModified);
			TargetObject->SetBoolField(TEXT("empty_result"), bEmptyResult);
			TargetObject->SetArrayField(TEXT("operations"), OperationDiagnostics);
			if (!bModified)
			{
				TargetObject->SetStringField(TEXT("status"), TEXT("not_modified"));
				TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
				continue;
			}

			if (bEmptyResult)
			{
				HideStep11SourceComponentForPress(Component, PressId);
				TargetObject->SetStringField(TEXT("status"), TEXT("target_fully_removed_empty_result"));
				MergeStep11StringSet(OutAcceptedCutterActorNames, TargetAcceptedCutters);
				++UpdatedProceduralComponentCount;
			}
			else
			{
				AActor* ResultActor = CreateBooleanResultActor(
					World,
					Component,
					Actor->GetName(),
					Component->GetName(),
					PressId,
					CurrentMesh,
					FinalTriangleSourceMeshById);
				if (ResultActor)
				{
					HideStep11SourceComponentForPress(Component, PressId);
					MergeStep11StringSet(OutAcceptedCutterActorNames, TargetAcceptedCutters);
					TargetObject->SetStringField(TEXT("status"), TEXT("created_boolean_result_actor"));
					TargetObject->SetStringField(TEXT("result_actor"), ResultActor->GetName());
					TargetObject->SetObjectField(
						TEXT("final"),
						Step11MeshDiagnosticsJson(AnalyzeStep11DynamicMesh(CurrentMesh, FString::Printf(TEXT("%s/%s final"), *Actor->GetName(), *Component->GetName()), TEXT("procedural_boolean_result"))));
					++UpdatedProceduralComponentCount;
					++CreatedBooleanResultActorCount;
				}
				else
				{
					TargetObject->SetStringField(TEXT("status"), TEXT("failed_to_create_boolean_result_actor"));
				}
			}
			TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
		}

		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (!Actor || Actor->IsHidden() ||
				Actor->ActorHasTag(ReconstructedFaceTag) ||
				ActorHasAnyStep11RuntimeTag(Actor))
			{
				continue;
			}

			TArray<UStaticMeshComponent*> StaticComponents;
			Actor->GetComponents<UStaticMeshComponent>(StaticComponents);
			for (UStaticMeshComponent* Component : StaticComponents)
			{
				if (!Component || !Component->IsRegistered() || !Component->IsVisible() || !Component->GetStaticMesh())
				{
					continue;
				}

				UE::Geometry::FDynamicMesh3 CurrentMesh;
				if (!BuildDynamicMeshFromStaticMeshComponent(Component, CurrentMesh))
				{
					TSharedRef<FJsonObject> TargetObject = MakeShared<FJsonObject>();
					TargetObject->SetStringField(TEXT("target_actor"), Actor->GetName());
					TargetObject->SetStringField(TEXT("target_component"), Component->GetName());
					TargetObject->SetStringField(TEXT("source_type"), TEXT("static_mesh"));
					TargetObject->SetStringField(TEXT("status"), TEXT("build_dynamic_mesh_failed"));
					TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
					UE_LOG(
						LogTemp,
						Warning,
						TEXT("Step11Diag: failed to build static target mesh actor=%s component=%s."),
						*Actor->GetName(),
						*Component->GetName());
					continue;
				}

				FStep11MeshDiagnostics InitialDiagnostics = AnalyzeStep11DynamicMesh(
					CurrentMesh,
					FString::Printf(TEXT("%s/%s"), *Actor->GetName(), *Component->GetName()),
					TEXT("static_mesh"));
				LogStep11MeshDiagnostics(InitialDiagnostics);
				TArray<TSharedPtr<FJsonValue>> OperationDiagnostics;
				TArray<int8> FinalTriangleSourceMeshById;
				TSet<FString> TargetAcceptedCutters;

				bool bEmptyResult = false;
				const bool bModified = ApplyCuttersToDynamicMesh(
					CurrentMesh,
					Actor->GetName(),
					Component->GetName(),
					TEXT("static_mesh"),
					Cutters,
					CutterMeshes,
					CutterBounds,
					CutterDiagnostics,
					&OperationDiagnostics,
					FinalTriangleSourceMeshById,
					TargetAcceptedCutters,
					bEmptyResult,
					FailedBooleanCount);

				TSharedRef<FJsonObject> TargetObject = MakeShared<FJsonObject>();
				TargetObject->SetStringField(TEXT("target_actor"), Actor->GetName());
				TargetObject->SetStringField(TEXT("target_component"), Component->GetName());
				TargetObject->SetStringField(TEXT("source_type"), TEXT("static_mesh"));
				TargetObject->SetObjectField(TEXT("initial"), Step11MeshDiagnosticsJson(InitialDiagnostics));
				TargetObject->SetBoolField(TEXT("modified"), bModified);
				TargetObject->SetBoolField(TEXT("empty_result"), bEmptyResult);
				TargetObject->SetArrayField(TEXT("operations"), OperationDiagnostics);
				if (!bModified)
				{
					TargetObject->SetStringField(TEXT("status"), TEXT("not_modified"));
					TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
					continue;
				}

				if (bEmptyResult)
				{
					HideStep11SourceComponentForPress(Component, PressId);
					MergeStep11StringSet(OutAcceptedCutterActorNames, TargetAcceptedCutters);
					TargetObject->SetStringField(TEXT("status"), TEXT("target_fully_removed_empty_result"));
					++UpdatedStaticComponentCount;
				}
				else
				{
					AActor* ResultActor = CreateBooleanResultActor(
						World,
						Component,
						Actor->GetName(),
						Component->GetName(),
						PressId,
						CurrentMesh,
						FinalTriangleSourceMeshById);
					if (ResultActor)
					{
						HideStep11SourceComponentForPress(Component, PressId);
						MergeStep11StringSet(OutAcceptedCutterActorNames, TargetAcceptedCutters);
						TargetObject->SetStringField(TEXT("status"), TEXT("created_boolean_result_actor"));
						TargetObject->SetStringField(TEXT("result_actor"), ResultActor->GetName());
						TargetObject->SetObjectField(
							TEXT("final"),
							Step11MeshDiagnosticsJson(AnalyzeStep11DynamicMesh(CurrentMesh, FString::Printf(TEXT("%s/%s final"), *Actor->GetName(), *Component->GetName()), TEXT("static_mesh_boolean_result"))));
						++UpdatedStaticComponentCount;
						++CreatedBooleanResultActorCount;
					}
					else
					{
						TargetObject->SetStringField(TEXT("status"), TEXT("failed_to_create_boolean_result_actor"));
					}
				}
				TargetDiagnosticsJson.Add(MakeShared<FJsonValueObject>(TargetObject));
			}
		}

		UE_LOG(
			LogTemp,
			Log,
			TEXT("Step11: press=%s applied %d excavate cutter(s), updated %d static mesh component(s), updated %d procedural component(s), created %d boolean result actor(s), failed boolean attempts=%d."),
			*PressId,
			Cutters.Num(),
			UpdatedStaticComponentCount,
			UpdatedProceduralComponentCount,
			CreatedBooleanResultActorCount,
			FailedBooleanCount);

		TSharedRef<FJsonObject> Summary = MakeShared<FJsonObject>();
		Summary->SetNumberField(TEXT("excavate_cutter_count"), Cutters.Num());
		Summary->SetNumberField(TEXT("updated_static_component_count"), UpdatedStaticComponentCount);
		Summary->SetNumberField(TEXT("updated_procedural_component_count"), UpdatedProceduralComponentCount);
		Summary->SetNumberField(TEXT("created_boolean_result_actor_count"), CreatedBooleanResultActorCount);
		Summary->SetNumberField(TEXT("failed_boolean_count"), FailedBooleanCount);
		Summary->SetNumberField(TEXT("target_record_count"), TargetDiagnosticsJson.Num());
		Summary->SetNumberField(TEXT("accepted_cutter_actor_count"), OutAcceptedCutterActorNames.Num());
		Summary->SetStringField(TEXT("status"), FailedBooleanCount > 0 ? TEXT("completed_with_boolean_failures") : TEXT("completed"));
		DiagnosticsRoot->SetObjectField(TEXT("summary"), Summary);
		DiagnosticsRoot->SetArrayField(TEXT("targets"), TargetDiagnosticsJson);
		if (!DiagnosticJsonPath.IsEmpty())
		{
			if (SaveJsonObject(DiagnosticsRoot, DiagnosticJsonPath))
			{
				UE_LOG(LogTemp, Log, TEXT("Step11Diag: saved diagnostics %s"), *DiagnosticJsonPath);
			}
			else
			{
				UE_LOG(LogTemp, Warning, TEXT("Step11Diag: failed to save diagnostics %s"), *DiagnosticJsonPath);
			}
		}
	}

	static FVector UnrealWorldToObjDebugSpace(const FVector& WorldPosition)
	{
		// UE is left-handed Z-up (X forward, Y right, Z up). OBJ viewers normally
		// interpret vertices in a right-handed space; mirror Y for debug export only.
		return FVector(WorldPosition.X, -WorldPosition.Y, WorldPosition.Z);
	}

	static int32 AppendObjWorldMesh(
		FString& Obj,
		int32& VertexOffset,
		const FString& ObjectName,
		const TCHAR* MaterialName,
		const TArray<FVector>& VerticesWorld,
		const TArray<int32>& Triangles)
	{
		if (VerticesWorld.Num() < 3 || Triangles.Num() < 3)
		{
			return 0;
		}

		const int32 BaseVertex = VertexOffset;
		Obj += FString::Printf(TEXT("\no %s\nusemtl %s\n"), *ObjSafeName(ObjectName), MaterialName);
		for (const FVector& V : VerticesWorld)
		{
			const FVector ObjPosition = UnrealWorldToObjDebugSpace(V);
			Obj += FString::Printf(TEXT("v %.6f %.6f %.6f\n"), ObjPosition.X, ObjPosition.Y, ObjPosition.Z);
		}

		int32 FaceCount = 0;
		for (int32 i = 0; i + 2 < Triangles.Num(); i += 3)
		{
			const int32 A = Triangles[i];
			const int32 B = Triangles[i + 1];
			const int32 C = Triangles[i + 2];
			if (!VerticesWorld.IsValidIndex(A) || !VerticesWorld.IsValidIndex(B) || !VerticesWorld.IsValidIndex(C))
			{
				continue;
			}
			Obj += FString::Printf(TEXT("f %d %d %d\n"), BaseVertex + A + 1, BaseVertex + C + 1, BaseVertex + B + 1);
			++FaceCount;
		}
		VertexOffset += VerticesWorld.Num();
		return FaceCount;
	}

	static int32 AppendStaticMeshComponentObj(FString& Obj, int32& VertexOffset, UStaticMeshComponent* Component)
	{
		if (!Component || !Component->IsRegistered() || !Component->IsVisible())
		{
			return 0;
		}

		UStaticMesh* StaticMesh = Component->GetStaticMesh();
		if (!StaticMesh || !StaticMesh->GetRenderData() || StaticMesh->GetRenderData()->LODResources.Num() == 0)
		{
			return 0;
		}

		const FStaticMeshLODResources& LOD = StaticMesh->GetRenderData()->LODResources[0];
		const FPositionVertexBuffer& PositionBuffer = LOD.VertexBuffers.PositionVertexBuffer;
		const int32 NumVertices = PositionBuffer.GetNumVertices();
		if (NumVertices <= 0 || LOD.Sections.Num() == 0)
		{
			return 0;
		}

		const FIndexArrayView Indices = LOD.IndexBuffer.GetArrayView();
		if (Indices.Num() < 3)
		{
			return 0;
		}

		const int32 BaseVertex = VertexOffset;
		const FString OwnerName = Component->GetOwner() ? Component->GetOwner()->GetName() : TEXT("StaticActor");
		Obj += FString::Printf(
			TEXT("\no %s_%s\nusemtl scene_gray\n"),
			*ObjSafeName(OwnerName),
			*ObjSafeName(Component->GetName()));

		const FTransform& ComponentTransform = Component->GetComponentTransform();
		for (int32 VertexIndex = 0; VertexIndex < NumVertices; ++VertexIndex)
		{
			const FVector3f LocalPosition = PositionBuffer.VertexPosition(VertexIndex);
			const FVector WorldPosition = ComponentTransform.TransformPosition(
				FVector(double(LocalPosition.X), double(LocalPosition.Y), double(LocalPosition.Z)));
			const FVector ObjPosition = UnrealWorldToObjDebugSpace(WorldPosition);
			Obj += FString::Printf(TEXT("v %.6f %.6f %.6f\n"), ObjPosition.X, ObjPosition.Y, ObjPosition.Z);
		}

		int32 FaceCount = 0;
		for (const FStaticMeshSection& Section : LOD.Sections)
		{
			for (uint32 TriIndex = 0; TriIndex < Section.NumTriangles; ++TriIndex)
			{
				const uint32 IndexBase = Section.FirstIndex + TriIndex * 3;
				if (IndexBase + 2 >= uint32(Indices.Num()))
				{
					continue;
				}

				const int32 A = int32(Indices[IndexBase]);
				const int32 B = int32(Indices[IndexBase + 1]);
				const int32 C = int32(Indices[IndexBase + 2]);
				if (A < 0 || B < 0 || C < 0 || A >= NumVertices || B >= NumVertices || C >= NumVertices)
				{
					continue;
				}

				Obj += FString::Printf(TEXT("f %d %d %d\n"), BaseVertex + A + 1, BaseVertex + C + 1, BaseVertex + B + 1);
				++FaceCount;
			}
		}

		VertexOffset += NumVertices;
		return FaceCount;
	}

	static int32 AppendProceduralMeshComponentObj(FString& Obj, int32& VertexOffset, UProceduralMeshComponent* Component)
	{
		if (!Component || !Component->IsRegistered() || !Component->IsVisible() || !Component->ComponentTags.Contains(Step11BooleanResultTag))
		{
			return 0;
		}

		const FString OwnerName = Component->GetOwner() ? Component->GetOwner()->GetName() : TEXT("BooleanActor");
		const FTransform& ComponentTransform = Component->GetComponentTransform();
		int32 FaceCount = 0;
		for (int32 SectionIndex = 0; SectionIndex < Component->GetNumSections(); ++SectionIndex)
		{
			if (!Component->IsMeshSectionVisible(SectionIndex))
			{
				continue;
			}

			const FProcMeshSection* Section = Component->GetProcMeshSection(SectionIndex);
			if (!Section || Section->ProcVertexBuffer.Num() < 3 || Section->ProcIndexBuffer.Num() < 3)
			{
				continue;
			}

			const int32 BaseVertex = VertexOffset;
			Obj += FString::Printf(
				TEXT("\no %s_%s_section_%d\nusemtl %s\n"),
				*ObjSafeName(OwnerName),
				*ObjSafeName(Component->GetName()),
				SectionIndex,
				SectionIndex == 0 ? TEXT("scene_gray") : TEXT("boolean_cap_gray"));

			for (const FProcMeshVertex& Vertex : Section->ProcVertexBuffer)
			{
				const FVector WorldPosition = ComponentTransform.TransformPosition(Vertex.Position);
				const FVector ObjPosition = UnrealWorldToObjDebugSpace(WorldPosition);
				Obj += FString::Printf(TEXT("v %.6f %.6f %.6f\n"), ObjPosition.X, ObjPosition.Y, ObjPosition.Z);
			}

			for (int32 i = 0; i + 2 < Section->ProcIndexBuffer.Num(); i += 3)
			{
				const int32 A = Section->ProcIndexBuffer[i];
				const int32 B = Section->ProcIndexBuffer[i + 1];
				const int32 C = Section->ProcIndexBuffer[i + 2];
				if (!Section->ProcVertexBuffer.IsValidIndex(A) ||
					!Section->ProcVertexBuffer.IsValidIndex(B) ||
					!Section->ProcVertexBuffer.IsValidIndex(C))
				{
					continue;
				}

				Obj += FString::Printf(TEXT("f %d %d %d\n"), BaseVertex + A + 1, BaseVertex + C + 1, BaseVertex + B + 1);
				++FaceCount;
			}

			VertexOffset += Section->ProcVertexBuffer.Num();
		}
		return FaceCount;
	}

	static bool ExportReconstructionSceneObj(
		UWorld* World,
		const TArray<FReconstructedMesh>& ReconstructedMeshes,
		const FString& ObjPath)
	{
		if (!World || ObjPath.IsEmpty())
		{
			return false;
		}

		const FString ObjDir = FPaths::GetPath(ObjPath);
		if (!ObjDir.IsEmpty())
		{
			IFileManager::Get().MakeDirectory(*ObjDir, true);
		}

		const FString MtlFilename = FPaths::GetBaseFilename(ObjPath) + TEXT(".mtl");
		const FString MtlPath = ObjDir / MtlFilename;
		const FString Mtl =
			TEXT("# FromLZ Step 10 reconstruction debug materials\n")
			TEXT("newmtl scene_gray\n")
			TEXT("Ka 0.650000 0.650000 0.650000\n")
			TEXT("Kd 0.650000 0.650000 0.650000\n")
			TEXT("Ks 0.000000 0.000000 0.000000\n")
			TEXT("d 1.000000\n\n")
			TEXT("newmtl boolean_cap_gray\n")
			TEXT("Ka 0.700000 0.700000 0.700000\n")
			TEXT("Kd 0.700000 0.700000 0.700000\n")
			TEXT("Ks 0.000000 0.000000 0.000000\n")
			TEXT("d 1.000000\n\n")
			TEXT("newmtl reconstructed_blue\n")
			TEXT("Ka 0.000000 0.250000 1.000000\n")
			TEXT("Kd 0.000000 0.470000 1.000000\n")
			TEXT("Ke 0.000000 0.050000 0.250000\n")
			TEXT("Ks 0.000000 0.000000 0.000000\n")
			TEXT("d 1.000000\n\n")
			TEXT("newmtl reconstructed_cutter_transparent\n")
			TEXT("Ka 0.000000 0.250000 1.000000\n")
			TEXT("Kd 0.000000 0.470000 1.000000\n")
			TEXT("Ke 0.000000 0.050000 0.250000\n")
			TEXT("Ks 0.000000 0.000000 0.000000\n")
			TEXT("d 0.250000\n");

		FString Obj;
		Obj.Reserve(1024 * 1024);
		Obj += TEXT("# FromLZ Step 10 reconstruction scene debug export\n");
		Obj += TEXT("# Coordinates are converted from Unreal left-handed Z-up to OBJ right-handed Z-up.\n");
		Obj += TEXT("# Mapping: obj_x = ue_x, obj_y = -ue_y, obj_z = ue_z. Units are centimeters.\n");
		Obj += FString::Printf(TEXT("mtllib %s\n"), *MtlFilename);

		int32 VertexOffset = 0;
		int32 StaticComponentCount = 0;
		int32 StaticFaceCount = 0;
		int32 BooleanComponentCount = 0;
		int32 BooleanFaceCount = 0;
		for (TActorIterator<AActor> It(World); It; ++It)
		{
			AActor* Actor = *It;
			if (!Actor || Actor->IsHidden() ||
				Actor->ActorHasTag(ReconstructedFaceTag) ||
				Actor->ActorHasTag(ReconstructedSolidTag))
			{
				continue;
			}

			TArray<UStaticMeshComponent*> Components;
			Actor->GetComponents<UStaticMeshComponent>(Components);
			for (UStaticMeshComponent* Component : Components)
			{
				const int32 FaceCount = AppendStaticMeshComponentObj(Obj, VertexOffset, Component);
				if (FaceCount > 0)
				{
					++StaticComponentCount;
					StaticFaceCount += FaceCount;
				}
			}

			TArray<UProceduralMeshComponent*> ProceduralComponents;
			Actor->GetComponents<UProceduralMeshComponent>(ProceduralComponents);
			for (UProceduralMeshComponent* Component : ProceduralComponents)
			{
				const int32 FaceCount = AppendProceduralMeshComponentObj(Obj, VertexOffset, Component);
				if (FaceCount > 0)
				{
					++BooleanComponentCount;
					BooleanFaceCount += FaceCount;
				}
			}
		}

		int32 ReconstructionFaceCount = 0;
		for (const FReconstructedMesh& MeshData : ReconstructedMeshes)
		{
			ReconstructionFaceCount += AppendObjWorldMesh(
				Obj,
				VertexOffset,
				MeshData.ActorName,
				MeshData.bIsExcavateCutter ? TEXT("reconstructed_cutter_transparent") : TEXT("reconstructed_blue"),
				MeshData.VerticesWorld,
				MeshData.Triangles);
		}

		const bool bSavedMtl = FFileHelper::SaveStringToFile(Mtl, *MtlPath);
		const bool bSavedObj = FFileHelper::SaveStringToFile(Obj, *ObjPath);
		if (bSavedMtl && bSavedObj)
		{
			UE_LOG(
				LogTemp,
				Log,
				TEXT("FaceReconstruct: exported debug OBJ %s (static components=%d, static faces=%d, boolean components=%d, boolean faces=%d, reconstruction faces=%d)."),
				*ObjPath,
				StaticComponentCount,
				StaticFaceCount,
				BooleanComponentCount,
				BooleanFaceCount,
				ReconstructionFaceCount);
			return true;
		}

		UE_LOG(LogTemp, Warning, TEXT("FaceReconstruct: failed to export debug OBJ %s or MTL %s."), *ObjPath, *MtlPath);
		return false;
	}

	static void SpawnMeshesOnGameThread(
		TWeakObjectPtr<UWorld> WorldPtr,
		TArray<FReconstructedMesh> Meshes,
		FString DebugObjPath = FString(),
		FString PressId = FString())
	{
		AsyncTask(ENamedThreads::GameThread, [WorldPtr, Meshes = MoveTemp(Meshes), DebugObjPath = MoveTemp(DebugObjPath), PressId = MoveTemp(PressId)]() mutable
		{
			UWorld* World = WorldPtr.Get();
			if (!World)
			{
				UE_LOG(LogTemp, Warning, TEXT("FaceReconstruct: world is no longer valid; skipped runtime mesh spawn."));
				return;
			}

			GActiveUndoPressId = PressId;
			const FName PressTag = Step11PressTag(PressId);
			TSet<FString> AcceptedCutterActorNames;
			const FString Step11DiagnosticJsonPath = DebugObjPath.IsEmpty()
				? FString()
				: FPaths::GetPath(DebugObjPath) / TEXT("11_boolean_diagnostics.json");
			ApplyStep11BooleanOperations(World, Meshes, PressId, Step11DiagnosticJsonPath, AcceptedCutterActorNames);

			UMaterialInterface* VertexColorMaterial = GetReconstructionVertexColorMaterial();
			for (const FReconstructedMesh& MeshData : Meshes)
			{
				if (MeshData.VerticesWorld.Num() < 3 || MeshData.Triangles.Num() < 3)
				{
					continue;
				}

				FVector Origin = FVector::ZeroVector;
				for (const FVector& V : MeshData.VerticesWorld)
				{
					Origin += V;
				}
				Origin /= double(MeshData.VerticesWorld.Num());

				FActorSpawnParameters Params;
				Params.Name = FName(*MeshData.ActorName);
				AActor* Actor = World->SpawnActor<AActor>(AActor::StaticClass(), FTransform(Origin), Params);
				if (!Actor)
				{
					UE_LOG(LogTemp, Warning, TEXT("FaceReconstruct: failed to spawn actor %s"), *MeshData.ActorName);
					continue;
				}

				Actor->Tags.AddUnique(MeshData.Tag);
				Actor->Tags.AddUnique(PressTag);
				Actor->Tags.AddUnique(MeshData.bIsExcavateCutter ? Step11ActionExcavateCutterTag : Step11ActionAttachTag);
#if WITH_EDITOR
				Actor->SetActorLabel(MeshData.ActorName);
#endif

				UProceduralMeshComponent* MeshComponent = NewObject<UProceduralMeshComponent>(Actor, TEXT("ReconstructedFaceMesh"));
				MeshComponent->SetMobility(EComponentMobility::Movable);
				MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
				MeshComponent->bUseAsyncCooking = false;
				MeshComponent->SetCastShadow(false);
				MeshComponent->ComponentTags.AddUnique(PressTag);
				MeshComponent->ComponentTags.AddUnique(MeshData.bIsExcavateCutter ? Step11ActionExcavateCutterTag : Step11ActionAttachTag);
				Actor->SetRootComponent(MeshComponent);
				Actor->AddInstanceComponent(MeshComponent);
				MeshComponent->RegisterComponent();
				if (!Actor->SetActorLocation(Origin, false, nullptr, ETeleportType::TeleportPhysics))
				{
					MeshComponent->SetWorldLocation(Origin);
				}
				MeshComponent->SetMaterial(0, MeshData.bIsExcavateCutter ? CreateCutterMaterial(MeshComponent) : VertexColorMaterial);

				TArray<FVector> LocalVertices;
				LocalVertices.Reserve(MeshData.VerticesWorld.Num());
				for (const FVector& V : MeshData.VerticesWorld)
				{
					LocalVertices.Add(V - Origin);
				}

				TArray<FVector> Normals;
				Normals.Init(MeshData.Normal.GetSafeNormal(), LocalVertices.Num());

				TArray<FVector2D> UV0;
				UV0.Init(FVector2D::ZeroVector, LocalVertices.Num());

				TArray<FColor> Colors;
				Colors.Init(MeshData.Color, LocalVertices.Num());

				TArray<FProcMeshTangent> Tangents;
				TArray<int32> DrawTriangles;
				AppendDoubleSidedTriangles(MeshData.Triangles, DrawTriangles);
				MeshComponent->CreateMeshSection(0, LocalVertices, DrawTriangles, Normals, UV0, Colors, Tangents, false);

				if (MeshData.bIsExcavateCutter && AcceptedCutterActorNames.Contains(MeshData.ActorName))
				{
					Actor->SetActorHiddenInGame(true);
					MeshComponent->SetVisibility(false, true);
					MeshComponent->SetHiddenInGame(true, true);
					UE_LOG(LogTemp, Log, TEXT("Step11Diag: cutter hidden after accepted boolean actor=%s press=%s."), *MeshData.ActorName, *PressId);
				}
			}

			if (!DebugObjPath.IsEmpty())
			{
				ExportReconstructionSceneObj(World, Meshes, DebugObjPath);
			}

			UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: press=%s spawned %d runtime reconstruction actor(s)."), *PressId, Meshes.Num());
		});
	}

	static void SaveCommonFailureForComponents(
		const TArray<FString>& ComponentNames, const FString& PressDir, const FString& Error)
	{
		for (const FString& ComponentName : ComponentNames)
		{
			MakeFailureResult(ComponentName, Error, PressDir / ComponentName);

			FSolidReconstructionResult Solid;
			Solid.ComponentName = ComponentName;
			Solid.ActorName = FString::Printf(TEXT("FromLZ_ReconstructedSolid_%s_%s"), *FPaths::GetCleanFilename(PressDir), *ComponentName);
			SaveSkippedSolidResult(
				Solid,
				PressDir / ComponentName / TEXT("10_solid_reconstruction.json"),
				FString::Printf(TEXT("Solid skipped because Step 10 common inputs failed: %s"), *Error));
		}
	}
}

void FFromLZFaceReconstructor::ProcessPress(const FString& PressDir, const FString& ActionPressDir, TWeakObjectPtr<UWorld> World)
{
	const FString PressId = FPaths::GetCleanFilename(PressDir);
	TArray<FString> ComponentNames;
	IFileManager::Get().IterateDirectory(*PressDir, [&ComponentNames](const TCHAR* InPath, bool bIsDir) -> bool
	{
		if (bIsDir)
		{
			const FString Name = FPaths::GetCleanFilename(FString(InPath));
			if (Name.StartsWith(TEXT("Component_")))
			{
				ComponentNames.Add(Name);
			}
		}
		return true;
	});
	ComponentNames.Sort();

	if (ComponentNames.Num() == 0)
	{
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>(), FString(), PressId);
		UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: no component folders found in %s"), *PressDir);
		return;
	}

	FCommonInputs Inputs;
	if (!LoadCaptureRef(PressDir, Inputs))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, TEXT("Failed to read capture_ref.json or resolve capture/faces paths"));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>(), FString(), PressId);
		return;
	}
	if (!DecodePngToRGBA(Inputs.FacesPngPath, Inputs.FacesRGBA, Inputs.FacesWidth, Inputs.FacesHeight))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to decode faces png: %s"), *Inputs.FacesPngPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>(), FString(), PressId);
		return;
	}
	if (!LoadFacesJson(Inputs.FacesJsonPath, Inputs.Faces))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to read faces json: %s"), *Inputs.FacesJsonPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>(), FString(), PressId);
		return;
	}
	if (!LoadCameraJson(Inputs.CaptureJsonPath, Inputs.Camera))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to read capture camera json: %s"), *Inputs.CaptureJsonPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>(), FString(), PressId);
		return;
	}
	BuildFaceLookups(Inputs);

	FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));

	TArray<FComponentResult> Results;
	Results.SetNum(ComponentNames.Num());
	ParallelFor(ComponentNames.Num(), [&](int32 Index)
	{
		Results[Index] = ProcessComponent(ComponentNames[Index], PressDir, ActionPressDir, Inputs);
	});

	TArray<FReconstructedMesh> MeshesToSpawn;
	for (const FComponentResult& Result : Results)
	{
		if (!Result.bSuccess)
		{
			continue;
		}

		if (Result.Solid.bSuccess)
		{
			FReconstructedMesh SolidMesh;
			SolidMesh.ActorName = Result.Solid.ActorName;
			SolidMesh.Tag = ReconstructedSolidTag;
			SolidMesh.VerticesWorld = Result.Solid.MeshVerticesWorld;
			SolidMesh.Triangles = Result.Solid.MeshTriangles;
			SolidMesh.Normal = Result.Solid.MeshNormal;
			SolidMesh.PressId = PressId;
			SolidMesh.bIsExcavateCutter = Result.Solid.Action.Equals(TEXT("excavate"), ESearchCase::IgnoreCase);
			if (SolidMesh.bIsExcavateCutter)
			{
				ScaleVerticesAlongAxis(SolidMesh.VerticesWorld, SolidMesh.Normal, ExcavationCutterNormalScale);
			}
			SolidMesh.Color = SolidMesh.bIsExcavateCutter ? FColor(0, 120, 255, 80) : ReconstructedDebugBlue;
			MeshesToSpawn.Add(MoveTemp(SolidMesh));
		}
	}

	SpawnMeshesOnGameThread(World, MoveTemp(MeshesToSpawn), PressDir / TEXT("10_reconstruction_scene.obj"), PressId);
	UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: processed %d component(s) for %s."), ComponentNames.Num(), *PressDir);
}

void FFromLZFaceReconstructor::RestoreStep11RuntimeBooleans(TWeakObjectPtr<UWorld> World)
{
	AsyncTask(ENamedThreads::GameThread, [World]()
	{
		UWorld* WorldPtr = World.Get();
		if (!WorldPtr)
		{
			UE_LOG(LogTemp, Warning, TEXT("Step11: world is no longer valid; skipped restore."));
			return;
		}

		RestoreStep11BooleanResults(WorldPtr, GActiveUndoPressId);
	});
}
