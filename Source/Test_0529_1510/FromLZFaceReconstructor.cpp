#include "FromLZFaceReconstructor.h"

#include "Async/Async.h"
#include "Async/ParallelFor.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "GameFramework/Actor.h"
#include "HAL/FileManager.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Materials/Material.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Modules/ModuleManager.h"
#include "ProceduralMeshComponent.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"

namespace
{
	const FName ReconstructedFaceTag(TEXT("FromLZ_ReconstructedFace"));
	constexpr double MinOverlapRatio = 0.05;
	constexpr double NormalParallelThresholdDegrees = 10.0;
	constexpr double MinProjectedNormalPixels = 1.0;

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
		TArray<FVector> VerticesWorld;
		TArray<int32> Triangles;
		FVector Normal = FVector::UpVector;
		FColor Color = FColor(80, 220, 120, 255);
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
		bool bSuccess = false;
		TArray<FFaceCandidate> Candidates;
		FVector SelectedPlaneHit = FVector::ZeroVector;
		TArray<FVector> MeshVerticesWorld;
		TArray<int32> MeshTriangles;
		FVector MeshNormal = FVector::UpVector;
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
		Root->TryGetStringField(TEXT("capture_json"), OutInputs.CaptureJsonRel);
		Root->TryGetStringField(TEXT("faces_png"), OutInputs.FacesPngRel);
		Root->TryGetStringField(TEXT("faces_json"), OutInputs.FacesJsonRel);

		if (OutInputs.FacesPngRel.IsEmpty() && !CaptureStem.IsEmpty())
		{
			OutInputs.FacesPngRel = TEXT("FromLZCaptures/") + CaptureStem + TEXT("_faces.png");
		}
		if (OutInputs.FacesJsonRel.IsEmpty() && !CaptureStem.IsEmpty())
		{
			OutInputs.FacesJsonRel = TEXT("FromLZCaptures/") + CaptureStem + TEXT("_faces.json");
		}

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
		const FVector RayOrigin = Camera.bOrthographic ? CameraOrthoRayOrigin(Camera, Width, Height, Pixel) : Camera.Location;
		const FVector RayDir = Camera.bOrthographic ? Camera.Forward : CameraRayDirection(Camera, Width, Height, Pixel);
		const double Denom = FVector::DotProduct(RayDir, Face.Normal);
		if (FMath::Abs(Denom) < 1e-8)
		{
			return false;
		}

		const double T = FVector::DotProduct(Face.PlanePoint - RayOrigin, Face.Normal) / Denom;
		OutHit = RayOrigin + RayDir * T;
		OutDistance = FVector::Distance(Camera.Location, OutHit);
		return true;
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
		FVector2D P0;
		if (!ProjectWorldToImage(Camera, Width, Height, AnchorWorld, P0))
		{
			return false;
		}

		const double ProbeLength = FMath::Clamp(FaceWorldExtent(Face) * 0.25, 10.0, 250.0);
		const FVector Normal = Face.Normal.GetSafeNormal();
		for (int32 SignIndex = 0; SignIndex < 2; ++SignIndex)
		{
			const double Sign = (SignIndex == 0) ? 1.0 : -1.0;
			FVector2D P1;
			if (!ProjectWorldToImage(Camera, Width, Height, AnchorWorld + Normal * (ProbeLength * Sign), P1))
			{
				continue;
			}

			const FVector2D Delta = P1 - P0;
			if (Delta.Size() >= MinProjectedNormalPixels)
			{
				OutDirection = Delta.GetSafeNormal();
				return true;
			}
		}
		return false;
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

		const FString OutputJson = ComponentDir / TEXT("10_face_reconstruction.json");
		const FString ActionPath = ActionPressDir / ComponentName / TEXT("Action.json");
		if (!LoadAction(ActionPath, Result.Action))
		{
			Result.Error = FString::Printf(TEXT("Failed to read action from %s"), *ActionPath);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		if (Result.Action == TEXT("excavate"))
		{
			Result.PolygonKey = TEXT("cap_polygon");
		}
		else if (Result.Action == TEXT("attach"))
		{
			Result.PolygonKey = TEXT("cap_polygon_translated");
		}
		else
		{
			Result.Error = FString::Printf(TEXT("Unsupported action '%s'"), *Result.Action);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		TSharedPtr<FJsonObject> CapJson;
		const FString CapJsonPath = ComponentDir / TEXT("09_cap_extrusion.json");
		if (!LoadJsonObject(CapJsonPath, CapJson))
		{
			Result.Error = FString::Printf(TEXT("Failed to read %s"), *CapJsonPath);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		TArray<FVector2D> CapPoly;
		if (!ParseVector2DArray(CapJson, *Result.PolygonKey, CapPoly) || CapPoly.Num() < 3)
		{
			Result.Error = FString::Printf(TEXT("Missing or invalid polygon '%s' in %s"), *Result.PolygonKey, *CapJsonPath);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		FVector2D SideVector = FVector2D::ZeroVector;
		if (!ParseVector2DField(CapJson, TEXT("side_vector"), SideVector))
		{
			Result.Error = FString::Printf(TEXT("Missing or invalid side_vector in %s"), *CapJsonPath);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		TArray<uint8> CapRGBA;
		const FString CapPngPath = ComponentDir / TEXT("09_cap_extrusion.png");
		if (!DecodePngToRGBA(CapPngPath, CapRGBA, Result.CapWidth, Result.CapHeight))
		{
			Result.Error = FString::Printf(TEXT("Failed to read cap image size from %s"), *CapPngPath);
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		const double ScaleX = double(Inputs.FacesWidth) / double(Result.CapWidth);
		const double ScaleY = double(Inputs.FacesHeight) / double(Result.CapHeight);
		const FVector2D ScaledSideVector(SideVector.X * ScaleX, SideVector.Y * ScaleY);
		if (ScaledSideVector.SizeSquared() < 1e-8)
		{
			Result.Error = TEXT("side_vector is too short after mapping to faces image space");
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}
		Result.GreenLineVector2D = ScaledSideVector.GetSafeNormal();

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
			SaveComponentResultJson(Result, OutputJson);
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
					const double Dot = FMath::Clamp(
						FMath::Abs(FVector2D::DotProduct(Candidate.ProjectedNormal2D.GetSafeNormal(), Result.GreenLineVector2D)),
						0.0, 1.0);
					Candidate.NormalGreenAngleDegrees = FMath::RadiansToDegrees(FMath::Acos(Dot));
					Candidate.bNormalParallelPass = Candidate.NormalGreenAngleDegrees <= NormalParallelThresholdDegrees;
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
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		const int32* SelectedIndex = Inputs.FaceIndexById.Find(Result.SelectedFaceId);
		if (!SelectedIndex)
		{
			Result.Error = TEXT("Selected face id was not found in face table");
			SaveComponentResultJson(Result, OutputJson);
			return Result;
		}

		const FFaceInfo& SelectedFace = Inputs.Faces[*SelectedIndex];
		Result.MeshVerticesWorld = SelectedFace.KeyPoints3D;
		if (!TriangulatePolygon2D(SelectedFace.KeyPoints2D, Result.MeshTriangles))
		{
			Result.Error = TEXT("Failed to triangulate selected face");
			SaveComponentResultJson(Result, OutputJson);
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
		return Result;
	}

	static void SpawnMeshesOnGameThread(TWeakObjectPtr<UWorld> WorldPtr, TArray<FReconstructedMesh> Meshes)
	{
		AsyncTask(ENamedThreads::GameThread, [WorldPtr, Meshes = MoveTemp(Meshes)]() mutable
		{
			UWorld* World = WorldPtr.Get();
			if (!World)
			{
				UE_LOG(LogTemp, Warning, TEXT("FaceReconstruct: world is no longer valid; skipped runtime mesh spawn."));
				return;
			}

			TArray<AActor*> Existing;
			for (TActorIterator<AActor> It(World); It; ++It)
			{
				AActor* Actor = *It;
				if (Actor && Actor->ActorHasTag(ReconstructedFaceTag))
				{
					Existing.Add(Actor);
				}
			}
			for (AActor* Actor : Existing)
			{
				if (Actor)
				{
					Actor->Destroy();
				}
			}

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

				Actor->Tags.AddUnique(ReconstructedFaceTag);
#if WITH_EDITOR
				Actor->SetActorLabel(MeshData.ActorName);
#endif

				UProceduralMeshComponent* MeshComponent = NewObject<UProceduralMeshComponent>(Actor, TEXT("ReconstructedFaceMesh"));
				MeshComponent->SetMobility(EComponentMobility::Movable);
				MeshComponent->SetCollisionEnabled(ECollisionEnabled::NoCollision);
				MeshComponent->bUseAsyncCooking = false;
				Actor->SetRootComponent(MeshComponent);
				Actor->AddInstanceComponent(MeshComponent);
				MeshComponent->RegisterComponent();
				MeshComponent->SetMaterial(0, UMaterial::GetDefaultMaterial(MD_Surface));

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
				MeshComponent->CreateMeshSection(0, LocalVertices, MeshData.Triangles, Normals, UV0, Colors, Tangents, false);
			}

			UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: spawned %d runtime face actor(s)."), Meshes.Num());
		});
	}

	static void SaveCommonFailureForComponents(
		const TArray<FString>& ComponentNames, const FString& PressDir, const FString& Error)
	{
		for (const FString& ComponentName : ComponentNames)
		{
			MakeFailureResult(ComponentName, Error, PressDir / ComponentName);
		}
	}
}

void FFromLZFaceReconstructor::ProcessPress(const FString& PressDir, const FString& ActionPressDir, TWeakObjectPtr<UWorld> World)
{
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
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>());
		UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: no component folders found in %s"), *PressDir);
		return;
	}

	FCommonInputs Inputs;
	if (!LoadCaptureRef(PressDir, Inputs))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, TEXT("Failed to read capture_ref.json or resolve capture/faces paths"));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>());
		return;
	}
	if (!DecodePngToRGBA(Inputs.FacesPngPath, Inputs.FacesRGBA, Inputs.FacesWidth, Inputs.FacesHeight))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to decode faces png: %s"), *Inputs.FacesPngPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>());
		return;
	}
	if (!LoadFacesJson(Inputs.FacesJsonPath, Inputs.Faces))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to read faces json: %s"), *Inputs.FacesJsonPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>());
		return;
	}
	if (!LoadCameraJson(Inputs.CaptureJsonPath, Inputs.Camera))
	{
		SaveCommonFailureForComponents(ComponentNames, PressDir, FString::Printf(TEXT("Failed to read capture camera json: %s"), *Inputs.CaptureJsonPath));
		SpawnMeshesOnGameThread(World, TArray<FReconstructedMesh>());
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

		FReconstructedMesh Mesh;
		Mesh.ActorName = Result.ActorName;
		Mesh.VerticesWorld = Result.MeshVerticesWorld;
		Mesh.Triangles = Result.MeshTriangles;
		Mesh.Normal = Result.MeshNormal;
		MeshesToSpawn.Add(MoveTemp(Mesh));
	}

	SpawnMeshesOnGameThread(World, MoveTemp(MeshesToSpawn));
	UE_LOG(LogTemp, Log, TEXT("FaceReconstruct: processed %d component(s) for %s."), ComponentNames.Num(), *PressDir);
}
