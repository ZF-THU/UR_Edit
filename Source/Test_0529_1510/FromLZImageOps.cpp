#include "FromLZImageOps.h"

#include "Algo/Reverse.h"
#include "Async/ParallelFor.h"
#include "HAL/FileManager.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Misc/FileHelper.h"
#include "Modules/ModuleManager.h"

namespace FromLZImageOps
{
	void BinarizeNonWhite(const TArray<uint8>& RGBA, int32 Width, int32 Height, uint8 WhiteThreshold, TArray<uint8>& OutBin)
	{
		const int32 N = Width * Height;
		OutBin.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			const int32 Off = i * 4;
			const uint8 R = RGBA[Off + 0];
			const uint8 G = RGBA[Off + 1];
			const uint8 B = RGBA[Off + 2];
			const bool bWhite = R > WhiteThreshold && G > WhiteThreshold && B > WhiteThreshold;
			OutBin[i] = bWhite ? 0 : 255;
		}
	}

	void Dilate(const TArray<uint8>& In, int32 Width, int32 Height, int32 OffMinX, int32 OffMaxX, int32 OffMinY, int32 OffMaxY, TArray<uint8>& Out)
	{
		Out.SetNumUninitialized(Width * Height);
		for (int32 y = 0; y < Height; ++y)
		{
			for (int32 x = 0; x < Width; ++x)
			{
				uint8 v = 0;
				for (int32 oy = OffMinY; oy <= OffMaxY && v == 0; ++oy)
				{
					const int32 yy = y + oy;
					if (yy < 0 || yy >= Height)
					{
						continue;
					}
					for (int32 ox = OffMinX; ox <= OffMaxX; ++ox)
					{
						const int32 xx = x + ox;
						if (xx < 0 || xx >= Width)
						{
							continue;
						}
						if (In[yy * Width + xx] > 0)
						{
							v = 255;
							break;
						}
					}
				}
				Out[y * Width + x] = v;
			}
		}
	}

	void Erode(const TArray<uint8>& In, int32 Width, int32 Height, int32 OffMinX, int32 OffMaxX, int32 OffMinY, int32 OffMaxY, TArray<uint8>& Out)
	{
		Out.SetNumUninitialized(Width * Height);
		for (int32 y = 0; y < Height; ++y)
		{
			for (int32 x = 0; x < Width; ++x)
			{
				uint8 v = 255;
				for (int32 oy = OffMinY; oy <= OffMaxY && v == 255; ++oy)
				{
					const int32 yy = y + oy;
					for (int32 ox = OffMinX; ox <= OffMaxX; ++ox)
					{
						const int32 xx = x + ox;
						const bool bInside = xx >= 0 && xx < Width && yy >= 0 && yy < Height;
						if (!bInside || In[yy * Width + xx] == 0)
						{
							v = 0;
							break;
						}
					}
				}
				Out[y * Width + x] = v;
			}
		}
	}

	void MorphClose(TArray<uint8>& InOut, int32 Width, int32 Height, int32 Kernel, int32 Iterations)
	{
		if (Kernel <= 1 || Iterations <= 0)
		{
			return;
		}
		const int32 r = Kernel / 2;
		TArray<uint8> Tmp;
		for (int32 it = 0; it < Iterations; ++it)
		{
			Dilate(InOut, Width, Height, -r, r, -r, r, Tmp);
			Erode(Tmp, Width, Height, -r, r, -r, r, InOut);
		}
	}

	void Dilate2x2(TArray<uint8>& InOut, int32 Width, int32 Height, int32 Iterations)
	{
		if (Iterations <= 0)
		{
			return;
		}
		TArray<uint8> Tmp;
		for (int32 it = 0; it < Iterations; ++it)
		{
			// 2x2 structuring element with anchor at top-left: offsets {0,1} in x and y.
			Dilate(InOut, Width, Height, 0, 1, 0, 1, Tmp);
			InOut = Tmp;
		}
	}

	void RemoveSmallComponents(TArray<uint8>& InOut, int32 Width, int32 Height, int32 MinArea)
	{
		if (MinArea <= 1)
		{
			return;
		}
		const int32 N = Width * Height;
		TArray<int32> Label;
		Label.Init(-1, N);

		TArray<int32> Stack;
		Stack.Reserve(1024);

		static const int32 DX[8] = { 1, -1, 0, 0, 1, 1, -1, -1 };
		static const int32 DY[8] = { 0, 0, 1, -1, 1, -1, 1, -1 };

		TArray<int32> CompPixels;
		CompPixels.Reserve(1024);

		for (int32 start = 0; start < N; ++start)
		{
			if (InOut[start] == 0 || Label[start] != -1)
			{
				continue;
			}

			CompPixels.Reset();
			Stack.Reset();
			Stack.Push(start);
			Label[start] = start;

			while (Stack.Num() > 0)
			{
				const int32 p = Stack.Pop(EAllowShrinking::No);
				CompPixels.Push(p);
				const int32 px = p % Width;
				const int32 py = p / Width;
				for (int32 k = 0; k < 8; ++k)
				{
					const int32 nx = px + DX[k];
					const int32 ny = py + DY[k];
					if (nx < 0 || nx >= Width || ny < 0 || ny >= Height)
					{
						continue;
					}
					const int32 np = ny * Width + nx;
					if (InOut[np] > 0 && Label[np] == -1)
					{
						Label[np] = start;
						Stack.Push(np);
					}
				}
			}

			if (CompPixels.Num() < MinArea)
			{
				for (int32 p : CompPixels)
				{
					InOut[p] = 0;
				}
			}
		}
	}

	void ZhangSuenThinning(const TArray<uint8>& In, int32 Width, int32 Height, TArray<uint8>& OutSkel, int32 MaxIter)
	{
		const int32 N = Width * Height;
		TArray<uint8> Img;
		Img.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			Img[i] = In[i] > 0 ? 1 : 0;
		}

		auto At = [&](int32 x, int32 y) -> uint8
		{
			if (x < 0 || x >= Width || y < 0 || y >= Height)
			{
				return 0;
			}
			return Img[y * Width + x];
		};

		TArray<uint8> Marker;
		Marker.SetNumUninitialized(N);

		for (int32 iter = 0; iter < MaxIter; ++iter)
		{
			bool bChanged = false;

			for (int32 step = 0; step < 2; ++step)
			{
				FMemory::Memzero(Marker.GetData(), Marker.Num());

				for (int32 y = 0; y < Height; ++y)
				{
					for (int32 x = 0; x < Width; ++x)
					{
						if (Img[y * Width + x] != 1)
						{
							continue;
						}

						const uint8 P2 = At(x, y - 1);
						const uint8 P3 = At(x + 1, y - 1);
						const uint8 P4 = At(x + 1, y);
						const uint8 P5 = At(x + 1, y + 1);
						const uint8 P6 = At(x, y + 1);
						const uint8 P7 = At(x - 1, y + 1);
						const uint8 P8 = At(x - 1, y);
						const uint8 P9 = At(x - 1, y - 1);

						const int32 B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9;
						if (B < 2 || B > 6)
						{
							continue;
						}

						int32 A = 0;
						A += (P2 == 0 && P3 == 1) ? 1 : 0;
						A += (P3 == 0 && P4 == 1) ? 1 : 0;
						A += (P4 == 0 && P5 == 1) ? 1 : 0;
						A += (P5 == 0 && P6 == 1) ? 1 : 0;
						A += (P6 == 0 && P7 == 1) ? 1 : 0;
						A += (P7 == 0 && P8 == 1) ? 1 : 0;
						A += (P8 == 0 && P9 == 1) ? 1 : 0;
						A += (P9 == 0 && P2 == 1) ? 1 : 0;
						if (A != 1)
						{
							continue;
						}

						if (step == 0)
						{
							if ((P2 * P4 * P6) == 0 && (P4 * P6 * P8) == 0)
							{
								Marker[y * Width + x] = 1;
							}
						}
						else
						{
							if ((P2 * P4 * P8) == 0 && (P2 * P6 * P8) == 0)
							{
								Marker[y * Width + x] = 1;
							}
						}
					}
				}

				for (int32 i = 0; i < N; ++i)
				{
					if (Marker[i])
					{
						Img[i] = 0;
						bChanged = true;
					}
				}
			}

			if (!bChanged)
			{
				break;
			}
		}

		OutSkel.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			OutSkel[i] = Img[i] ? 255 : 0;
		}
	}

	// ====================================================================
	// Step 3 helpers: skeleton graph topology (ported from the Python
	// get_neighbors / crossing_number / skeleton_node_type primitives).
	// All operate on a foreground=255 mask laid out row-major.
	// ====================================================================
	namespace SkelGraph
	{
		FORCEINLINE bool Fg(const uint8* M, int32 W, int32 H, int32 x, int32 y)
		{
			return x >= 0 && x < W && y >= 0 && y < H && M[y * W + x] > 0;
		}

		// "Safe" 8-neighbors: 4-neighbors always; a diagonal only when neither of
		// its two orthogonal sides is foreground (avoids corner-cutting shortcuts).
		// Fills OutN (capacity 8) and returns the count.
		int32 SafeNeighbors(const uint8* M, int32 W, int32 H, int32 x, int32 y, int32 OutN[8])
		{
			int32 Count = 0;
			static const int32 OD[4][2] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };
			for (int32 k = 0; k < 4; ++k)
			{
				const int32 xx = x + OD[k][0];
				const int32 yy = y + OD[k][1];
				if (Fg(M, W, H, xx, yy))
				{
					OutN[Count++] = yy * W + xx;
				}
			}
			static const int32 DD[4][2] = { {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
			for (int32 k = 0; k < 4; ++k)
			{
				const int32 dx = DD[k][0];
				const int32 dy = DD[k][1];
				const int32 xx = x + dx;
				const int32 yy = y + dy;
				if (!Fg(M, W, H, xx, yy))
				{
					continue;
				}
				const bool bSide1 = Fg(M, W, H, x + dx, y);
				const bool bSide2 = Fg(M, W, H, x, y + dy);
				if (!bSide1 && !bSide2)
				{
					OutN[Count++] = yy * W + xx;
				}
			}
			return Count;
		}

		// Crossing number over the raw 8-neighborhood (P2..P9 = N,NE,E,SE,S,SW,W,NW).
		int32 CrossingNumber(const uint8* M, int32 W, int32 H, int32 x, int32 y)
		{
			const int32 V[8] = {
				Fg(M, W, H, x,     y - 1) ? 1 : 0,
				Fg(M, W, H, x + 1, y - 1) ? 1 : 0,
				Fg(M, W, H, x + 1, y)     ? 1 : 0,
				Fg(M, W, H, x + 1, y + 1) ? 1 : 0,
				Fg(M, W, H, x,     y + 1) ? 1 : 0,
				Fg(M, W, H, x - 1, y + 1) ? 1 : 0,
				Fg(M, W, H, x - 1, y)     ? 1 : 0,
				Fg(M, W, H, x - 1, y - 1) ? 1 : 0,
			};
			int32 Transitions = 0;
			for (int32 i = 0; i < 8; ++i)
			{
				if (V[i] != V[(i + 1) % 8])
				{
					++Transitions;
				}
			}
			return Transitions / 2;
		}

		enum class ENode : uint8 { Isolated, Endpoint, Branch, Regular };

		ENode NodeType(const uint8* M, int32 W, int32 H, int32 x, int32 y)
		{
			int32 Tmp[8];
			const int32 Deg = SafeNeighbors(M, W, H, x, y, Tmp);
			if (Deg == 0)
			{
				return ENode::Isolated;
			}
			if (Deg == 1)
			{
				return ENode::Endpoint;
			}
			if (CrossingNumber(M, W, H, x, y) >= 3)
			{
				return ENode::Branch;
			}
			return ENode::Regular;
		}

		void FindEndpoints(const uint8* M, int32 W, int32 H, TArray<FIntPoint>& Out)
		{
			Out.Reset();
			for (int32 y = 0; y < H; ++y)
			{
				for (int32 x = 0; x < W; ++x)
				{
					if (M[y * W + x] > 0 && NodeType(M, W, H, x, y) == ENode::Endpoint)
					{
						Out.Emplace(x, y);
					}
				}
			}
		}

		// Bresenham 8-connected line points (matches cv2.LINE_8), inclusive of both ends.
		void LinePoints(FIntPoint P0, FIntPoint P1, TArray<FIntPoint>& Out)
		{
			Out.Reset();
			int32 x0 = P0.X, y0 = P0.Y;
			const int32 x1 = P1.X, y1 = P1.Y;
			const int32 dx = FMath::Abs(x1 - x0);
			const int32 dy = -FMath::Abs(y1 - y0);
			const int32 sx = x0 < x1 ? 1 : -1;
			const int32 sy = y0 < y1 ? 1 : -1;
			int32 err = dx + dy;
			for (;;)
			{
				Out.Emplace(x0, y0);
				if (x0 == x1 && y0 == y1)
				{
					break;
				}
				const int32 e2 = 2 * err;
				if (e2 >= dy)
				{
					err += dy;
					x0 += sx;
				}
				if (e2 <= dx)
				{
					err += dx;
					y0 += sy;
				}
			}
		}

		// Shortest pixel path (BFS over safe neighbors) from Start to Goal; both must be
		// foreground. Returns true and fills OutPath (Start..Goal inclusive) when reachable.
		bool ShortestPath(const uint8* M, int32 W, int32 H, FIntPoint Start, FIntPoint Goal, TArray<FIntPoint>& OutPath)
		{
			OutPath.Reset();
			if (!Fg(M, W, H, Start.X, Start.Y) || !Fg(M, W, H, Goal.X, Goal.Y))
			{
				return false;
			}
			const int32 N = W * H;
			const int32 StartIdx = Start.Y * W + Start.X;
			const int32 GoalIdx = Goal.Y * W + Goal.X;

			TArray<int32> Parent;
			Parent.Init(-2, N); // -2 = unvisited, -1 = root
			TArray<int32> Queue;
			Queue.Reserve(256);
			Queue.Add(StartIdx);
			Parent[StartIdx] = -1;

			int32 Head = 0;
			bool bFound = (StartIdx == GoalIdx);
			while (Head < Queue.Num() && !bFound)
			{
				const int32 Cur = Queue[Head++];
				const int32 cx = Cur % W;
				const int32 cy = Cur / W;
				int32 Nb[8];
				const int32 Deg = SafeNeighbors(M, W, H, cx, cy, Nb);
				for (int32 i = 0; i < Deg; ++i)
				{
					const int32 NI = Nb[i];
					if (Parent[NI] != -2)
					{
						continue;
					}
					Parent[NI] = Cur;
					if (NI == GoalIdx)
					{
						bFound = true;
						break;
					}
					Queue.Add(NI);
				}
			}

			if (!bFound)
			{
				return false;
			}

			for (int32 Cur = GoalIdx; Cur != -1; Cur = Parent[Cur])
			{
				OutPath.Emplace(Cur % W, Cur / W);
			}
			Algo::Reverse(OutPath);
			return true;
		}

		void BranchStopPoints(const uint8* M, int32 W, int32 H, TSet<int32>& Out)
		{
			Out.Reset();
			for (int32 y = 0; y < H; ++y)
			{
				for (int32 x = 0; x < W; ++x)
				{
					if (M[y * W + x] > 0 && NodeType(M, W, H, x, y) == ENode::Branch)
					{
						Out.Add(y * W + x);
					}
				}
			}
		}

		// Pick the candidate that best preserves direction (max dot of unit vectors).
		int32 ChooseBestContinuation(FIntPoint Prev, FIntPoint Cur, const int32* Cands, int32 NumCands, int32 W)
		{
			if (NumCands == 0)
			{
				return -1;
			}
			if (NumCands == 1)
			{
				return Cands[0];
			}
			FVector2D PrevV(Cur.X - Prev.X, Cur.Y - Prev.Y);
			if (PrevV.Size() < 1e-8)
			{
				return Cands[0];
			}
			PrevV.Normalize();
			int32 Best = -1;
			double BestScore = -1e9;
			for (int32 i = 0; i < NumCands; ++i)
			{
				const int32 qx = Cands[i] % W;
				const int32 qy = Cands[i] / W;
				FVector2D NextV(qx - Cur.X, qy - Cur.Y);
				if (NextV.Size() < 1e-8)
				{
					continue;
				}
				NextV.Normalize();
				const double Score = FVector2D::DotProduct(PrevV, NextV);
				if (Score > BestScore)
				{
					BestScore = Score;
					Best = Cands[i];
				}
			}
			return Best;
		}

		// Trace one dangling path from an endpoint until the first branch/endpoint
		// (or a stop node). Returns coords including Start and the terminal node.
		void TraceBranchPath(const uint8* M, int32 W, int32 H, FIntPoint Start, const TSet<int32>& StopNodes, TArray<FIntPoint>& OutPath)
		{
			OutPath.Reset();
			OutPath.Add(Start);
			FIntPoint Prev(-1, -1);
			bool bHasPrev = false;
			FIntPoint Cur = Start;
			int32 Safety = 0;
			for (;;)
			{
				if (++Safety > 20000)
				{
					break;
				}
				int32 Nb[8];
				const int32 Deg = SafeNeighbors(M, W, H, Cur.X, Cur.Y, Nb);
				int32 Cands[8];
				int32 NumCands = 0;
				const int32 PrevIdx = bHasPrev ? (Prev.Y * W + Prev.X) : -1;
				for (int32 i = 0; i < Deg; ++i)
				{
					if (Nb[i] != PrevIdx)
					{
						Cands[NumCands++] = Nb[i];
					}
				}
				if (NumCands == 0)
				{
					break;
				}
				if (OutPath.Num() > 1)
				{
					const ENode T = NodeType(M, W, H, Cur.X, Cur.Y);
					if (T == ENode::Endpoint || T == ENode::Branch)
					{
						break;
					}
				}
				const FIntPoint Ref = bHasPrev ? Prev : Cur;
				const int32 NextIdx = ChooseBestContinuation(Ref, Cur, Cands, NumCands, W);
				if (NextIdx < 0)
				{
					break;
				}
				Prev = Cur;
				bHasPrev = true;
				Cur = FIntPoint(NextIdx % W, NextIdx / W);
				OutPath.Add(Cur);
				if (StopNodes.Contains(NextIdx))
				{
					break;
				}
				const ENode T = NodeType(M, W, H, Cur.X, Cur.Y);
				if (T == ENode::Endpoint || T == ENode::Branch)
				{
					break;
				}
			}
		}
	} // namespace SkelGraph

	void CleanupSkeletonEndpoints(
		const TArray<uint8>& Skel, int32 Width, int32 Height,
		float GapTol, int32 ConnectThickness,
		float SmallLoopBboxAreaThresh, float BranchPruneMaxPixels,
		const TArray<uint8>& SourceColorMap, int32 SourceColorSampleRadius,
		const FString& RedBlackConnectorsDebugPngPath, const FString& RedBlackReconnectedDebugPngPath,
		const FString& ConnectorPruneDebugJsonPath,
		TArray<uint8>& OutConnected, TArray<uint8>& OutSmallLoopPruned, TArray<uint8>& OutCleaned)
	{
		const int32 N = Width * Height;
		const bool bHasSourceColorMap = SourceColorMap.Num() >= N;
		const int32 SampleRadius = FMath::Max(0, SourceColorSampleRadius);

		// work = binarized copy (foreground = 255).
		TArray<uint8> Work;
		Work.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			Work[i] = Skel[i] > 0 ? 255 : 0;
		}

		TArray<FIntPoint> EndpointsBefore;
		SkelGraph::FindEndpoints(Work.GetData(), Width, Height, EndpointsBefore);

		// Nothing to connect.
		if (GapTol <= 0.0f || EndpointsBefore.Num() < 2)
		{
			OutConnected = Work;
			OutSmallLoopPruned = Work;
			OutCleaned = Work;
			return;
		}

		// ---- 1. mutual-nearest endpoint pairs within GapTol --------------------
		const int32 E = EndpointsBefore.Num();
		const double Tol2 = double(GapTol) * double(GapTol);
		TArray<int32> NearestJ;   // best partner index for endpoint i, or -1
		TArray<double> NearestD2;
		NearestJ.Init(-1, E);
		NearestD2.Init(TNumericLimits<double>::Max(), E);
		for (int32 i = 0; i < E; ++i)
		{
			for (int32 j = 0; j < E; ++j)
			{
				if (i == j)
				{
					continue;
				}
				const double ddx = double(EndpointsBefore[i].X - EndpointsBefore[j].X);
				const double ddy = double(EndpointsBefore[i].Y - EndpointsBefore[j].Y);
				const double D2 = ddx * ddx + ddy * ddy;
				if (D2 > Tol2)
				{
					continue;
				}
				if (D2 < NearestD2[i])
				{
					NearestD2[i] = D2;
					NearestJ[i] = j;
				}
			}
		}

		struct FConn
		{
			FString Source = TEXT("all_skeleton");
			FIntPoint P0;
			FIntPoint P1;
			TArray<FIntPoint> LinePts;
			TArray<FIntPoint> AddedPts;
			bool bRbAltPath = false;
			int32 RbAltPathPixels = 0;
			int64 RbBboxArea = -1;
			bool bProtectedKeep = false;
			bool bFullSkippedByProtection = false;
			bool bFullAltPath = false;
			int32 FullAltPathPixels = 0;
			int64 FullBboxArea = -1;
			int32 FullAddedPixels = 0;
			bool bKept = true;
			FString Decision = TEXT("not_processed");
		};
		TArray<FConn> Connections;
		TArray<bool> Used;
		Used.Init(false, E);
		for (int32 i = 0; i < E; ++i)
		{
			const int32 j = NearestJ[i];
			if (j < 0 || Used[i] || Used[j])
			{
				continue;
			}
			if (NearestJ[j] != i) // require mutual nearest
			{
				continue;
			}
			FConn Conn;
			Conn.P0 = EndpointsBefore[i];
			Conn.P1 = EndpointsBefore[j];
			Connections.Add(MoveTemp(Conn));
			Used[i] = true;
			Used[j] = true;
		}

		// connected = work + drawn 1px connection lines.
		TArray<uint8> Connected = Work;
		for (FConn& C : Connections)
		{
			SkelGraph::LinePoints(C.P0, C.P1, C.LinePts);
			for (const FIntPoint& P : C.LinePts)
			{
				if (P.X >= 0 && P.X < Width && P.Y >= 0 && P.Y < Height)
				{
					const int32 Idx = P.Y * Width + P.X;
					if (Work[Idx] == 0)
					{
						C.AddedPts.Add(P);
					}
					Connected[Idx] = 255;
				}
			}
		}
		OutConnected = Connected;

		const float LoopThresh = FMath::Max(0.0f, SmallLoopBboxAreaThresh);

		auto SampleSourceColorClass = [&](const FIntPoint& P) -> EStrokeColor
		{
			if (!bHasSourceColorMap)
			{
				return EStrokeColor::None;
			}

			int32 Counts[5] = { 0, 0, 0, 0, 0 };
			const int32 MinY = FMath::Max(0, P.Y - SampleRadius);
			const int32 MaxY = FMath::Min(Height - 1, P.Y + SampleRadius);
			const int32 MinX = FMath::Max(0, P.X - SampleRadius);
			const int32 MaxX = FMath::Min(Width - 1, P.X + SampleRadius);
			for (int32 yy = MinY; yy <= MaxY; ++yy)
			{
				for (int32 xx = MinX; xx <= MaxX; ++xx)
				{
					const uint8 C = SourceColorMap[yy * Width + xx];
					if (C > uint8(EStrokeColor::None) && C <= uint8(EStrokeColor::Blue))
					{
						++Counts[C];
					}
				}
			}

			int32 Best = int32(EStrokeColor::None);
			int32 BestCount = 0;
			for (int32 c = 1; c <= 4; ++c)
			{
				if (Counts[c] > BestCount)
				{
					BestCount = Counts[c];
					Best = c;
				}
			}
			return EStrokeColor(Best);
		};

		auto LoopBboxArea = [](const TArray<FIntPoint>& Path, const TArray<FIntPoint>& AddedPts) -> int64
		{
			if (Path.Num() == 0 && AddedPts.Num() == 0)
			{
				return -1;
			}
			int32 MinX = MAX_int32, MinY = MAX_int32, MaxX = MIN_int32, MaxY = MIN_int32;
			auto Acc = [&](const FIntPoint& P)
			{
				MinX = FMath::Min(MinX, P.X);
				MinY = FMath::Min(MinY, P.Y);
				MaxX = FMath::Max(MaxX, P.X);
				MaxY = FMath::Max(MaxY, P.Y);
			};
			for (const FIntPoint& P : Path) { Acc(P); }
			for (const FIntPoint& P : AddedPts) { Acc(P); }
			return int64(MaxX - MinX + 1) * int64(MaxY - MinY + 1);
		};

		TArray<uint8> RedBlackCandidate;
		RedBlackCandidate.Init(0, N);
		TArray<uint8> RedBlackDebugKind; // 0 background, 1 source black, 2 source/connector red
		RedBlackDebugKind.Init(0, N);
		if (bHasSourceColorMap)
		{
			for (int32 y = 0; y < Height; ++y)
			{
				for (int32 x = 0; x < Width; ++x)
				{
					const int32 Idx = y * Width + x;
					if (Work[Idx] == 0)
					{
						continue;
					}
					const EStrokeColor C = SampleSourceColorClass(FIntPoint(x, y));
					if (C == EStrokeColor::Red || C == EStrokeColor::Black)
					{
						RedBlackCandidate[Idx] = 255;
						RedBlackDebugKind[Idx] = (C == EStrokeColor::Black) ? 1 : 2;
					}
				}
			}
		}
		for (const FConn& C : Connections)
		{
			for (const FIntPoint& P : C.AddedPts)
			{
				if (P.X >= 0 && P.X < Width && P.Y >= 0 && P.Y < Height)
				{
					const int32 Idx = P.Y * Width + P.X;
					RedBlackCandidate[Idx] = 255;
					RedBlackDebugKind[Idx] = 2;
				}
			}
		}

		if (!RedBlackConnectorsDebugPngPath.IsEmpty())
		{
			TArray<uint8> RGBA;
			RGBA.Init(255, N * 4);
			for (int32 i = 0; i < N; ++i)
			{
				const int32 Off = i * 4;
				if (RedBlackDebugKind[i] == 1)
				{
					RGBA[Off + 0] = 0;
					RGBA[Off + 1] = 0;
					RGBA[Off + 2] = 0;
				}
				else if (RedBlackDebugKind[i] == 2)
				{
					RGBA[Off + 0] = 220;
					RGBA[Off + 1] = 20;
					RGBA[Off + 2] = 60;
				}
				RGBA[Off + 3] = 255;
			}
			IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
			TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
			if (IW.IsValid())
			{
				IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
				const TArray64<uint8>& C = IW->GetCompressed();
				FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *RedBlackConnectorsDebugPngPath);
			}
		}

		// Re-run the endpoint repair on the red/black candidate graph. These
		// second-pass connectors are promoted into the same connector list, so they
		// participate in the red/black protection check and the full-skeleton prune.
		TArray<FIntPoint> RbEndpoints;
		SkelGraph::FindEndpoints(RedBlackCandidate.GetData(), Width, Height, RbEndpoints);
		const int32 RbE = RbEndpoints.Num();
		TArray<int32> RbNearestJ;
		TArray<double> RbNearestD2;
		RbNearestJ.Init(-1, RbE);
		RbNearestD2.Init(TNumericLimits<double>::Max(), RbE);
		for (int32 i = 0; i < RbE; ++i)
		{
			for (int32 j = 0; j < RbE; ++j)
			{
				if (i == j)
				{
					continue;
				}
				const double ddx = double(RbEndpoints[i].X - RbEndpoints[j].X);
				const double ddy = double(RbEndpoints[i].Y - RbEndpoints[j].Y);
				const double D2 = ddx * ddx + ddy * ddy;
				if (D2 > Tol2)
				{
					continue;
				}
				if (D2 < RbNearestD2[i])
				{
					RbNearestD2[i] = D2;
					RbNearestJ[i] = j;
				}
			}
		}

		TArray<bool> RbUsed;
		RbUsed.Init(false, RbE);
		for (int32 i = 0; i < RbE; ++i)
		{
			const int32 j = RbNearestJ[i];
			if (j < 0 || RbUsed[i] || RbUsed[j])
			{
				continue;
			}
			if (RbNearestJ[j] != i)
			{
				continue;
			}

			FConn Conn;
			Conn.Source = TEXT("red_black_reconnect");
			Conn.P0 = RbEndpoints[i];
			Conn.P1 = RbEndpoints[j];
			SkelGraph::LinePoints(Conn.P0, Conn.P1, Conn.LinePts);

			bool bAddsRedBlackPixel = false;
			for (const FIntPoint& P : Conn.LinePts)
			{
				if (P.X < 0 || P.X >= Width || P.Y < 0 || P.Y >= Height)
				{
					continue;
				}
				const int32 Idx = P.Y * Width + P.X;
				if (RedBlackCandidate[Idx] == 0)
				{
					bAddsRedBlackPixel = true;
				}
				if (Connected[Idx] == 0)
				{
					Conn.AddedPts.Add(P);
				}
			}

			if (!bAddsRedBlackPixel && Conn.AddedPts.Num() == 0)
			{
				RbUsed[i] = true;
				RbUsed[j] = true;
				continue;
			}

			for (const FIntPoint& P : Conn.LinePts)
			{
				if (P.X < 0 || P.X >= Width || P.Y < 0 || P.Y >= Height)
				{
					continue;
				}
				const int32 Idx = P.Y * Width + P.X;
				RedBlackCandidate[Idx] = 255;
				RedBlackDebugKind[Idx] = 2;
				Connected[Idx] = 255;
			}
			Connections.Add(MoveTemp(Conn));
			RbUsed[i] = true;
			RbUsed[j] = true;
		}

		if (!RedBlackReconnectedDebugPngPath.IsEmpty())
		{
			TArray<uint8> RGBA;
			RGBA.Init(255, N * 4);
			for (int32 i = 0; i < N; ++i)
			{
				const int32 Off = i * 4;
				if (RedBlackDebugKind[i] == 1)
				{
					RGBA[Off + 0] = 0;
					RGBA[Off + 1] = 0;
					RGBA[Off + 2] = 0;
				}
				else if (RedBlackDebugKind[i] == 2)
				{
					RGBA[Off + 0] = 220;
					RGBA[Off + 1] = 20;
					RGBA[Off + 2] = 60;
				}
				RGBA[Off + 3] = 255;
			}
			IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
			TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
			if (IW.IsValid())
			{
				IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
				const TArray64<uint8>& C = IW->GetCompressed();
				FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *RedBlackReconnectedDebugPngPath);
			}
		}

		TArray<uint8> ProtectedConnectorPixels;
		ProtectedConnectorPixels.Init(0, N);
		if (LoopThresh > 0.0f)
		{
			for (FConn& C : Connections)
			{
				if (C.AddedPts.Num() == 0)
				{
					C.Decision = TEXT("skip_no_added_pixels");
					continue;
				}

				TArray<uint8> Probe = RedBlackCandidate;
				for (const FIntPoint& P : C.AddedPts)
				{
					Probe[P.Y * Width + P.X] = 0;
				}

				TArray<FIntPoint> RbPath;
				C.bRbAltPath = SkelGraph::ShortestPath(Probe.GetData(), Width, Height, C.P0, C.P1, RbPath);
				if (!C.bRbAltPath)
				{
					continue;
				}

				C.RbAltPathPixels = RbPath.Num();
				C.RbBboxArea = LoopBboxArea(RbPath, C.AddedPts);
				if (double(C.RbBboxArea) >= double(LoopThresh))
				{
					C.bProtectedKeep = true;
					for (const FIntPoint& P : C.AddedPts)
					{
						ProtectedConnectorPixels[P.Y * Width + P.X] = 255;
					}
				}
			}
		}

		// ---- 2. prune connections that only close a small loop -----------------
		TArray<uint8> Current = Connected;
		if (LoopThresh > 0.0f)
		{
			for (FConn& C : Connections)
			{
				if (C.bProtectedKeep)
				{
					C.bFullSkippedByProtection = true;
					C.bKept = true;
					C.Decision = TEXT("keep_red_black_large_loop_protected");
					continue;
				}

				// Pixels added by this connection (background in original Work) that are
				// currently still foreground. Pixels owned by a protected connector are
				// never removed through another connector's full-skeleton prune.
				TArray<FIntPoint> AddedPts;
				for (const FIntPoint& P : C.LinePts)
				{
					if (P.X < 0 || P.X >= Width || P.Y < 0 || P.Y >= Height)
					{
						continue;
					}
					const int32 Idx = P.Y * Width + P.X;
					if (Work[Idx] == 0 && Current[Idx] > 0 && ProtectedConnectorPixels[Idx] == 0)
					{
						AddedPts.Add(P);
					}
				}
				C.FullAddedPixels = AddedPts.Num();
				if (AddedPts.Num() == 0)
				{
					C.bKept = true;
					C.Decision = C.AddedPts.Num() == 0 ? TEXT("skip_no_added_pixels") : TEXT("keep_no_unprotected_added_pixels");
					continue;
				}

				// Temporarily remove the added pixels and check for an alternate path.
				for (const FIntPoint& P : AddedPts)
				{
					Current[P.Y * Width + P.X] = 0;
				}
				TArray<FIntPoint> Path;
				C.bFullAltPath = SkelGraph::ShortestPath(Current.GetData(), Width, Height, C.P0, C.P1, Path);
				if (!C.bFullAltPath)
				{
					// No alternate path => not a loop; restore the connection.
					for (const FIntPoint& P : AddedPts)
					{
						Current[P.Y * Width + P.X] = 255;
					}
					C.bKept = true;
					C.Decision = TEXT("keep_no_full_alt_path");
					continue;
				}

				C.FullAltPathPixels = Path.Num();
				C.FullBboxArea = LoopBboxArea(Path, AddedPts);

				if (double(C.FullBboxArea) < double(LoopThresh))
				{
					// Small loop: keep the added pixels removed.
					C.bKept = false;
					C.Decision = TEXT("delete_full_small_loop");
				}
				else
				{
					// Large loop: restore the connection.
					for (const FIntPoint& P : AddedPts)
					{
						Current[P.Y * Width + P.X] = 255;
					}
					C.bKept = true;
					C.Decision = TEXT("keep_full_large_loop");
				}
			}
		}
		OutSmallLoopPruned = Current;

		if (!ConnectorPruneDebugJsonPath.IsEmpty())
		{
			FString Json;
			Json += TEXT("{\n");
			Json += FString::Printf(TEXT("  \"gap_tol\": %.3f,\n"), GapTol);
			Json += FString::Printf(TEXT("  \"small_loop_bbox_area_thresh\": %.3f,\n"), SmallLoopBboxAreaThresh);
			Json += FString::Printf(TEXT("  \"red_black_source_sample_radius\": %d,\n"), SampleRadius);
			Json += FString::Printf(TEXT("  \"connector_count\": %d,\n"), Connections.Num());
			Json += TEXT("  \"connectors\": [\n");
			for (int32 i = 0; i < Connections.Num(); ++i)
			{
				const FConn& C = Connections[i];
				Json += TEXT("    {\n");
				Json += FString::Printf(TEXT("      \"id\": %d,\n"), i + 1);
				Json += FString::Printf(TEXT("      \"source\": \"%s\",\n"), *C.Source);
				Json += FString::Printf(TEXT("      \"p0\": [%d, %d],\n"), C.P0.X, C.P0.Y);
				Json += FString::Printf(TEXT("      \"p1\": [%d, %d],\n"), C.P1.X, C.P1.Y);
				Json += FString::Printf(TEXT("      \"line_pixels\": %d,\n"), C.LinePts.Num());
				Json += FString::Printf(TEXT("      \"added_pixels\": %d,\n"), C.AddedPts.Num());
				Json += FString::Printf(TEXT("      \"rb_alt_path_found\": %s,\n"), C.bRbAltPath ? TEXT("true") : TEXT("false"));
				Json += FString::Printf(TEXT("      \"rb_alt_path_pixels\": %d,\n"), C.RbAltPathPixels);
				Json += FString::Printf(TEXT("      \"rb_bbox_area\": %lld,\n"), C.RbBboxArea);
				Json += FString::Printf(TEXT("      \"protected_keep\": %s,\n"), C.bProtectedKeep ? TEXT("true") : TEXT("false"));
				Json += FString::Printf(TEXT("      \"full_skipped_by_protection\": %s,\n"), C.bFullSkippedByProtection ? TEXT("true") : TEXT("false"));
				Json += FString::Printf(TEXT("      \"full_added_pixels_checked\": %d,\n"), C.FullAddedPixels);
				Json += FString::Printf(TEXT("      \"full_alt_path_found\": %s,\n"), C.bFullAltPath ? TEXT("true") : TEXT("false"));
				Json += FString::Printf(TEXT("      \"full_alt_path_pixels\": %d,\n"), C.FullAltPathPixels);
				Json += FString::Printf(TEXT("      \"full_bbox_area\": %lld,\n"), C.FullBboxArea);
				Json += FString::Printf(TEXT("      \"kept\": %s,\n"), C.bKept ? TEXT("true") : TEXT("false"));
				Json += FString::Printf(TEXT("      \"decision\": \"%s\"\n"), *C.Decision);
				Json += FString::Printf(TEXT("    }%s\n"), (i + 1 < Connections.Num()) ? TEXT(",") : TEXT(""));
			}
			Json += TEXT("  ]\n");
			Json += TEXT("}\n");
			FFileHelper::SaveStringToFile(Json, *ConnectorPruneDebugJsonPath);
		}

		// ---- 3. trim dangling branches ----------------------------------------
		// Branch-prune endpoint set + stop nodes are taken from the small-loop-pruned snapshot.
		TArray<FIntPoint> PrunePoints;
		if (LoopThresh > 0.0f)
		{
			SkelGraph::FindEndpoints(Current.GetData(), Width, Height, PrunePoints);
		}
		else
		{
			for (int32 i = 0; i < E; ++i)
			{
				if (!Used[i])
				{
					PrunePoints.Add(EndpointsBefore[i]);
				}
			}
		}

		float MaxPixels = -1.0f;
		if (BranchPruneMaxPixels > 0.0f)
		{
			MaxPixels = BranchPruneMaxPixels;
		}
		else if (LoopThresh > 0.0f)
		{
			MaxPixels = FMath::Max(30.0f, 3.0f * GapTol);
		}

		TSet<int32> StopNodes;
		SkelGraph::BranchStopPoints(Current.GetData(), Width, Height, StopNodes);

		auto SamplesAsProtectedSourceColor = [&](const FIntPoint& P) -> bool
		{
			if (!bHasSourceColorMap)
			{
				return false;
			}

			int32 ProtectedSamples = 0;
			int32 ColoredSamples = 0;
			const int32 MinY = FMath::Max(0, P.Y - SampleRadius);
			const int32 MaxY = FMath::Min(Height - 1, P.Y + SampleRadius);
			const int32 MinX = FMath::Max(0, P.X - SampleRadius);
			const int32 MaxX = FMath::Min(Width - 1, P.X + SampleRadius);
			for (int32 yy = MinY; yy <= MaxY; ++yy)
			{
				for (int32 xx = MinX; xx <= MaxX; ++xx)
				{
					const uint8 C = SourceColorMap[yy * Width + xx];
					if (C == uint8(EStrokeColor::None))
					{
						continue;
					}
					++ColoredSamples;
					if (C == uint8(EStrokeColor::Red) || C == uint8(EStrokeColor::Black))
					{
						++ProtectedSamples;
					}
				}
			}

			return ProtectedSamples > 0 && ProtectedSamples * 2 >= ColoredSamples;
		};

		auto BranchHasProtectedSourceColor = [&](const TArray<FIntPoint>& BranchPath) -> bool
		{
			// The prune operation removes all but the terminal node, so classify only
			// pixels that would actually be deleted.
			const int32 DeleteCount = FMath::Max(0, BranchPath.Num() - 1);
			if (DeleteCount == 0)
			{
				return false;
			}

			int32 ProtectedPathSamples = 0;
			for (int32 k = 0; k < DeleteCount; ++k)
			{
				if (SamplesAsProtectedSourceColor(BranchPath[k]))
				{
					++ProtectedPathSamples;
				}
			}

			const int32 Needed = FMath::Max(1, (DeleteCount + 1) / 2);
			return ProtectedPathSamples >= Needed;
		};

		TArray<uint8> Cleaned = Current;
		TArray<FIntPoint> Path;
		for (const FIntPoint& P : PrunePoints)
		{
			if (P.X < 0 || P.X >= Width || P.Y < 0 || P.Y >= Height)
			{
				continue;
			}
			const int32 Idx = P.Y * Width + P.X;
			if (Cleaned[Idx] == 0)
			{
				continue;
			}
			if (SkelGraph::NodeType(Cleaned.GetData(), Width, Height, P.X, P.Y) != SkelGraph::ENode::Endpoint)
			{
				continue;
			}
			SkelGraph::TraceBranchPath(Cleaned.GetData(), Width, Height, P, StopNodes, Path);
			if (Path.Num() == 0)
			{
				continue;
			}
			if (MaxPixels > 0.0f && Path.Num() > MaxPixels)
			{
				continue; // branch too long to be noise
			}
			if (BranchHasProtectedSourceColor(Path))
			{
				continue; // preserve short branches that still map to source red/black strokes
			}
			// Remove all but the terminal node.
			for (int32 k = 0; k < Path.Num() - 1; ++k)
			{
				Cleaned[Path[k].Y * Width + Path[k].X] = 0;
			}
			const FIntPoint End = Path.Last();
			const int32 EndIdx = End.Y * Width + End.X;
			if (!StopNodes.Contains(EndIdx) &&
				SkelGraph::NodeType(Cleaned.GetData(), Width, Height, End.X, End.Y) == SkelGraph::ENode::Endpoint)
			{
				Cleaned[EndIdx] = 0;
			}
		}

		OutCleaned = MoveTemp(Cleaned);
	}

	// ====================================================================
	// Step 4: stroke tracing (port of Python trace_strokes).
	// ====================================================================
	void TraceStrokes(const TArray<uint8>& Skel, int32 Width, int32 Height, int32 MinPixels, TArray<FStroke>& OutStrokes)
	{
		OutStrokes.Reset();
		const uint8* M = Skel.GetData();

		auto EdgeKey = [Width](int32 a, int32 b) -> uint64
		{
			const uint32 lo = uint32(FMath::Min(a, b));
			const uint32 hi = uint32(FMath::Max(a, b));
			return (uint64(lo) << 32) | uint64(hi);
		};

		// True endpoints / branches are the topological nodes we trace between.
		TSet<int32> Nodes;
		for (int32 y = 0; y < Height; ++y)
		{
			for (int32 x = 0; x < Width; ++x)
			{
				if (M[y * Width + x] == 0)
				{
					continue;
				}
				const SkelGraph::ENode T = SkelGraph::NodeType(M, Width, Height, x, y);
				if (T == SkelGraph::ENode::Endpoint || T == SkelGraph::ENode::Branch)
				{
					Nodes.Add(y * Width + x);
				}
			}
		}

		TSet<uint64> VisitedEdges;

		auto IdxToPoint = [Width](int32 Idx) -> FVector2D
		{
			return FVector2D(double(Idx % Width), double(Idx / Width));
		};

		// Walk one polyline starting along edge (Start -> First). bStopAtNodes mirrors
		// the first Python pass (break when reaching another node); the second pass
		// instead closes pure cycles back onto Start.
		auto WalkPath = [&](int32 Start, int32 First, bool bStopAtNodes, TArray<int32>& OutPath)
		{
			OutPath.Reset();
			OutPath.Add(Start);
			int32 Prev = Start;
			int32 Cur = First;
			int32 Safety = 0;
			for (;;)
			{
				if (++Safety > 20000)
				{
					break;
				}
				OutPath.Add(Cur);

				if (bStopAtNodes && Nodes.Contains(Cur) && Cur != Start)
				{
					break;
				}

				const int32 cx = Cur % Width;
				const int32 cy = Cur / Width;
				int32 Nb[8];
				const int32 Deg = SkelGraph::SafeNeighbors(M, Width, Height, cx, cy, Nb);

				int32 Cands[8];
				int32 NumCands = 0;
				for (int32 i = 0; i < Deg; ++i)
				{
					if (Nb[i] == Prev)
					{
						continue;
					}
					if (VisitedEdges.Contains(EdgeKey(Cur, Nb[i])))
					{
						continue;
					}
					Cands[NumCands++] = Nb[i];
				}
				if (NumCands == 0)
				{
					break;
				}

				const FVector2D PrevPt = IdxToPoint(Prev);
				const FVector2D CurPt = IdxToPoint(Cur);
				const int32 Next = SkelGraph::ChooseBestContinuation(
					FIntPoint(int32(PrevPt.X), int32(PrevPt.Y)),
					FIntPoint(int32(CurPt.X), int32(CurPt.Y)),
					Cands, NumCands, Width);
				if (Next < 0)
				{
					break;
				}

				if (!bStopAtNodes && Next == Start)
				{
					VisitedEdges.Add(EdgeKey(Cur, Next));
					OutPath.Add(Start);
					break;
				}

				VisitedEdges.Add(EdgeKey(Cur, Next));
				Prev = Cur;
				Cur = Next;
			}
		};

		auto EmitPath = [&](const TArray<int32>& Path)
		{
			if (Path.Num() < MinPixels)
			{
				return;
			}
			FStroke S;
			S.Reserve(Path.Num());
			for (int32 Idx : Path)
			{
				S.Add(IdxToPoint(Idx));
			}
			OutStrokes.Add(MoveTemp(S));
		};

		TArray<int32> Path;

		// Pass 1: trace from true nodes (raster order for determinism).
		TArray<int32> NodeList = Nodes.Array();
		NodeList.Sort();
		for (int32 Start : NodeList)
		{
			const int32 sx = Start % Width;
			const int32 sy = Start / Width;
			int32 Nb[8];
			const int32 Deg = SkelGraph::SafeNeighbors(M, Width, Height, sx, sy, Nb);
			for (int32 i = 0; i < Deg; ++i)
			{
				const uint64 EK = EdgeKey(Start, Nb[i]);
				if (VisitedEdges.Contains(EK))
				{
					continue;
				}
				VisitedEdges.Add(EK);
				WalkPath(Start, Nb[i], /*bStopAtNodes*/ true, Path);
				EmitPath(Path);
			}
		}

		// Pass 2: remaining pure cycles / unvisited chains.
		for (int32 y = 0; y < Height; ++y)
		{
			for (int32 x = 0; x < Width; ++x)
			{
				const int32 P = y * Width + x;
				if (M[P] == 0)
				{
					continue;
				}
				int32 Nb[8];
				const int32 Deg = SkelGraph::SafeNeighbors(M, Width, Height, x, y, Nb);
				for (int32 i = 0; i < Deg; ++i)
				{
					const uint64 EK = EdgeKey(P, Nb[i]);
					if (VisitedEdges.Contains(EK))
					{
						continue;
					}
					VisitedEdges.Add(EK);
					WalkPath(P, Nb[i], /*bStopAtNodes*/ false, Path);
					EmitPath(Path);
				}
			}
		}
	}

	// ====================================================================
	// Step 5: corner splitting (port of Python split_stroke_at_corners).
	// ====================================================================
	namespace StrokeMath
	{
		// Principal (largest-eigenvalue) PCA axis direction of S[a..b] inclusive.
		// Returns a normalized vector, or (0,0) when degenerate.
		FVector2D PcaDir(const FStroke& S, int32 a, int32 b)
		{
			const int32 Cnt = b - a + 1;
			if (Cnt < 2)
			{
				return FVector2D::ZeroVector;
			}
			FVector2D Mean(0, 0);
			for (int32 k = a; k <= b; ++k)
			{
				Mean += S[k];
			}
			Mean /= double(Cnt);
			double Sxx = 0, Sxy = 0, Syy = 0;
			for (int32 k = a; k <= b; ++k)
			{
				const double dx = S[k].X - Mean.X;
				const double dy = S[k].Y - Mean.Y;
				Sxx += dx * dx; Sxy += dx * dy; Syy += dy * dy;
			}
			const double Tr = Sxx + Syy;
			const double Det = Sxx * Syy - Sxy * Sxy;
			const double Tmp = FMath::Sqrt(FMath::Max(0.0, Tr * Tr * 0.25 - Det));
			const double L1 = Tr * 0.5 + Tmp; // largest eigenvalue
			FVector2D Dir;
			if (FMath::Abs(Sxy) > 1e-12)
			{
				Dir = FVector2D(Sxy, L1 - Sxx); // eigenvector for L1
			}
			else
			{
				Dir = (Sxx >= Syy) ? FVector2D(1, 0) : FVector2D(0, 1);
			}
			if (Dir.Size() < 1e-12)
			{
				return FVector2D::ZeroVector;
			}
			Dir.Normalize();
			return Dir;
		}

		// Unoriented angle (degrees, 0..90) between two segment PCA axes.
		double SegAxisAngleDeg(const FStroke& S, int32 la, int32 lb, int32 ra, int32 rb)
		{
			const FVector2D D1 = PcaDir(S, la, lb);
			const FVector2D D2 = PcaDir(S, ra, rb);
			if (D1.IsNearlyZero() || D2.IsNearlyZero())
			{
				return 0.0;
			}
			double C = FMath::Abs(FVector2D::DotProduct(D1, D2));
			C = FMath::Clamp(C, 0.0, 1.0);
			return FMath::RadiansToDegrees(FMath::Acos(C));
		}

		int32 WalkLeftByArc(const FStroke& S, int32 i, int32 StopI, double MaxArc)
		{
			int32 Start = i;
			double Total = 0.0;
			for (int32 k = i; k > StopI; --k)
			{
				Total += (S[k] - S[k - 1]).Size();
				Start = k - 1;
				if (Total >= MaxArc)
				{
					break;
				}
			}
			return Start;
		}

		// Exclusive right end index. StopI < 0 means "no next split" (-> n-1).
		int32 WalkRightByArc(const FStroke& S, int32 i, int32 StopI, double MaxArc)
		{
			const int32 N = S.Num();
			if (StopI < 0)
			{
				StopI = N - 1;
			}
			StopI = FMath::Clamp(StopI, i, N - 1);
			int32 End = i + 1;
			double Total = 0.0;
			for (int32 k = i; k < StopI; ++k)
			{
				Total += (S[k + 1] - S[k]).Size();
				End = k + 2;
				if (Total >= MaxArc)
				{
					break;
				}
			}
			return End;
		}
	} // namespace StrokeMath

	// Compute the accepted corner-split indices for one stroke. Returns sorted indices.
	static void ComputeCornerSplitIndices(
		const FStroke& S, float AngleThresh, int32 MinPixels,
		float SegmentArc, float SplitPeakMinDistance, int32 MaxIters,
		TArray<int32>& OutSplits)
	{
		using namespace StrokeMath;
		OutSplits.Reset();
		const int32 N = S.Num();
		if (AngleThresh <= 0.0f || N < FMath::Max(3, MinPixels * 2 + 1))
		{
			return;
		}
		const double SegArc = FMath::Max(1.0, double(SegmentArc));
		const double PeakMinDist = FMath::Max(0.0, double(SplitPeakMinDistance));
		const int32 OptIters = FMath::Max(1, MaxIters);

		// Arc-length prefix sums.
		TArray<double> Arc;
		Arc.SetNumZeroed(N);
		for (int32 k = 1; k < N; ++k)
		{
			Arc[k] = Arc[k - 1] + (S[k] - S[k - 1]).Size();
		}
		auto ArcDist = [&](int32 i, int32 j) -> double
		{
			i = FMath::Clamp(i, 0, N - 1);
			j = FMath::Clamp(j, 0, N - 1);
			return FMath::Abs(Arc[i] - Arc[j]);
		};

		// Segment-angle for split index i given left bound PrevSplit and right bound
		// NextSplit (-1 = none). Fills window lengths.
		auto SegAngleAt = [&](int32 i, int32 PrevSplit, int32 NextSplit, int32& OutLeftLen, int32& OutRightLen) -> double
		{
			const int32 LeftStart = WalkLeftByArc(S, i, PrevSplit, SegArc);
			const int32 RightEnd = WalkRightByArc(S, i, NextSplit, SegArc);
			OutLeftLen = i - LeftStart + 1;
			OutRightLen = (RightEnd - 1) - i + 1;
			if (OutLeftLen < 2 || OutRightLen < 2)
			{
				return 0.0;
			}
			return SegAxisAngleDeg(S, LeftStart, i, i, RightEnd - 1);
		};

		// Dynamic-neighbor evaluation (context = current selected minus i).
		auto Evaluate = [&](int32 i, const TArray<int32>& SortedContext, double& OutScore) -> bool
		{
			OutScore = 0.0;
			int32 PrevSplit = 0;
			bool bHasPrev = false;
			int32 NextSplit = -1;
			for (int32 C : SortedContext)
			{
				if (C < i) { PrevSplit = C; bHasPrev = true; }
				else if (C > i) { NextSplit = C; break; }
			}

			if (i - PrevSplit < MinPixels)
			{
				return false;
			}
			if (NextSplit < 0)
			{
				if (N - i < MinPixels)
				{
					return false;
				}
			}
			else if (NextSplit - i < MinPixels)
			{
				return false;
			}
			if (bHasPrev && PeakMinDist > 0.0 && ArcDist(i, PrevSplit) < PeakMinDist)
			{
				return false;
			}
			if (NextSplit >= 0 && PeakMinDist > 0.0 && ArcDist(i, NextSplit) < PeakMinDist)
			{
				return false;
			}
			int32 LL = 0, RL = 0;
			const double Angle = SegAngleAt(i, PrevSplit, NextSplit, LL, RL);
			if (LL < 2 || RL < 2)
			{
				return false;
			}
			OutScore = Angle;
			return Angle >= double(AngleThresh);
		};

		// Fixed-bound scan high-score test (segment bounds PrevSplit/NextSplit, both indices).
		auto ScanHigh = [&](int32 i, int32 PrevSplit, int32 NextSplit, double& OutAngle) -> bool
		{
			OutAngle = 0.0;
			if (i - PrevSplit < MinPixels)
			{
				return false;
			}
			if (NextSplit - i < MinPixels)
			{
				return false;
			}
			if (PrevSplit > 0 && PeakMinDist > 0.0 && ArcDist(i, PrevSplit) < PeakMinDist)
			{
				return false;
			}
			if (NextSplit < N - 1 && PeakMinDist > 0.0 && ArcDist(i, NextSplit) < PeakMinDist)
			{
				return false;
			}
			int32 LL = 0, RL = 0;
			const double Angle = SegAngleAt(i, PrevSplit, NextSplit, LL, RL);
			OutAngle = Angle;
			if (LL < 2 || RL < 2)
			{
				return false;
			}
			return Angle >= double(AngleThresh);
		};

		TSet<int32> Selected;
		TSet<FString> SeenSets;
		SeenSets.Add(TEXT("")); // empty set already "seen"

		auto EncodeSet = [](const TArray<int32>& Sorted) -> FString
		{
			FString Key;
			for (int32 V : Sorted)
			{
				Key += FString::Printf(TEXT("%d,"), V);
			}
			return Key;
		};

		for (int32 Pass = 0; Pass < OptIters; ++Pass)
		{
			TArray<int32> SelSorted = Selected.Array();
			SelSorted.Sort();

			// --- scan candidate peaks under current segmentation ---
			TArray<int32> Interior;
			for (int32 V : SelSorted)
			{
				if (V > 0 && V < N - 1)
				{
					Interior.Add(V);
				}
			}
			TArray<int32> Bounds;
			Bounds.Add(0);
			Bounds.Append(Interior);
			Bounds.Add(N - 1);

			TArray<int32> CandidatePeaks;
			for (int32 b = 0; b + 1 < Bounds.Num(); ++b)
			{
				const int32 PrevSplit = Bounds[b];
				const int32 NextSplit = Bounds[b + 1];

				// Walk i in (PrevSplit+1 .. NextSplit-1), grouping consecutive high-score points.
				TArray<int32> GroupIdx;
				TArray<double> GroupAngle;
				auto FlushGroup = [&]()
				{
					if (GroupIdx.Num() == 0)
					{
						return;
					}
					// Local maxima of segment angle within the consecutive group.
					TArray<int32> Peaks;
					for (int32 j = 0; j < GroupIdx.Num(); ++j)
					{
						const double Sc = GroupAngle[j];
						const bool bGEleft = (j == 0) || (Sc >= GroupAngle[j - 1]);
						const bool bGEright = (j + 1 >= GroupIdx.Num()) || (Sc >= GroupAngle[j + 1]);
						if (bGEleft && bGEright)
						{
							Peaks.Add(GroupIdx[j]);
						}
					}
					if (Peaks.Num() == 0)
					{
						int32 BestJ = 0;
						for (int32 j = 1; j < GroupIdx.Num(); ++j)
						{
							if (GroupAngle[j] > GroupAngle[BestJ])
							{
								BestJ = j;
							}
						}
						Peaks.Add(GroupIdx[BestJ]);
					}
					CandidatePeaks.Append(Peaks);
					GroupIdx.Reset();
					GroupAngle.Reset();
				};

				for (int32 i = PrevSplit + 1; i < NextSplit; ++i)
				{
					double Angle = 0.0;
					const bool bHigh = ScanHigh(i, PrevSplit, NextSplit, Angle);
					if (bHigh)
					{
						if (GroupIdx.Num() > 0 && i == GroupIdx.Last() + 1)
						{
							GroupIdx.Add(i);
							GroupAngle.Add(Angle);
						}
						else
						{
							FlushGroup();
							GroupIdx.Add(i);
							GroupAngle.Add(Angle);
						}
					}
					else
					{
						FlushGroup();
					}
				}
				FlushGroup();
			}

			// --- evaluate candidate pool (peaks + carried selected) ---
			TSet<int32> Pool;
			for (int32 P : CandidatePeaks)
			{
				Pool.Add(P);
			}
			for (int32 Si : Selected)
			{
				if (Si > 0 && Si < N - 1)
				{
					Pool.Add(Si);
				}
			}

			TArray<int32> PoolSorted = Pool.Array();
			PoolSorted.Sort();

			TArray<int32> Proposed;
			TMap<int32, double> ScoreMap;
			for (int32 Si : PoolSorted)
			{
				TArray<int32> Context = SelSorted;
				Context.Remove(Si);
				double Score = 0.0;
				const bool bOk = Evaluate(Si, Context, Score);
				ScoreMap.Add(Si, Score);
				if (bOk)
				{
					Proposed.Add(Si);
				}
			}

			// --- resolve close-split conflicts (greedy, strongest first) ---
			Proposed.Sort([&](int32 A, int32 B)
			{
				const double Sa = ScoreMap[A];
				const double Sb = ScoreMap[B];
				if (Sa != Sb)
				{
					return Sa > Sb; // higher score first
				}
				return A < B; // tie: smaller index first
			});

			TArray<int32> Kept;
			for (int32 Idx : Proposed)
			{
				bool bTooClose = false;
				if (PeakMinDist > 0.0)
				{
					for (int32 K : Kept)
					{
						if (ArcDist(Idx, K) < PeakMinDist)
						{
							bTooClose = true;
							break;
						}
					}
				}
				if (!bTooClose)
				{
					Kept.Add(Idx);
				}
			}

			TSet<int32> NewSelected;
			NewSelected.Append(Kept);

			// --- convergence check ---
			TArray<int32> NewSorted = NewSelected.Array();
			NewSorted.Sort();
			const FString Key = EncodeSet(NewSorted);
			const bool bSame = (NewSelected.Num() == Selected.Num()) && !NewSelected.Difference(Selected).Num();
			Selected = MoveTemp(NewSelected);
			if (bSame || SeenSets.Contains(Key))
			{
				break;
			}
			SeenSets.Add(Key);
		}

		// Final acceptance: keep selected indices that still pass evaluation
		// against the final segmentation.
		TArray<int32> FinalSel = Selected.Array();
		FinalSel.Sort();
		for (int32 Si : FinalSel)
		{
			TArray<int32> Context = FinalSel;
			Context.Remove(Si);
			double Score = 0.0;
			if (Evaluate(Si, Context, Score))
			{
				OutSplits.Add(Si);
			}
		}
	}

	void SplitStrokesAtCorners(
		const TArray<FStroke>& In, float AngleThresh, int32 MinPixels,
		float SegmentArc, float SplitPeakMinDistance, int32 MaxIters,
		TArray<FStroke>& Out)
	{
		Out.Reset();
		if (AngleThresh <= 0.0f)
		{
			Out = In;
			return;
		}

		TArray<int32> Splits;
		for (const FStroke& S : In)
		{
			ComputeCornerSplitIndices(S, AngleThresh, MinPixels, SegmentArc, SplitPeakMinDistance, MaxIters, Splits);

			if (Splits.Num() == 0)
			{
				Out.Add(S);
				continue;
			}

			int32 Added = 0;
			int32 Start = 0;
			for (int32 SplitI : Splits)
			{
				const int32 Count = SplitI - Start + 1; // inclusive of split point
				if (Count >= MinPixels)
				{
					FStroke Piece(S.GetData() + Start, Count);
					Out.Add(MoveTemp(Piece));
					++Added;
				}
				Start = SplitI;
			}
			const int32 TailCount = S.Num() - Start;
			if (TailCount >= MinPixels)
			{
				FStroke Tail(S.GetData() + Start, TailCount);
				Out.Add(MoveTemp(Tail));
				++Added;
			}
			if (Added == 0)
			{
				Out.Add(S); // never drop a stroke entirely
			}
		}
	}

	bool SaveStrokesPng(const TArray<FStroke>& Strokes, int32 Width, int32 Height, const FString& Path, int32 Thickness)
	{
		const int32 N = Width * Height;
		TArray<uint8> RGBA;
		RGBA.Init(255, N * 4); // white background

		static const uint8 Palette[][3] = {
			{220,  20,  60}, { 30, 144, 255}, { 34, 139,  34}, {255, 140,   0},
			{148,   0, 211}, {  0, 191, 191}, {199,  21, 133}, {139,  69,  19},
			{ 70, 130, 180}, {255,  20, 147}, {107, 142,  35}, {  0,   0,   0},
		};
		const int32 PaletteCount = UE_ARRAY_COUNT(Palette);
		const int32 R = FMath::Max(0, Thickness / 2);

		auto Plot = [&](int32 x, int32 y, const uint8* Col)
		{
			for (int32 oy = -R; oy <= R; ++oy)
			{
				for (int32 ox = -R; ox <= R; ++ox)
				{
					const int32 xx = x + ox;
					const int32 yy = y + oy;
					if (xx < 0 || xx >= Width || yy < 0 || yy >= Height)
					{
						continue;
					}
					const int32 Off = (yy * Width + xx) * 4;
					RGBA[Off + 0] = Col[0];
					RGBA[Off + 1] = Col[1];
					RGBA[Off + 2] = Col[2];
					RGBA[Off + 3] = 255;
				}
			}
		};

		TArray<FIntPoint> LineBuf;
		for (int32 s = 0; s < Strokes.Num(); ++s)
		{
			const uint8* Col = Palette[s % PaletteCount];
			const FStroke& Stroke = Strokes[s];
			for (int32 k = 0; k + 1 < Stroke.Num(); ++k)
			{
				const FIntPoint P0(FMath::RoundToInt(Stroke[k].X), FMath::RoundToInt(Stroke[k].Y));
				const FIntPoint P1(FMath::RoundToInt(Stroke[k + 1].X), FMath::RoundToInt(Stroke[k + 1].Y));
				SkelGraph::LinePoints(P0, P1, LineBuf);
				for (const FIntPoint& P : LineBuf)
				{
					Plot(P.X, P.Y, Col);
				}
			}
		}

		IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
		if (!IW.IsValid())
		{
			return false;
		}
		IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
		const TArray64<uint8>& C = IW->GetCompressed();
		return FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *Path);
	}

	// ====================================================================
	// Color classification + color-aware stroke splitting.
	// ====================================================================
	const TCHAR* StrokeColorToString(EStrokeColor Color)
	{
		switch (Color)
		{
		case EStrokeColor::Black: return TEXT("black");
		case EStrokeColor::Red:   return TEXT("red");
		case EStrokeColor::Green: return TEXT("green");
		case EStrokeColor::Blue:  return TEXT("blue");
		default:                  return TEXT("none");
		}
	}

	EStrokeColor ClassifyRGB(uint8 R, uint8 G, uint8 B)
	{
		// Near-white background.
		if (R > 220 && G > 220 && B > 220)
		{
			return EStrokeColor::None;
		}
		const int32 r = R, g = G, b = B;
		const int32 Dom = 30; // channel dominance margin
		if (r >= g + Dom && r >= b + Dom)
		{
			return EStrokeColor::Red;
		}
		if (g >= r + Dom && g >= b + Dom)
		{
			return EStrokeColor::Green;
		}
		if (b >= r + Dom && b >= g + Dom)
		{
			return EStrokeColor::Blue;
		}
		// Dark or neutral (no clear hue) -> the black line-art render.
		return EStrokeColor::Black;
	}

	void BuildColorClassMap(const TArray<uint8>& RGBA, int32 Width, int32 Height, TArray<uint8>& OutMap)
	{
		const int32 N = Width * Height;
		OutMap.SetNumUninitialized(N);
		for (int32 i = 0; i < N; ++i)
		{
			const int32 Off = i * 4;
			OutMap[i] = uint8(ClassifyRGB(RGBA[Off + 0], RGBA[Off + 1], RGBA[Off + 2]));
		}
	}

	EStrokeColor SampleColorAt(const TArray<uint8>& ColorMap, int32 Width, int32 Height, int32 X, int32 Y, int32 Radius)
	{
		int32 Counts[5] = { 0, 0, 0, 0, 0 };
		for (int32 oy = -Radius; oy <= Radius; ++oy)
		{
			const int32 yy = Y + oy;
			if (yy < 0 || yy >= Height)
			{
				continue;
			}
			for (int32 ox = -Radius; ox <= Radius; ++ox)
			{
				const int32 xx = X + ox;
				if (xx < 0 || xx >= Width)
				{
					continue;
				}
				const uint8 C = ColorMap[yy * Width + xx];
				if (C != uint8(EStrokeColor::None))
				{
					++Counts[C];
				}
			}
		}
		int32 Best = int32(EStrokeColor::None);
		int32 BestCount = 0;
		for (int32 c = 1; c <= 4; ++c)
		{
			if (Counts[c] > BestCount)
			{
				BestCount = Counts[c];
				Best = c;
			}
		}
		return EStrokeColor(Best);
	}

	namespace ColorSplit
	{
		FORCEINLINE bool IsPrimary(EStrokeColor C)
		{
			return C == EStrokeColor::Red || C == EStrokeColor::Green || C == EStrokeColor::Blue;
		}

		// Unoriented direction of S[a..b]; falls back to PCA, then zero.
		FVector2D RunDir(const FStroke& S, int32 a, int32 b)
		{
			if (a == b)
			{
				return FVector2D::ZeroVector;
			}
			FVector2D D = S[b] - S[a];
			if (D.Size() > 1e-6)
			{
				D.Normalize();
				return D;
			}
			return StrokeMath::PcaDir(S, FMath::Min(a, b), FMath::Max(a, b));
		}

		double UnorientedAngleDeg(const FVector2D& A, const FVector2D& B)
		{
			if (A.IsNearlyZero() || B.IsNearlyZero())
			{
				return 90.0; // treat unknown as worst alignment
			}
			double C = FMath::Abs(FVector2D::DotProduct(A, B));
			C = FMath::Clamp(C, 0.0, 1.0);
			return FMath::RadiansToDegrees(FMath::Acos(C));
		}

		// Classify a gap-repair (None) run from its neighbor colors and directions.
		EStrokeColor ClassifyConnection(
			EStrokeColor Left, EStrokeColor Right,
			const FVector2D& ConnDir, const FVector2D& LeftDir, const FVector2D& RightDir)
		{
			if (Left == EStrokeColor::None && Right == EStrokeColor::None)
			{
				return EStrokeColor::Black;
			}
			if (Left == EStrokeColor::None)
			{
				return Right;
			}
			if (Right == EStrokeColor::None)
			{
				return Left;
			}
			if (Left == Right)
			{
				return Left;
			}

			// Both colored but different.
			if (IsPrimary(Left) && IsPrimary(Right))
			{
				const double AngLeft = UnorientedAngleDeg(ConnDir, LeftDir);
				const double AngRight = UnorientedAngleDeg(ConnDir, RightDir);
				return (AngRight <= AngLeft) ? Right : Left; // nearest-aligned neighbor wins
			}

			// One side is Black, the other a primary.
			const EStrokeColor Primary = (Left == EStrokeColor::Black) ? Right : Left;
			switch (Primary)
			{
			case EStrokeColor::Red:   return EStrokeColor::Black; // red | black -> black
			case EStrokeColor::Green: return EStrokeColor::Green; // green | black -> green
			case EStrokeColor::Blue:  return EStrokeColor::Blue;  // blue | black -> blue
			default:                  return EStrokeColor::Black;
			}
		}
	} // namespace ColorSplit

	void ColorizeAndSplitStrokes(
		const TArray<FStroke>& In, const TArray<uint8>& ColorMap, int32 Width, int32 Height,
		int32 SampleRadius, float MinRunArc, TArray<FColoredStroke>& Out)
	{
		using namespace ColorSplit;
		Out.Reset();

		// Build maximal equal-color runs of a per-point label array as [Start,End] index pairs.
		auto BuildRuns = [](const TArray<EStrokeColor>& Lab, TArray<int32>& RunStart, TArray<int32>& RunEnd, TArray<EStrokeColor>& RunCol)
		{
			RunStart.Reset(); RunEnd.Reset(); RunCol.Reset();
			const int32 N = Lab.Num();
			for (int32 i = 0; i < N; )
			{
				int32 j = i;
				while (j + 1 < N && Lab[j + 1] == Lab[i])
				{
					++j;
				}
				RunStart.Add(i); RunEnd.Add(j); RunCol.Add(Lab[i]);
				i = j + 1;
			}
		};

		for (const FStroke& S : In)
		{
			const int32 N = S.Num();
			if (N == 0)
			{
				continue;
			}

			// Per-point color class + original "synthetic" (None) flag.
			TArray<EStrokeColor> Cls;
			Cls.SetNumUninitialized(N);
			TArray<uint8> WasNone;
			WasNone.SetNumUninitialized(N);
			for (int32 k = 0; k < N; ++k)
			{
				Cls[k] = SampleColorAt(ColorMap, Width, Height, FMath::RoundToInt(S[k].X), FMath::RoundToInt(S[k].Y), SampleRadius);
				WasNone[k] = (Cls[k] == EStrokeColor::None) ? 1 : 0;
			}

			// Arc-length prefix sums for run-length thresholds.
			TArray<double> Arc;
			Arc.SetNumZeroed(N);
			for (int32 k = 1; k < N; ++k)
			{
				Arc[k] = Arc[k - 1] + (S[k] - S[k - 1]).Size();
			}

			TArray<int32> RunStart, RunEnd;
			TArray<EStrokeColor> RunCol;
			TArray<EStrokeColor> Final = Cls;

			// 1) Reclassify synthetic (None) runs from their neighbor color/direction.
			BuildRuns(Final, RunStart, RunEnd, RunCol);
			for (int32 r = 0; r < RunCol.Num(); ++r)
			{
				if (RunCol[r] != EStrokeColor::None)
				{
					continue;
				}
				const EStrokeColor LeftC = (r > 0) ? RunCol[r - 1] : EStrokeColor::None;
				const EStrokeColor RightC = (r + 1 < RunCol.Num()) ? RunCol[r + 1] : EStrokeColor::None;
				const FVector2D ConnDir = RunDir(S, RunStart[r], RunEnd[r]);
				const FVector2D LeftDir = (r > 0) ? RunDir(S, RunStart[r - 1], RunEnd[r - 1]) : FVector2D::ZeroVector;
				const FVector2D RightDir = (r + 1 < RunCol.Num()) ? RunDir(S, RunStart[r + 1], RunEnd[r + 1]) : FVector2D::ZeroVector;
				const EStrokeColor NC = ClassifyConnection(LeftC, RightC, ConnDir, LeftDir, RightDir);
				for (int32 k = RunStart[r]; k <= RunEnd[r]; ++k)
				{
					Final[k] = NC;
				}
			}

			// 2) Absorb short runs (e.g. black blips at colored/black crossings) into neighbors.
			for (;;)
			{
				BuildRuns(Final, RunStart, RunEnd, RunCol);
				const int32 RunCount = RunCol.Num();
				if (RunCount <= 1)
				{
					break;
				}
				auto RunArc = [&](int32 i) -> double { return Arc[RunEnd[i]] - Arc[RunStart[i]]; };

				// Sub-threshold runs, shortest first.
				TArray<int32> Cand;
				for (int32 i = 0; i < RunCount; ++i)
				{
					if (RunArc(i) < double(MinRunArc))
					{
						Cand.Add(i);
					}
				}
				Cand.Sort([&](int32 A, int32 B) { return RunArc(A) < RunArc(B); });

				bool bApplied = false;
				for (int32 Idx : Cand)
				{
					const EStrokeColor LeftC = (Idx > 0) ? RunCol[Idx - 1] : EStrokeColor::None;
					const EStrokeColor RightC = (Idx + 1 < RunCount) ? RunCol[Idx + 1] : EStrokeColor::None;
					EStrokeColor NC = RunCol[Idx];
					if (LeftC != EStrokeColor::None && RightC != EStrokeColor::None)
					{
						NC = (LeftC == RightC) ? LeftC
							: (RunArc(Idx - 1) >= RunArc(Idx + 1) ? LeftC : RightC);
					}
					else if (LeftC != EStrokeColor::None)
					{
						NC = LeftC;
					}
					else if (RightC != EStrokeColor::None)
					{
						NC = RightC;
					}

					if (NC != RunCol[Idx])
					{
						for (int32 k = RunStart[Idx]; k <= RunEnd[Idx]; ++k)
						{
							Final[k] = NC;
						}
						bApplied = true;
						break;
					}
				}
				if (!bApplied)
				{
					break;
				}
			}

			// 3) Emit one mono-color piece per final run.
			BuildRuns(Final, RunStart, RunEnd, RunCol);
			for (int32 r = 0; r < RunCol.Num(); ++r)
			{
				const int32 StartIdx = RunStart[r];
				const int32 EndIdx = RunEnd[r];
				FColoredStroke CS;
				CS.Color = RunCol[r];
				CS.Points = FStroke(S.GetData() + StartIdx, EndIdx - StartIdx + 1);
				int32 ConnPts = 0;
				for (int32 k = StartIdx; k <= EndIdx; ++k)
				{
					ConnPts += WasNone[k];
				}
				CS.ConnectionPointCount = ConnPts;
				Out.Add(MoveTemp(CS));
			}
		}
	}

	void SplitColoredStrokesAtCorners(
		const TArray<FColoredStroke>& In, float AngleThresh, int32 MinPixels,
		float SegmentArc, float SplitPeakMinDistance, int32 MaxIters,
		TArray<FColoredStroke>& Out)
	{
		Out.Reset();
		TArray<int32> Splits;
		for (const FColoredStroke& CS : In)
		{
			if (AngleThresh <= 0.0f)
			{
				Out.Add(CS);
				continue;
			}
			ComputeCornerSplitIndices(CS.Points, AngleThresh, MinPixels, SegmentArc, SplitPeakMinDistance, MaxIters, Splits);
			if (Splits.Num() == 0)
			{
				Out.Add(CS);
				continue;
			}

			int32 Added = 0;
			int32 Start = 0;
			auto Emit = [&](int32 S0, int32 Count)
			{
				if (Count < MinPixels)
				{
					return;
				}
				FColoredStroke Piece;
				Piece.Color = CS.Color;
				Piece.Points = FStroke(CS.Points.GetData() + S0, Count);
				Out.Add(MoveTemp(Piece));
				++Added;
			};
			for (int32 SplitI : Splits)
			{
				Emit(Start, SplitI - Start + 1);
				Start = SplitI;
			}
			Emit(Start, CS.Points.Num() - Start);
			if (Added == 0)
			{
				Out.Add(CS);
			}
		}
	}

	// ====================================================================
	// Step 6: same-color merge of corner-split fragments.
	// ====================================================================
	namespace MergeOps
	{
		FVector2D EndpointChordAxis(const FStroke& S)
		{
			if (S.Num() < 2)
			{
				return FVector2D::ZeroVector;
			}
			FVector2D D = S.Last() - S[0];
			if (D.Size() < 1e-8)
			{
				return FVector2D::ZeroVector;
			}
			D.Normalize();
			return D;
		}

		FStroke MergePolylineByEndpoints(const FStroke& S1, int32 End1, const FStroke& S2, int32 End2)
		{
			FStroke A = S1;
			if (End1 == 0)
			{
				Algo::Reverse(A);
			}
			FStroke B = S2;
			if (End2 == 1)
			{
				Algo::Reverse(B);
			}
			FStroke Out = A;
			int32 StartB = 0;
			if (Out.Num() > 0 && B.Num() > 0 && (Out.Last() - B[0]).Size() < 1e-6)
			{
				StartB = 1;
			}
			for (int32 k = StartB; k < B.Num(); ++k)
			{
				Out.Add(B[k]);
			}
			return Out;
		}

		struct FMergeInfo
		{
			bool bOk = false;
			int32 End1 = 0;
			int32 End2 = 0;
			double Gap = 0.0;
			double Angle = 0.0;
			FVector2D MergePoint = FVector2D::ZeroVector;
		};

		FMergeInfo CanPostSplitMerge(const FStroke& A, const FStroke& B, float MaxGap, float MaxAngle)
		{
			FMergeInfo R;
			if (A.Num() < 2 || B.Num() < 2)
			{
				return R;
			}
			const FVector2D D1 = StrokeMath::PcaDir(A, 0, A.Num() - 1);
			const FVector2D D2 = StrokeMath::PcaDir(B, 0, B.Num() - 1);
			if (D1.IsNearlyZero() || D2.IsNearlyZero())
			{
				return R;
			}
			const double Angle = ColorSplit::UnorientedAngleDeg(D1, D2);
			if (Angle > double(MaxAngle))
			{
				return R;
			}
			const FVector2D C1 = EndpointChordAxis(A);
			const FVector2D C2 = EndpointChordAxis(B);
			if (C1.IsNearlyZero() || C2.IsNearlyZero())
			{
				return R;
			}

			double BestGap = TNumericLimits<double>::Max();
			for (int32 E1 = 0; E1 <= 1; ++E1)
			{
				const FVector2D P1 = (E1 == 0) ? A[0] : A.Last();
				for (int32 E2 = 0; E2 <= 1; ++E2)
				{
					const FVector2D P2 = (E2 == 0) ? B[0] : B.Last();
					const double Gap = FVector2D::Distance(P1, P2);
					if (Gap > double(MaxGap))
					{
						continue;
					}
					const FStroke Merged = MergePolylineByEndpoints(A, E1, B, E2);
					const FVector2D MC = EndpointChordAxis(Merged);
					if (MC.IsNearlyZero())
					{
						continue;
					}
					const double MergedEndpointAngle = FMath::Max(
						ColorSplit::UnorientedAngleDeg(MC, C1),
						ColorSplit::UnorientedAngleDeg(MC, C2));
					if (MergedEndpointAngle > double(MaxAngle))
					{
						continue;
					}
					if (Gap < BestGap)
					{
						BestGap = Gap;
						R.bOk = true;
						R.End1 = E1;
						R.End2 = E2;
						R.Gap = Gap;
						R.Angle = Angle;
						R.MergePoint = (P1 + P2) * 0.5;
					}
				}
			}
			return R;
		}

		bool ThirdEndpointNearMergePoint(const TArray<FColoredStroke>& Strokes, int32 I, int32 J, const FVector2D& MergePoint, float Radius)
		{
			if (Radius <= 0.0f)
			{
				return false;
			}
			const double R2 = double(Radius) * double(Radius);
			for (int32 k = 0; k < Strokes.Num(); ++k)
			{
				if (k == I || k == J || Strokes[k].Points.Num() == 0)
				{
					continue;
				}
				if (FVector2D::DistSquared(Strokes[k].Points[0], MergePoint) <= R2 ||
					FVector2D::DistSquared(Strokes[k].Points.Last(), MergePoint) <= R2)
				{
					return true;
				}
			}
			return false;
		}
	} // namespace MergeOps

	void MergeColoredStrokesSameColor(
		const TArray<FColoredStroke>& In, float MaxGap, float MaxAngle, int32 MaxIters,
		float ProtectJunctionRadius, TArray<FColoredStroke>& Out)
	{
		using namespace MergeOps;
		Out = In;

		for (int32 Iter = 0; Iter < MaxIters; ++Iter)
		{
			int32 BestI = -1, BestJ = -1;
			FMergeInfo BestInfo;
			double BestCost = TNumericLimits<double>::Max();

			for (int32 i = 0; i < Out.Num(); ++i)
			{
				for (int32 j = i + 1; j < Out.Num(); ++j)
				{
					if (Out[i].Color != Out[j].Color) // same color class only
					{
						continue;
					}
					const FMergeInfo Info = CanPostSplitMerge(Out[i].Points, Out[j].Points, MaxGap, MaxAngle);
					if (!Info.bOk)
					{
						continue;
					}
					if (ThirdEndpointNearMergePoint(Out, i, j, Info.MergePoint, ProtectJunctionRadius))
					{
						continue; // protect true junction
					}
					const double Cost = Info.Gap + 0.1 * Info.Angle;
					if (Cost < BestCost)
					{
						BestCost = Cost;
						BestI = i;
						BestJ = j;
						BestInfo = Info;
					}
				}
			}

			if (BestI < 0)
			{
				break;
			}

			FColoredStroke Merged;
			Merged.Color = Out[BestI].Color;
			Merged.Points = MergePolylineByEndpoints(Out[BestI].Points, BestInfo.End1, Out[BestJ].Points, BestInfo.End2);
			Merged.ConnectionPointCount = Out[BestI].ConnectionPointCount + Out[BestJ].ConnectionPointCount;

			Out.RemoveAt(BestJ); // BestJ > BestI, remove higher index first
			Out.RemoveAt(BestI);
			Out.Add(MoveTemp(Merged));
		}
	}

	// ====================================================================
	// Step 7: stroke geometry metrics.
	// ====================================================================
	static double Percentile90(TArray<double>& V)
	{
		const int32 N = V.Num();
		if (N == 0)
		{
			return 0.0;
		}
		V.Sort();
		if (N == 1)
		{
			return V[0];
		}
		const double Rank = 0.9 * double(N - 1);
		const int32 Lo = FMath::FloorToInt(Rank);
		const int32 Hi = FMath::Min(Lo + 1, N - 1);
		const double Frac = Rank - double(Lo);
		return V[Lo] + (V[Hi] - V[Lo]) * Frac;
	}

	void ComputeStrokeMetrics(TArray<FColoredStroke>& InOut)
	{
		for (FColoredStroke& CS : InOut)
		{
			const FStroke& P = CS.Points;
			const int32 N = P.Num();
			CS.bHasMetrics = true;
			if (N < 2)
			{
				CS.Arc = 0; CS.Chord = 0; CS.Straightness = 0;
				CS.P90PcaError = 0; CS.PcaRmsError = 0; CS.P90ChordDev = 0; CS.ChordDevRatio = 0;
				CS.Direction = FVector2D::ZeroVector;
				continue;
			}

			double Arc = 0.0;
			for (int32 k = 1; k < N; ++k)
			{
				Arc += (P[k] - P[k - 1]).Size();
			}
			const double Chord = (P[N - 1] - P[0]).Size();
			CS.Arc = Arc;
			CS.Chord = Chord;
			CS.Straightness = (Arc < 1e-8) ? 0.0 : Chord / Arc;

			// PCA principal axis + line (normal form a*x + b*y + c = 0, |(a,b)| = 1).
			const FVector2D Dir = StrokeMath::PcaDir(P, 0, N - 1);
			CS.Direction = Dir;
			FVector2D Center(0, 0);
			for (const FVector2D& Pt : P)
			{
				Center += Pt;
			}
			Center /= double(N);

			TArray<double> PcaDist;
			PcaDist.SetNumUninitialized(N);
			if (!Dir.IsNearlyZero())
			{
				const FVector2D Normal(-Dir.Y, Dir.X);
				const double C = -FVector2D::DotProduct(Normal, Center);
				for (int32 k = 0; k < N; ++k)
				{
					PcaDist[k] = FMath::Abs(Normal.X * P[k].X + Normal.Y * P[k].Y + C);
				}
			}
			else
			{
				for (int32 k = 0; k < N; ++k)
				{
					PcaDist[k] = 0.0;
				}
			}
			double SumSq = 0.0;
			for (double d : PcaDist)
			{
				SumSq += d * d;
			}
			CS.PcaRmsError = FMath::Sqrt(SumSq / double(N));
			CS.P90PcaError = Percentile90(PcaDist);

			// Chord-line deviation.
			FVector2D V = P[N - 1] - P[0];
			if (V.Size() < 1e-8 || Chord < 1e-8)
			{
				CS.P90ChordDev = 0.0;
				CS.ChordDevRatio = 1e9; // "inf"
			}
			else
			{
				const double Vn = V.Size();
				const FVector2D NormalChord(-V.Y / Vn, V.X / Vn);
				const double CC = -FVector2D::DotProduct(NormalChord, P[0]);
				TArray<double> ChordDist;
				ChordDist.SetNumUninitialized(N);
				for (int32 k = 0; k < N; ++k)
				{
					ChordDist[k] = FMath::Abs(NormalChord.X * P[k].X + NormalChord.Y * P[k].Y + CC);
				}
				CS.P90ChordDev = Percentile90(ChordDist);
				CS.ChordDevRatio = CS.P90ChordDev / Chord;
			}
		}
	}

	// ====================================================================
	// Step 8: enclosed-region mask (endpoint-nearest-connect + flood).
	// ====================================================================
	void ComputeEnclosedRegionMask(
		const TArray<FColoredStroke>& Strokes, int32 Width, int32 Height,
		int32 Thickness, TArray<uint8>& OutMask, TArray<uint8>& OutBarrier)
	{
		const int32 N = Width * Height;
		TArray<uint8> Barrier;
		Barrier.Init(0, N);
		const int32 Rad = FMath::Max(1, Thickness / 2);

		auto Stamp = [&](int32 x, int32 y)
		{
			for (int32 oy = -Rad; oy <= Rad; ++oy)
			{
				for (int32 ox = -Rad; ox <= Rad; ++ox)
				{
					const int32 xx = x + ox;
					const int32 yy = y + oy;
					if (xx >= 0 && xx < Width && yy >= 0 && yy < Height)
					{
						Barrier[yy * Width + xx] = 255;
					}
				}
			}
		};

		TArray<FIntPoint> LineBuf;
		auto DrawSeg = [&](const FVector2D& A, const FVector2D& B)
		{
			SkelGraph::LinePoints(
				FIntPoint(FMath::RoundToInt(A.X), FMath::RoundToInt(A.Y)),
				FIntPoint(FMath::RoundToInt(B.X), FMath::RoundToInt(B.Y)), LineBuf);
			for (const FIntPoint& P : LineBuf)
			{
				Stamp(P.X, P.Y);
			}
		};

		// Rasterize all strokes.
		for (const FColoredStroke& CS : Strokes)
		{
			for (int32 k = 0; k + 1 < CS.Points.Num(); ++k)
			{
				DrawSeg(CS.Points[k], CS.Points[k + 1]);
			}
		}

		// Connect each stroke endpoint to its nearest other endpoint to close gaps.
		TArray<FVector2D> Endpoints;
		for (const FColoredStroke& CS : Strokes)
		{
			if (CS.Points.Num() > 0)
			{
				Endpoints.Add(CS.Points[0]);
				Endpoints.Add(CS.Points.Last());
			}
		}
		for (int32 i = 0; i < Endpoints.Num(); ++i)
		{
			int32 Best = -1;
			double BestD2 = TNumericLimits<double>::Max();
			for (int32 j = 0; j < Endpoints.Num(); ++j)
			{
				if (j == i)
				{
					continue;
				}
				const double D2 = FVector2D::DistSquared(Endpoints[i], Endpoints[j]);
				if (D2 < BestD2)
				{
					BestD2 = D2;
					Best = j;
				}
			}
			if (Best >= 0)
			{
				DrawSeg(Endpoints[i], Endpoints[Best]);
			}
		}

		// Flood the background (barrier == 0) from all border pixels.
		TArray<uint8> Reachable;
		Reachable.Init(0, N);
		TArray<int32> Stack;
		Stack.Reserve(1024);
		auto TryPush = [&](int32 x, int32 y)
		{
			if (x < 0 || x >= Width || y < 0 || y >= Height)
			{
				return;
			}
			const int32 Idx = y * Width + x;
			if (Barrier[Idx] == 0 && Reachable[Idx] == 0)
			{
				Reachable[Idx] = 1;
				Stack.Push(Idx);
			}
		};
		for (int32 x = 0; x < Width; ++x)
		{
			TryPush(x, 0);
			TryPush(x, Height - 1);
		}
		for (int32 y = 0; y < Height; ++y)
		{
			TryPush(0, y);
			TryPush(Width - 1, y);
		}
		while (Stack.Num() > 0)
		{
			const int32 P = Stack.Pop(EAllowShrinking::No);
			const int32 px = P % Width;
			const int32 py = P / Width;
			TryPush(px + 1, py);
			TryPush(px - 1, py);
			TryPush(px, py + 1);
			TryPush(px, py - 1);
		}

		// Interior = background not reached from the borders.
		OutMask.Init(0, N);
		for (int32 i = 0; i < N; ++i)
		{
			if (Barrier[i] == 0 && Reachable[i] == 0)
			{
				OutMask[i] = 255;
			}
		}
		OutBarrier = MoveTemp(Barrier);
	}

	// ====================================================================
	// Step 9: red cap-loop detection + longest-green side + translate-copy.
	// ====================================================================
	namespace CapOps
	{
		// Endpoint graph. Real gaps are bridged by explicit connector strokes first;
		// this snap tolerance is only for nearly coincident endpoints.
		struct FGraph
		{
			TArray<int32> NodeU;   // per edge: start node
			TArray<int32> NodeV;   // per edge: end node
			TArray<int32> StrokeId;// per edge: index into the source stroke array
			TArray<uint8> bBlack;  // per edge: 1 if black, 0 if red
			TArray<uint8> bSynthetic; // per edge: 1 if generated connector stroke
			TArray<FVector2D> NodePos; // representative position per node
			int32 NumNodes = 0;
		};

		int32 Find(TArray<int32>& Parent, int32 X)
		{
			while (Parent[X] != X)
			{
				Parent[X] = Parent[Parent[X]];
				X = Parent[X];
			}
			return X;
		}

		// Build a graph over the given edge strokes (by index into Strokes), snapping
		// only nearly coincident endpoints. Wider red/black gaps are represented by
		// explicit connector strokes before this graph is built.
		void BuildGraph(const TArray<FColoredStroke>& Strokes, const TArray<int32>& EdgeStrokes, float NodeSnapTol, int32 FirstSyntheticStrokeId, FGraph& G)
		{
			const int32 E = EdgeStrokes.Num();
			TArray<FVector2D> Pts; // 2 endpoints per edge
			Pts.Reserve(E * 2);
			for (int32 s : EdgeStrokes)
			{
				const FStroke& P = Strokes[s].Points;
				Pts.Add(P.Num() > 0 ? P[0] : FVector2D::ZeroVector);
				Pts.Add(P.Num() > 0 ? P.Last() : FVector2D::ZeroVector);
			}

			const int32 NP = Pts.Num();
			TArray<int32> Parent;
			Parent.SetNumUninitialized(NP);
			for (int32 i = 0; i < NP; ++i)
			{
				Parent[i] = i;
			}
			const double Tol2 = double(NodeSnapTol) * double(NodeSnapTol);
			for (int32 i = 0; i < NP; ++i)
			{
				for (int32 j = i + 1; j < NP; ++j)
				{
					if (FVector2D::DistSquared(Pts[i], Pts[j]) <= Tol2)
					{
						Parent[Find(Parent, j)] = Find(Parent, i);
					}
				}
			}

			// Compact node ids.
			TMap<int32, int32> Remap;
			TArray<FVector2D> Accum;
			TArray<int32> AccumN;
			auto NodeOf = [&](int32 PtIdx) -> int32
			{
				const int32 Root = Find(Parent, PtIdx);
				if (int32* Found = Remap.Find(Root))
				{
					Accum[*Found] += Pts[PtIdx];
					AccumN[*Found] += 1;
					return *Found;
				}
				const int32 NewId = Accum.Num();
				Remap.Add(Root, NewId);
				Accum.Add(Pts[PtIdx]);
				AccumN.Add(1);
				return NewId;
			};

			G = FGraph();
			for (int32 e = 0; e < E; ++e)
			{
				const int32 U = NodeOf(2 * e);
				const int32 V = NodeOf(2 * e + 1);
				G.NodeU.Add(U);
				G.NodeV.Add(V);
				G.StrokeId.Add(EdgeStrokes[e]);
				G.bBlack.Add(Strokes[EdgeStrokes[e]].Color == EStrokeColor::Black ? 1 : 0);
				G.bSynthetic.Add(EdgeStrokes[e] >= FirstSyntheticStrokeId ? 1 : 0);
			}
			G.NumNodes = Accum.Num();
			G.NodePos.SetNumUninitialized(G.NumNodes);
			for (int32 n = 0; n < G.NumNodes; ++n)
			{
				G.NodePos[n] = Accum[n] / double(FMath::Max(1, AccumN[n]));
			}
		}

		// BFS shortest edge-path from Src to Dst over the subset of edges flagged usable,
		// excluding edge index ExcludeEdge. EdgeUsable[e]!=0 means the edge may be traversed.
		// Returns the ordered edge indices (into G arrays).
		bool BfsPath(const FGraph& G, int32 Src, int32 Dst, int32 ExcludeEdge, const TArray<uint8>& EdgeUsable, TArray<int32>& OutEdgePath)
		{
			OutEdgePath.Reset();
			if (Src == Dst)
			{
				return true; // empty path
			}
			TArray<int32> ParentNode, ParentEdge;
			ParentNode.Init(-2, G.NumNodes);
			ParentEdge.Init(-1, G.NumNodes);
			TArray<int32> Q;
			Q.Add(Src);
			ParentNode[Src] = -1;
			int32 Head = 0;
			bool bFound = false;
			while (Head < Q.Num() && !bFound)
			{
				const int32 Cur = Q[Head++];
				for (int32 e = 0; e < G.StrokeId.Num(); ++e)
				{
					if (e == ExcludeEdge)
					{
						continue;
					}
					if (!EdgeUsable[e])
					{
						continue;
					}
					int32 Other = -1;
					if (G.NodeU[e] == Cur) { Other = G.NodeV[e]; }
					else if (G.NodeV[e] == Cur) { Other = G.NodeU[e]; }
					else { continue; }
					if (ParentNode[Other] != -2)
					{
						continue;
					}
					ParentNode[Other] = Cur;
					ParentEdge[Other] = e;
					if (Other == Dst)
					{
						bFound = true;
						break;
					}
					Q.Add(Other);
				}
			}
			if (!bFound)
			{
				return false;
			}
			for (int32 N = Dst; N != Src; N = ParentNode[N])
			{
				OutEdgePath.Add(ParentEdge[N]);
			}
			Algo::Reverse(OutEdgePath);
			return true;
		}

		// Find the smallest loop anchored on a red edge, traversing only EdgeUsable edges.
		// Returns ordered edge indices (cycle).
		bool FindRedCycle(const FGraph& G, const TArray<uint8>& bRedEdge, const TArray<uint8>& EdgeUsable, TArray<int32>& OutCycle)
		{
			OutCycle.Reset();
			int32 BestLen = MAX_int32;
			for (int32 e = 0; e < G.StrokeId.Num(); ++e)
			{
				if (!bRedEdge[e])
				{
					continue; // cycle must include a red edge; anchor on red
				}
				const int32 U = G.NodeU[e];
				const int32 V = G.NodeV[e];
				if (U == V)
				{
					// A single self-closed red stroke is NOT treated as a cap loop;
					// the loop must connect at least two strokes end-to-end.
					continue;
				}
				TArray<int32> Path;
				if (BfsPath(G, U, V, /*ExcludeEdge*/ e, EdgeUsable, Path))
				{
					const int32 Len = Path.Num() + 1;
					if (Len < BestLen)
					{
						BestLen = Len;
						OutCycle = Path;
						OutCycle.Add(e);
					}
				}
			}
			return OutCycle.Num() > 0;
		}

		// Walk an unordered cycle (edge indices) into an ordered node + point sequence.
		void BuildPolygon(const FColoredStroke* StrokesData, const FGraph& G, const TArray<int32>& Cycle,
			FStroke& OutPolygon, TArray<FVector2D>& OutNodes)
		{
			OutPolygon.Reset();
			OutNodes.Reset();
			if (Cycle.Num() == 0)
			{
				return;
			}

			// Order edges head-to-tail starting from one endpoint of the first edge.
			TArray<int32> Remaining = Cycle;
			int32 CurNode = G.NodeU[Cycle[0]];
			const int32 StartNode = CurNode;

			while (Remaining.Num() > 0)
			{
				int32 PickPos = -1;
				bool bForward = true;
				for (int32 r = 0; r < Remaining.Num(); ++r)
				{
					const int32 e = Remaining[r];
					if (G.NodeU[e] == CurNode) { PickPos = r; bForward = true; break; }
					if (G.NodeV[e] == CurNode) { PickPos = r; bForward = false; break; }
				}
				if (PickPos < 0)
				{
					break; // disconnected (shouldn't happen for a valid cycle)
				}
				const int32 e = Remaining[PickPos];
				Remaining.RemoveAt(PickPos);

				const FStroke& SP = StrokesData[G.StrokeId[e]].Points;
				OutNodes.Add(G.NodePos[CurNode]);

				if (bForward)
				{
					for (int32 k = 0; k < SP.Num(); ++k)
					{
						if (OutPolygon.Num() == 0 || (OutPolygon.Last() - SP[k]).SizeSquared() > 1e-6)
						{
							OutPolygon.Add(SP[k]);
						}
					}
					CurNode = G.NodeV[e];
				}
				else
				{
					for (int32 k = SP.Num() - 1; k >= 0; --k)
					{
						if (OutPolygon.Num() == 0 || (OutPolygon.Last() - SP[k]).SizeSquared() > 1e-6)
						{
							OutPolygon.Add(SP[k]);
						}
					}
					CurNode = G.NodeU[e];
				}
			}

			// Close the polygon back to the start node.
			if (OutPolygon.Num() > 0)
			{
				OutPolygon.Add(G.NodePos[StartNode]);
			}
		}

		// Render a graph: real red/black edges keep their source colors; synthetic
		// connector edges are orange (red connector) or gray (black connector).
		// Only edges with Kept[e]!=0 are drawn (pass an all-ones array to draw everything).
		bool SaveGraphPng(const TArray<FColoredStroke>& Strokes, const FGraph& G, const TArray<uint8>& Kept,
			int32 Width, int32 Height, const FString& Path)
		{
			TArray<uint8> RGBA;
			RGBA.Init(255, Width * Height * 4);
			auto Plot = [&](int32 x, int32 y, uint8 r, uint8 g, uint8 b)
			{
				if (x < 0 || x >= Width || y < 0 || y >= Height) { return; }
				const int32 Off = (y * Width + x) * 4;
				RGBA[Off + 0] = r; RGBA[Off + 1] = g; RGBA[Off + 2] = b; RGBA[Off + 3] = 255;
			};
			TArray<FIntPoint> LineBuf;
			for (int32 e = 0; e < G.StrokeId.Num(); ++e)
			{
				if (!Kept.IsValidIndex(e) || !Kept[e]) { continue; }
				const bool bSynthetic = G.bSynthetic.IsValidIndex(e) && G.bSynthetic[e] != 0;
				const uint8 r = bSynthetic ? (G.bBlack[e] ? 120 : 255) : (G.bBlack[e] ? 0 : 220);
				const uint8 g = bSynthetic ? (G.bBlack[e] ? 120 : 140) : (G.bBlack[e] ? 0 : 20);
				const uint8 b = bSynthetic ? (G.bBlack[e] ? 120 : 0) : (G.bBlack[e] ? 0 : 60);
				const FStroke& SP = Strokes[G.StrokeId[e]].Points;
				for (int32 k = 0; k + 1 < SP.Num(); ++k)
				{
					SkelGraph::LinePoints(
						FIntPoint(FMath::RoundToInt(SP[k].X), FMath::RoundToInt(SP[k].Y)),
						FIntPoint(FMath::RoundToInt(SP[k + 1].X), FMath::RoundToInt(SP[k + 1].Y)), LineBuf);
					for (const FIntPoint& P : LineBuf) { Plot(P.X, P.Y, r, g, b); }
				}
			}
			for (int32 n = 0; n < G.NumNodes; ++n)
			{
				const int32 nx = FMath::RoundToInt(G.NodePos[n].X);
				const int32 ny = FMath::RoundToInt(G.NodePos[n].Y);
				for (int32 oy = -2; oy <= 2; ++oy)
				{
					for (int32 ox = -2; ox <= 2; ++ox)
					{
						Plot(nx + ox, ny + oy, 30, 80, 220);
					}
				}
			}
			IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
			TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
			if (!IW.IsValid()) { return false; }
			IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
			const TArray64<uint8>& C = IW->GetCompressed();
			return FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *Path);
		}

		bool SaveGraphJson(const TArray<FColoredStroke>& Strokes, const FGraph& G, const TArray<uint8>& Kept, const FString& Path)
		{
			FString Json;
			Json += TEXT("{\n  \"nodes\": [\n");
			for (int32 n = 0; n < G.NumNodes; ++n)
			{
				Json += FString::Printf(TEXT("    {\"id\": %d, \"pos\": [%.2f, %.2f]}%s\n"),
					n, G.NodePos[n].X, G.NodePos[n].Y, (n + 1 < G.NumNodes ? TEXT(",") : TEXT("")));
			}
			Json += TEXT("  ],\n  \"edges\": [\n");
			const int32 E = G.StrokeId.Num();
			for (int32 e = 0; e < E; ++e)
			{
				const bool bKept = Kept.IsValidIndex(e) && Kept[e] != 0;
				const bool bSynthetic = G.bSynthetic.IsValidIndex(e) && G.bSynthetic[e] != 0;
				Json += FString::Printf(TEXT("    {\"stroke_id\": %d, \"u\": %d, \"v\": %d, \"color\": \"%s\", \"synthetic\": %s, \"kept\": %s}%s\n"),
					G.StrokeId[e], G.NodeU[e], G.NodeV[e],
					G.bBlack[e] ? TEXT("black") : TEXT("red"),
					bSynthetic ? TEXT("true") : TEXT("false"),
					bKept ? TEXT("true") : TEXT("false"),
					(e + 1 < E ? TEXT(",") : TEXT("")));
			}
			Json += TEXT("  ]\n}\n");
			return FFileHelper::SaveStringToFile(Json, *Path);
		}
	} // namespace CapOps

	// After real intersections and synthetic endpoint connectors are added, merge
	// the blue graph endpoints in 09_all_red_black_graph.png within this radius.
	static constexpr float CapLoopGraphNodeSnapTol = 5.0f;
	static constexpr double CapGeometryEpsilon = 1e-6;
	static constexpr double CapEndpointDirectionArc = 5.0;
	static constexpr double CapBorrowedLoopMinBboxArea = 1500.0;

	struct FCapStrokeSplit
	{
		int32 SegmentIndex = -1;
		double T = 0.0;
		FVector2D Pos = FVector2D::ZeroVector;
	};

	struct FCapSegmentHit
	{
		double T = 0.0;
		double U = 0.0;
		FVector2D Pos = FVector2D::ZeroVector;
	};

	static double Cross2D(const FVector2D& A, const FVector2D& B)
	{
		return A.X * B.Y - A.Y * B.X;
	}

	static void AddSegmentHit(
		TArray<FCapSegmentHit>& OutHits,
		double T,
		double U,
		const FVector2D& Pos)
	{
		for (const FCapSegmentHit& Existing : OutHits)
		{
			if (FVector2D::DistSquared(Existing.Pos, Pos) <= CapGeometryEpsilon * CapGeometryEpsilon)
			{
				return;
			}
		}

		FCapSegmentHit Hit;
		Hit.T = FMath::Clamp(T, 0.0, 1.0);
		Hit.U = FMath::Clamp(U, 0.0, 1.0);
		Hit.Pos = Pos;
		OutHits.Add(Hit);
	}

	static void FindSegmentIntersections(
		const FVector2D& A,
		const FVector2D& B,
		const FVector2D& C,
		const FVector2D& D,
		TArray<FCapSegmentHit>& OutHits)
	{
		OutHits.Reset();
		const FVector2D R = B - A;
		const FVector2D S = D - C;
		const double RLen2 = R.SizeSquared();
		const double SLen2 = S.SizeSquared();
		if (RLen2 <= CapGeometryEpsilon || SLen2 <= CapGeometryEpsilon)
		{
			return;
		}

		const FVector2D CA = C - A;
		const double RCrossS = Cross2D(R, S);
		const double CACrossR = Cross2D(CA, R);
		if (FMath::Abs(RCrossS) > CapGeometryEpsilon)
		{
			const double T = Cross2D(CA, S) / RCrossS;
			const double U = CACrossR / RCrossS;
			if (T >= -CapGeometryEpsilon && T <= 1.0 + CapGeometryEpsilon &&
				U >= -CapGeometryEpsilon && U <= 1.0 + CapGeometryEpsilon)
			{
				const double ClampedT = FMath::Clamp(T, 0.0, 1.0);
				AddSegmentHit(OutHits, ClampedT, U, A + R * ClampedT);
			}
			return;
		}

		if (FMath::Abs(CACrossR) > CapGeometryEpsilon)
		{
			return;
		}

		// Collinear overlap: its boundaries are sufficient to planarize both polylines.
		const double T0 = FVector2D::DotProduct(C - A, R) / RLen2;
		const double T1 = FVector2D::DotProduct(D - A, R) / RLen2;
		const double Lo = FMath::Max(0.0, FMath::Min(T0, T1));
		const double Hi = FMath::Min(1.0, FMath::Max(T0, T1));
		if (Lo > Hi + CapGeometryEpsilon)
		{
			return;
		}

		auto AddCollinearBoundary = [&](double T)
		{
			const FVector2D Pos = A + R * T;
			const double U = FVector2D::DotProduct(Pos - C, S) / SLen2;
			AddSegmentHit(OutHits, T, U, Pos);
		};
		AddCollinearBoundary(Lo);
		if (Hi - Lo > CapGeometryEpsilon)
		{
			AddCollinearBoundary(Hi);
		}
	}

	static void AddStrokeSplit(
		TArray<TArray<FCapStrokeSplit>>& SplitsByStroke,
		int32 StrokeId,
		int32 SegmentIndex,
		double T,
		const FVector2D& Pos)
	{
		if (!SplitsByStroke.IsValidIndex(StrokeId))
		{
			return;
		}
		TArray<FCapStrokeSplit>& Splits = SplitsByStroke[StrokeId];
		for (const FCapStrokeSplit& Existing : Splits)
		{
			if (FVector2D::DistSquared(Existing.Pos, Pos) <= CapGeometryEpsilon * CapGeometryEpsilon)
			{
				return;
			}
		}

		FCapStrokeSplit Split;
		Split.SegmentIndex = SegmentIndex;
		Split.T = FMath::Clamp(T, 0.0, 1.0);
		Split.Pos = Pos;
		Splits.Add(Split);
	}

	static void AppendPlanarStrokeFragments(
		const FColoredStroke& Source,
		const TArray<FCapStrokeSplit>& SourceSplits,
		TArray<FColoredStroke>& OutStrokes)
	{
		if (Source.Points.Num() < 2 || SourceSplits.Num() == 0)
		{
			OutStrokes.Add(Source);
			return;
		}

		TArray<FCapStrokeSplit> Splits = SourceSplits;
		Splits.Sort([](const FCapStrokeSplit& A, const FCapStrokeSplit& B)
		{
			const double AKey = double(A.SegmentIndex) + A.T;
			const double BKey = double(B.SegmentIndex) + B.T;
			return AKey < BKey;
		});

		TArray<TArray<FCapStrokeSplit>> SegmentSplits;
		SegmentSplits.SetNum(Source.Points.Num() - 1);
		for (const FCapStrokeSplit& Split : Splits)
		{
			if (SegmentSplits.IsValidIndex(Split.SegmentIndex))
			{
				SegmentSplits[Split.SegmentIndex].Add(Split);
			}
		}

		auto AppendFragment = [&](const FStroke& Points)
		{
			if (Points.Num() < 2)
			{
				return;
			}
			double Arc = 0.0;
			for (int32 i = 1; i < Points.Num(); ++i)
			{
				Arc += (Points[i] - Points[i - 1]).Size();
			}
			if (Arc <= CapGeometryEpsilon)
			{
				return;
			}

			FColoredStroke Fragment = Source;
			Fragment.Points = Points;
			Fragment.ConnectionPointCount = 0;
			Fragment.bHasMetrics = false;
			OutStrokes.Add(MoveTemp(Fragment));
		};

		FStroke Current;
		Current.Add(Source.Points[0]);
		for (int32 SegmentIndex = 0; SegmentIndex + 1 < Source.Points.Num(); ++SegmentIndex)
		{
			for (const FCapStrokeSplit& Split : SegmentSplits[SegmentIndex])
			{
				if (FVector2D::DistSquared(Current.Last(), Split.Pos) > CapGeometryEpsilon * CapGeometryEpsilon)
				{
					Current.Add(Split.Pos);
				}
				AppendFragment(Current);
				Current.Reset();
				Current.Add(Split.Pos);
			}

			const FVector2D& SegmentEnd = Source.Points[SegmentIndex + 1];
			if (FVector2D::DistSquared(Current.Last(), SegmentEnd) > CapGeometryEpsilon * CapGeometryEpsilon)
			{
				Current.Add(SegmentEnd);
			}
		}
		AppendFragment(Current);
	}

	static FVector2D GetEndpointOutwardDirection(const FStroke& Points, int32 EndpointIndex)
	{
		if (Points.Num() < 2)
		{
			return FVector2D::ZeroVector;
		}

		const bool bStart = EndpointIndex == 0;
		const int32 EndpointPointIndex = bStart ? 0 : Points.Num() - 1;
		const int32 Step = bStart ? 1 : -1;
		FVector2D Interior = Points[EndpointPointIndex];
		double Arc = 0.0;
		for (int32 i = EndpointPointIndex + Step; i >= 0 && i < Points.Num(); i += Step)
		{
			Arc += (Points[i] - Interior).Size();
			Interior = Points[i];
			if (Arc >= CapEndpointDirectionArc)
			{
				break;
			}
		}
		return (Points[EndpointPointIndex] - Interior).GetSafeNormal();
	}

	struct FCapRedDeadEnd
	{
		FVector2D Pos = FVector2D::ZeroVector;
		FVector2D Forward = FVector2D::ZeroVector;
		int32 SourceStrokeId = INDEX_NONE;
		int32 TopologyStrokeId = INDEX_NONE;
		int32 EndpointIndex = 0;
		bool bSynthetic = false;
	};

	struct FCapRedTopologyNode
	{
		FVector2D Pos = FVector2D::ZeroVector;
		int32 Degree = 0;
		int32 IncidentStrokeId = INDEX_NONE;
		int32 IncidentEndpointIndex = 0;
	};

	static void CollectPlanarRedDeadEnds(
		const TArray<FColoredStroke>& SourceStrokes,
		const TArray<int32>& RedIdx,
		const TArray<TArray<FCapStrokeSplit>>& SplitsByStroke,
		const TArray<FColoredStroke>* ExtraRedStrokes,
		TArray<FCapRedDeadEnd>& OutDeadEnds)
	{
		OutDeadEnds.Reset();

		TArray<FColoredStroke> PlanarRedStrokes;
		TArray<int32> PlanarRedSourceIds;
		TArray<uint8> PlanarRedSynthetic;
		const TArray<FCapStrokeSplit> EmptySplits;

		auto AppendRedTopologyStroke = [&](const FColoredStroke& Stroke, const TArray<FCapStrokeSplit>& Splits, int32 SourceStrokeId, bool bSynthetic)
		{
			const int32 Before = PlanarRedStrokes.Num();
			AppendPlanarStrokeFragments(Stroke, Splits, PlanarRedStrokes);
			for (int32 i = Before; i < PlanarRedStrokes.Num(); ++i)
			{
				PlanarRedSourceIds.Add(SourceStrokeId);
				PlanarRedSynthetic.Add(bSynthetic ? 1 : 0);
			}
		};

		for (int32 RedStrokeId : RedIdx)
		{
			if (SourceStrokes.IsValidIndex(RedStrokeId) && SplitsByStroke.IsValidIndex(RedStrokeId))
			{
				AppendRedTopologyStroke(SourceStrokes[RedStrokeId], SplitsByStroke[RedStrokeId], RedStrokeId, false);
			}
		}
		if (ExtraRedStrokes)
		{
			for (const FColoredStroke& Extra : *ExtraRedStrokes)
			{
				if (Extra.Color == EStrokeColor::Red)
				{
					AppendRedTopologyStroke(Extra, EmptySplits, INDEX_NONE, true);
				}
			}
		}

		TArray<FCapRedTopologyNode> Nodes;
		const double TopologyTol2 = CapGeometryEpsilon * CapGeometryEpsilon;
		for (int32 StrokeId = 0; StrokeId < PlanarRedStrokes.Num(); ++StrokeId)
		{
			const FStroke& Points = PlanarRedStrokes[StrokeId].Points;
			if (Points.Num() < 2)
			{
				continue;
			}

			for (int32 EndpointIndex = 0; EndpointIndex < 2; ++EndpointIndex)
			{
				const FVector2D Endpoint = EndpointIndex == 0 ? Points[0] : Points.Last();
				int32 NodeId = INDEX_NONE;
				for (int32 CandidateNodeId = 0; CandidateNodeId < Nodes.Num(); ++CandidateNodeId)
				{
					if (FVector2D::DistSquared(Endpoint, Nodes[CandidateNodeId].Pos) <= TopologyTol2)
					{
						NodeId = CandidateNodeId;
						break;
					}
				}

				if (NodeId == INDEX_NONE)
				{
					NodeId = Nodes.AddDefaulted();
					Nodes[NodeId].Pos = Endpoint;
				}

				FCapRedTopologyNode& Node = Nodes[NodeId];
				++Node.Degree;
				Node.IncidentStrokeId = StrokeId;
				Node.IncidentEndpointIndex = EndpointIndex;
			}
		}

		for (const FCapRedTopologyNode& Node : Nodes)
		{
			if (Node.Degree != 1 || !PlanarRedStrokes.IsValidIndex(Node.IncidentStrokeId))
			{
				continue;
			}

			const FVector2D Forward = GetEndpointOutwardDirection(
				PlanarRedStrokes[Node.IncidentStrokeId].Points,
				Node.IncidentEndpointIndex);
			if (Forward.IsNearlyZero())
			{
				continue;
			}

			FCapRedDeadEnd DeadEnd;
			DeadEnd.Pos = Node.Pos;
			DeadEnd.Forward = Forward;
			DeadEnd.SourceStrokeId = PlanarRedSourceIds.IsValidIndex(Node.IncidentStrokeId) ? PlanarRedSourceIds[Node.IncidentStrokeId] : INDEX_NONE;
			DeadEnd.TopologyStrokeId = Node.IncidentStrokeId;
			DeadEnd.EndpointIndex = Node.IncidentEndpointIndex;
			DeadEnd.bSynthetic = PlanarRedSynthetic.IsValidIndex(Node.IncidentStrokeId) && PlanarRedSynthetic[Node.IncidentStrokeId] != 0;
			OutDeadEnds.Add(DeadEnd);
		}
	}

	static bool ClosestPointOnSegmentInForwardHalfPlane(
		const FVector2D& Endpoint,
		const FVector2D& Forward,
		const FVector2D& A,
		const FVector2D& B,
		double& OutT,
		FVector2D& OutPoint)
	{
		const FVector2D Segment = B - A;
		const double SegmentLen2 = Segment.SizeSquared();
		if (SegmentLen2 <= CapGeometryEpsilon)
		{
			return false;
		}

		const double DA = FVector2D::DotProduct(A - Endpoint, Forward);
		const double DB = FVector2D::DotProduct(B - Endpoint, Forward);
		if (DA < -CapGeometryEpsilon && DB < -CapGeometryEpsilon)
		{
			return false;
		}

		double MinT = 0.0;
		double MaxT = 1.0;
		if (DA < -CapGeometryEpsilon)
		{
			MinT = FMath::Clamp(-DA / (DB - DA), 0.0, 1.0);
		}
		else if (DB < -CapGeometryEpsilon)
		{
			MaxT = FMath::Clamp(-DA / (DB - DA), 0.0, 1.0);
		}
		if (MinT > MaxT + CapGeometryEpsilon)
		{
			return false;
		}

		const double ProjectionT = FVector2D::DotProduct(Endpoint - A, Segment) / SegmentLen2;
		OutT = FMath::Clamp(ProjectionT, MinT, MaxT);
		OutPoint = A + Segment * OutT;
		return FVector2D::DotProduct(OutPoint - Endpoint, Forward) >= -CapGeometryEpsilon;
	}

	static FString RoundedConnectorKey(const FVector2D& A, const FVector2D& B)
	{
		const int32 Ax = FMath::RoundToInt(A.X);
		const int32 Ay = FMath::RoundToInt(A.Y);
		const int32 Bx = FMath::RoundToInt(B.X);
		const int32 By = FMath::RoundToInt(B.Y);
		return (Ax < Bx || (Ax == Bx && Ay <= By))
			? FString::Printf(TEXT("%d,%d-%d,%d"), Ax, Ay, Bx, By)
			: FString::Printf(TEXT("%d,%d-%d,%d"), Bx, By, Ax, Ay);
	}

	static int32 AddRedForwardConnectorsToRedTargets(
		const TArray<FColoredStroke>& SourceStrokes,
		const TArray<int32>& RedIdx,
		const TArray<FCapRedDeadEnd>& RedDeadEnds,
		float ConnectorTol,
		TArray<TArray<FCapStrokeSplit>>& SplitsByStroke,
		TArray<FColoredStroke>& SyntheticConnectors)
	{
		const double ConnectorTol2 = double(ConnectorTol) * double(ConnectorTol);
		TSet<FString> ConnectorKeys;
		for (const FColoredStroke& Existing : SyntheticConnectors)
		{
			if (Existing.Color == EStrokeColor::Red && Existing.Points.Num() >= 2)
			{
				ConnectorKeys.Add(RoundedConnectorKey(Existing.Points[0], Existing.Points.Last()));
			}
		}

		int32 AddedCount = 0;
		for (const FCapRedDeadEnd& DeadEnd : RedDeadEnds)
		{
			const FVector2D Endpoint = DeadEnd.Pos;
			int32 BestRedStrokeId = INDEX_NONE;
			int32 BestRedSegment = INDEX_NONE;
			double BestRedT = 0.0;
			double BestD2 = TNumericLimits<double>::Max();
			FVector2D BestPoint = FVector2D::ZeroVector;

			for (int32 RedStrokeId : RedIdx)
			{
				if (!SourceStrokes.IsValidIndex(RedStrokeId) || RedStrokeId == DeadEnd.SourceStrokeId)
				{
					continue;
				}
				const FStroke& RedPoints = SourceStrokes[RedStrokeId].Points;
				for (int32 SegmentIndex = 0; SegmentIndex + 1 < RedPoints.Num(); ++SegmentIndex)
				{
					double SegmentT = 0.0;
					FVector2D CandidatePoint;
					if (!ClosestPointOnSegmentInForwardHalfPlane(
						Endpoint, DeadEnd.Forward,
						RedPoints[SegmentIndex], RedPoints[SegmentIndex + 1],
						SegmentT, CandidatePoint))
					{
						continue;
					}

					const double D2 = FVector2D::DistSquared(Endpoint, CandidatePoint);
					if (D2 <= CapGeometryEpsilon * CapGeometryEpsilon || D2 > ConnectorTol2 || D2 >= BestD2)
					{
						continue;
					}

					BestRedStrokeId = RedStrokeId;
					BestRedSegment = SegmentIndex;
					BestRedT = SegmentT;
					BestD2 = D2;
					BestPoint = CandidatePoint;
				}
			}

			if (BestRedStrokeId == INDEX_NONE)
			{
				continue;
			}

			const FString Key = RoundedConnectorKey(Endpoint, BestPoint);
			if (ConnectorKeys.Contains(Key))
			{
				continue;
			}
			ConnectorKeys.Add(Key);

			AddStrokeSplit(
				SplitsByStroke,
				BestRedStrokeId,
				BestRedSegment,
				BestRedT,
				BestPoint);

			FColoredStroke Connector;
			Connector.Points.Add(Endpoint);
			Connector.Points.Add(BestPoint);
			Connector.Color = EStrokeColor::Red;
			Connector.ConnectionPointCount = Connector.Points.Num();
			SyntheticConnectors.Add(MoveTemp(Connector));
			++AddedCount;
		}

		return AddedCount;
	}

	static bool EndpointTouchesAnyBlackStroke(
		const FVector2D& Endpoint,
		const TArray<FColoredStroke>& SourceStrokes,
		const TArray<int32>& BlackIdx)
	{
		for (int32 BlackStrokeId : BlackIdx)
		{
			if (!SourceStrokes.IsValidIndex(BlackStrokeId))
			{
				continue;
			}
			const FStroke& Points = SourceStrokes[BlackStrokeId].Points;
			for (int32 SegmentIndex = 0; SegmentIndex + 1 < Points.Num(); ++SegmentIndex)
			{
				const FVector2D Segment = Points[SegmentIndex + 1] - Points[SegmentIndex];
				const double SegmentLen2 = Segment.SizeSquared();
				if (SegmentLen2 <= CapGeometryEpsilon)
				{
					continue;
				}
				const double T = FMath::Clamp(
					FVector2D::DotProduct(Endpoint - Points[SegmentIndex], Segment) / SegmentLen2,
					0.0, 1.0);
				const FVector2D Closest = Points[SegmentIndex] + Segment * T;
				if (FVector2D::DistSquared(Endpoint, Closest) <= CapGeometryEpsilon * CapGeometryEpsilon)
				{
					return true;
				}
			}
		}
		return false;
	}

	static void BuildPlanarEndpointConnectorStrokes(
		const TArray<FColoredStroke>& SourceStrokes,
		const TArray<int32>& RedIdx,
		const TArray<int32>& BlackIdx,
		float ConnectorTol,
		TArray<FColoredStroke>& OutStrokes,
		int32& OutFirstSyntheticStrokeId)
	{
		TArray<TArray<FCapStrokeSplit>> SplitsByStroke;
		SplitsByStroke.SetNum(SourceStrokes.Num());
		int32 TrueIntersectionCount = 0;

		// First planarize every real red/black polyline intersection.
		TArray<FCapSegmentHit> Hits;
		for (int32 RedStrokeId : RedIdx)
		{
			if (!SourceStrokes.IsValidIndex(RedStrokeId))
			{
				continue;
			}
			const FStroke& RedPoints = SourceStrokes[RedStrokeId].Points;
			for (int32 BlackStrokeId : BlackIdx)
			{
				if (!SourceStrokes.IsValidIndex(BlackStrokeId))
				{
					continue;
				}
				const FStroke& BlackPoints = SourceStrokes[BlackStrokeId].Points;
				for (int32 RedSegment = 0; RedSegment + 1 < RedPoints.Num(); ++RedSegment)
				{
					for (int32 BlackSegment = 0; BlackSegment + 1 < BlackPoints.Num(); ++BlackSegment)
					{
						FindSegmentIntersections(
							RedPoints[RedSegment], RedPoints[RedSegment + 1],
							BlackPoints[BlackSegment], BlackPoints[BlackSegment + 1],
							Hits);
						for (const FCapSegmentHit& Hit : Hits)
						{
							const int32 BeforeRed = SplitsByStroke[RedStrokeId].Num();
							AddStrokeSplit(SplitsByStroke, RedStrokeId, RedSegment, Hit.T, Hit.Pos);
							AddStrokeSplit(SplitsByStroke, BlackStrokeId, BlackSegment, Hit.U, Hit.Pos);
							if (SplitsByStroke[RedStrokeId].Num() > BeforeRed)
							{
								++TrueIntersectionCount;
							}
						}
					}
				}
			}
		}

		// Build the red topology after real intersection splitting. Only nodes with one
		// incident red fragment are dead ends; shared red endpoints, loops, and branches
		// must not initiate connectors. This uses only geometric equality, not the final
		// 5px graph-node snap.
		TArray<FColoredStroke> SyntheticConnectors;
		TArray<FCapRedDeadEnd> RedDeadEnds;
		CollectPlanarRedDeadEnds(SourceStrokes, RedIdx, SplitsByStroke, nullptr, RedDeadEnds);
		const double ConnectorTol2 = double(ConnectorTol) * double(ConnectorTol);
		const int32 InitialRedDeadEndCount = RedDeadEnds.Num();

		// Prefer repairing red topology with red targets before borrowing black edges.
		const int32 RedToRedConnectorCount = AddRedForwardConnectorsToRedTargets(
			SourceStrokes,
			RedIdx,
			RedDeadEnds,
			ConnectorTol,
			SplitsByStroke,
			SyntheticConnectors);

		TArray<FCapRedDeadEnd> RedDeadEndsAfterRedRepair;
		CollectPlanarRedDeadEnds(SourceStrokes, RedIdx, SplitsByStroke, &SyntheticConnectors, RedDeadEndsAfterRedRepair);

		int32 RedToBlackConnectorCount = 0;
		for (const FCapRedDeadEnd& DeadEnd : RedDeadEndsAfterRedRepair)
		{
			const FVector2D Endpoint = DeadEnd.Pos;
			if (EndpointTouchesAnyBlackStroke(Endpoint, SourceStrokes, BlackIdx))
			{
				continue;
			}

			int32 BestBlackStrokeId = INDEX_NONE;
			int32 BestBlackSegment = INDEX_NONE;
			double BestBlackT = 0.0;
			double BestD2 = TNumericLimits<double>::Max();
			FVector2D BestPoint = FVector2D::ZeroVector;
			for (int32 BlackStrokeId : BlackIdx)
			{
				if (!SourceStrokes.IsValidIndex(BlackStrokeId))
				{
					continue;
				}
				const FStroke& BlackPoints = SourceStrokes[BlackStrokeId].Points;
				for (int32 SegmentIndex = 0; SegmentIndex + 1 < BlackPoints.Num(); ++SegmentIndex)
				{
					double SegmentT = 0.0;
					FVector2D CandidatePoint;
					if (!ClosestPointOnSegmentInForwardHalfPlane(
						Endpoint, DeadEnd.Forward,
						BlackPoints[SegmentIndex], BlackPoints[SegmentIndex + 1],
						SegmentT, CandidatePoint))
					{
						continue;
					}
					const double D2 = FVector2D::DistSquared(Endpoint, CandidatePoint);
					if (D2 <= ConnectorTol2 && D2 < BestD2)
					{
						BestBlackStrokeId = BlackStrokeId;
						BestBlackSegment = SegmentIndex;
						BestBlackT = SegmentT;
						BestD2 = D2;
						BestPoint = CandidatePoint;
					}
				}
			}
			if (BestBlackStrokeId == INDEX_NONE || BestD2 <= CapGeometryEpsilon * CapGeometryEpsilon)
			{
				continue;
			}

			AddStrokeSplit(
				SplitsByStroke,
				BestBlackStrokeId,
				BestBlackSegment,
				BestBlackT,
				BestPoint);

			FColoredStroke Connector;
			Connector.Points.Add(Endpoint);
			Connector.Points.Add(BestPoint);
			Connector.Color = EStrokeColor::Red;
			Connector.ConnectionPointCount = Connector.Points.Num();
			SyntheticConnectors.Add(MoveTemp(Connector));
			++RedToBlackConnectorCount;
		}

		TArray<FColoredStroke> PlanarSourceStrokes;
		TArray<int32> PlanarSourceIds;
		for (int32 StrokeId = 0; StrokeId < SourceStrokes.Num(); ++StrokeId)
		{
			const int32 Before = PlanarSourceStrokes.Num();
			AppendPlanarStrokeFragments(SourceStrokes[StrokeId], SplitsByStroke[StrokeId], PlanarSourceStrokes);
			for (int32 i = Before; i < PlanarSourceStrokes.Num(); ++i)
			{
				PlanarSourceIds.Add(StrokeId);
			}
		}

		struct FEndpointNode
		{
			FVector2D Pos = FVector2D::ZeroVector;
			int32 Degree = 0;
		};
		struct FEndpointRef
		{
			FVector2D Pos = FVector2D::ZeroVector;
			EStrokeColor Color = EStrokeColor::None;
			int32 SourceStrokeId = INDEX_NONE;
			int32 LocalStrokeId = INDEX_NONE;
			int32 EndpointIndex = 0;
			bool bSynthetic = false;
		};
		TArray<FEndpointNode> EndpointNodes;
		TArray<FEndpointRef> EndpointRefs;
		auto AddEndpointRef = [&](const FStroke& Points, EStrokeColor Color, int32 SourceStrokeId, int32 LocalStrokeId, int32 EndpointIndex, bool bSynthetic)
		{
			if (Points.Num() < 2)
			{
				return;
			}
			const FVector2D Pos = EndpointIndex == 0 ? Points[0] : Points.Last();
			int32 NodeId = INDEX_NONE;
			for (int32 CandidateNodeId = 0; CandidateNodeId < EndpointNodes.Num(); ++CandidateNodeId)
			{
				if (FVector2D::DistSquared(Pos, EndpointNodes[CandidateNodeId].Pos) <= CapGeometryEpsilon * CapGeometryEpsilon)
				{
					NodeId = CandidateNodeId;
					break;
				}
			}
			if (NodeId == INDEX_NONE)
			{
				NodeId = EndpointNodes.AddDefaulted();
				EndpointNodes[NodeId].Pos = Pos;
			}
			++EndpointNodes[NodeId].Degree;

			FEndpointRef Ref;
			Ref.Pos = Pos;
			Ref.Color = Color;
			Ref.SourceStrokeId = SourceStrokeId;
			Ref.LocalStrokeId = LocalStrokeId;
			Ref.EndpointIndex = EndpointIndex;
			Ref.bSynthetic = bSynthetic;
			EndpointRefs.Add(Ref);
		};
		for (int32 i = 0; i < PlanarSourceStrokes.Num(); ++i)
		{
			const FColoredStroke& S = PlanarSourceStrokes[i];
			if (S.Color != EStrokeColor::Red && S.Color != EStrokeColor::Black)
			{
				continue;
			}
			AddEndpointRef(S.Points, S.Color, PlanarSourceIds.IsValidIndex(i) ? PlanarSourceIds[i] : INDEX_NONE, i, 0, false);
			AddEndpointRef(S.Points, S.Color, PlanarSourceIds.IsValidIndex(i) ? PlanarSourceIds[i] : INDEX_NONE, i, 1, false);
		}
		const int32 FirstSyntheticBeforeBlackPass = SyntheticConnectors.Num();
		for (int32 i = 0; i < FirstSyntheticBeforeBlackPass; ++i)
		{
			const FColoredStroke& S = SyntheticConnectors[i];
			if (S.Color != EStrokeColor::Red && S.Color != EStrokeColor::Black)
			{
				continue;
			}
			AddEndpointRef(S.Points, S.Color, INDEX_NONE, i, 0, true);
			AddEndpointRef(S.Points, S.Color, INDEX_NONE, i, 1, true);
		}

		int32 BlackDeadEndCount = 0;
		int32 BlackEndpointConnectorCount = 0;
		TSet<FString> BlackConnectorKeys;
		for (const FEndpointRef& DeadEnd : EndpointRefs)
		{
			if (DeadEnd.Color != EStrokeColor::Black || DeadEnd.bSynthetic || !PlanarSourceStrokes.IsValidIndex(DeadEnd.LocalStrokeId))
			{
				continue;
			}

			int32 NodeDegree = 0;
			for (const FEndpointNode& Node : EndpointNodes)
			{
				if (FVector2D::DistSquared(DeadEnd.Pos, Node.Pos) <= CapGeometryEpsilon * CapGeometryEpsilon)
				{
					NodeDegree = Node.Degree;
					break;
				}
			}
			if (NodeDegree != 1)
			{
				continue;
			}
			++BlackDeadEndCount;

			const FVector2D Forward = GetEndpointOutwardDirection(PlanarSourceStrokes[DeadEnd.LocalStrokeId].Points, DeadEnd.EndpointIndex);
			if (Forward.IsNearlyZero())
			{
				continue;
			}

			int32 BestEndpointIndex = INDEX_NONE;
			double BestD2 = TNumericLimits<double>::Max();
			for (int32 CandidateIndex = 0; CandidateIndex < EndpointRefs.Num(); ++CandidateIndex)
			{
				const FEndpointRef& Candidate = EndpointRefs[CandidateIndex];
				if (Candidate.Color != EStrokeColor::Red && Candidate.Color != EStrokeColor::Black)
				{
					continue;
				}
				if (!Candidate.bSynthetic && Candidate.SourceStrokeId == DeadEnd.SourceStrokeId)
				{
					continue;
				}
				const FVector2D Delta = Candidate.Pos - DeadEnd.Pos;
				const double D2 = Delta.SizeSquared();
				if (D2 <= CapGeometryEpsilon * CapGeometryEpsilon || D2 > ConnectorTol2)
				{
					continue;
				}
				if (FVector2D::DotProduct(Delta, Forward) < -CapGeometryEpsilon)
				{
					continue;
				}
				if (D2 < BestD2)
				{
					BestD2 = D2;
					BestEndpointIndex = CandidateIndex;
				}
			}
			if (!EndpointRefs.IsValidIndex(BestEndpointIndex))
			{
				continue;
			}

			const FVector2D A = DeadEnd.Pos;
			const FVector2D B = EndpointRefs[BestEndpointIndex].Pos;
			const int32 Ax = FMath::RoundToInt(A.X);
			const int32 Ay = FMath::RoundToInt(A.Y);
			const int32 Bx = FMath::RoundToInt(B.X);
			const int32 By = FMath::RoundToInt(B.Y);
			const FString Key = (Ax < Bx || (Ax == Bx && Ay <= By))
				? FString::Printf(TEXT("%d,%d-%d,%d"), Ax, Ay, Bx, By)
				: FString::Printf(TEXT("%d,%d-%d,%d"), Bx, By, Ax, Ay);
			if (BlackConnectorKeys.Contains(Key))
			{
				continue;
			}
			BlackConnectorKeys.Add(Key);

			FColoredStroke Connector;
			Connector.Points.Add(A);
			Connector.Points.Add(B);
			Connector.Color = EStrokeColor::Black;
			Connector.ConnectionPointCount = Connector.Points.Num();
			SyntheticConnectors.Add(MoveTemp(Connector));
			++BlackEndpointConnectorCount;
		}

		OutStrokes.Reset();
		OutStrokes = MoveTemp(PlanarSourceStrokes);
		OutFirstSyntheticStrokeId = OutStrokes.Num();
		OutStrokes.Append(MoveTemp(SyntheticConnectors));
		ComputeStrokeMetrics(OutStrokes);

		UE_LOG(LogTemp, Log,
			TEXT("Sketch2D Step 9 topology: %d red/black intersection(s), %d initial red dead-end(s), %d red dead-end(s) after red repair, %d black dead-end(s), %d red-red connector(s), %d red-black connector(s), %d black endpoint connector(s), %d planar stroke(s)"),
			TrueIntersectionCount,
			InitialRedDeadEndCount,
			RedDeadEndsAfterRedRepair.Num(),
			BlackDeadEndCount,
			RedToRedConnectorCount,
			RedToBlackConnectorCount,
			BlackEndpointConnectorCount,
			OutStrokes.Num());
	}

	struct FLoopCandidate
	{
		FString Source;
		FString Key;
		FString RejectReason;
		int32 Priority = 0;
		int32 AnchorStrokeId = -1;
		int32 EdgeCount = 0;
		double LoopArea = 0.0;
		double LoopBboxArea = 0.0;
		TArray<int32> StrokeIds;
		TArray<int32> RedStrokeIds;
		TArray<int32> RealRedStrokeIds;
		TArray<int32> BlackStrokeIds;
		TArray<int32> SyntheticStrokeIds;
		FCapExtrusionResult Result;
		bool bSelected = false;
	};

	static void SortUniqueInts(TArray<int32>& Values)
	{
		Values.Sort();
		for (int32 i = Values.Num() - 1; i > 0; --i)
		{
			if (Values[i] == Values[i - 1])
			{
				Values.RemoveAt(i);
			}
		}
	}

	static FString JoinInts(const TArray<int32>& Values)
	{
		FString Out;
		for (int32 i = 0; i < Values.Num(); ++i)
		{
			Out += FString::Printf(TEXT("%s%d"), (i == 0 ? TEXT("") : TEXT(",")), Values[i]);
		}
		return Out;
	}

	static FString StrokeSetKey(TArray<int32> StrokeIds)
	{
		SortUniqueInts(StrokeIds);
		return JoinInts(StrokeIds);
	}

	static double PolygonAbsArea(const FStroke& Poly)
	{
		if (Poly.Num() < 3)
		{
			return 0.0;
		}
		double Sum = 0.0;
		for (int32 i = 0; i < Poly.Num(); ++i)
		{
			const FVector2D& A = Poly[i];
			const FVector2D& B = Poly[(i + 1) % Poly.Num()];
			Sum += A.X * B.Y - B.X * A.Y;
		}
		return FMath::Abs(0.5 * Sum);
	}

	static double PolygonBboxArea(const FStroke& Poly)
	{
		if (Poly.Num() == 0)
		{
			return 0.0;
		}

		double MinX = TNumericLimits<double>::Max();
		double MinY = TNumericLimits<double>::Max();
		double MaxX = -TNumericLimits<double>::Max();
		double MaxY = -TNumericLimits<double>::Max();
		for (const FVector2D& P : Poly)
		{
			MinX = FMath::Min(MinX, double(P.X));
			MinY = FMath::Min(MinY, double(P.Y));
			MaxX = FMath::Max(MaxX, double(P.X));
			MaxY = FMath::Max(MaxY, double(P.Y));
		}

		const int32 MinXi = FMath::FloorToInt(MinX);
		const int32 MinYi = FMath::FloorToInt(MinY);
		const int32 MaxXi = FMath::CeilToInt(MaxX);
		const int32 MaxYi = FMath::CeilToInt(MaxY);
		return double(MaxXi - MinXi + 1) * double(MaxYi - MinYi + 1);
	}

	static bool IsValidLoopPolygon(const FStroke& Poly)
	{
		return Poly.Num() >= 4 && PolygonAbsArea(Poly) > 1.0;
	}

	static bool MakeLoopCandidateFromCycle(
		const TArray<FColoredStroke>& Strokes,
		const CapOps::FGraph& G,
		const TArray<int32>& Cycle,
		const TCHAR* Source,
		int32 Priority,
		int32 AnchorStrokeId,
		FLoopCandidate& Out)
	{
		using namespace CapOps;
		Out = FLoopCandidate();
		if (Cycle.Num() < 2)
		{
			return false;
		}

		Out.Source = Source;
		Out.Priority = Priority;
		Out.AnchorStrokeId = AnchorStrokeId;
		Out.EdgeCount = Cycle.Num();
		Out.Result = FCapExtrusionResult();
		Out.Result.bFound = true;
		Out.Result.CandidateSource = Source;
		Out.Result.CandidateAnchorStrokeId = AnchorStrokeId;

		for (int32 e : Cycle)
		{
			if (!G.StrokeId.IsValidIndex(e))
			{
				return false;
			}
			const int32 StrokeId = G.StrokeId[e];
			const bool bSynthetic = G.bSynthetic.IsValidIndex(e) && G.bSynthetic[e] != 0;
			Out.StrokeIds.Add(StrokeId);
			Out.Result.CapStrokeIds.Add(StrokeId);
			if (bSynthetic)
			{
				Out.SyntheticStrokeIds.Add(StrokeId);
			}
			if (G.bBlack[e])
			{
				Out.BlackStrokeIds.Add(StrokeId);
			}
			else
			{
				Out.RedStrokeIds.Add(StrokeId);
				if (!bSynthetic)
				{
					Out.RealRedStrokeIds.Add(StrokeId);
				}
			}
		}
		SortUniqueInts(Out.StrokeIds);
		SortUniqueInts(Out.RedStrokeIds);
		SortUniqueInts(Out.RealRedStrokeIds);
		SortUniqueInts(Out.BlackStrokeIds);
		SortUniqueInts(Out.SyntheticStrokeIds);
		if (Out.RealRedStrokeIds.Num() == 0)
		{
			return false;
		}

		Out.Result.bUsedBlack = Out.BlackStrokeIds.Num() > 0;
		BuildPolygon(Strokes.GetData(), G, Cycle, Out.Result.CapPolygon, Out.Result.CapNodes);
		if (!IsValidLoopPolygon(Out.Result.CapPolygon))
		{
			return false;
		}
		Out.LoopArea = PolygonAbsArea(Out.Result.CapPolygon);
		Out.LoopBboxArea = PolygonBboxArea(Out.Result.CapPolygon);
		Out.Key = StrokeSetKey(Out.StrokeIds);
		return !Out.Key.IsEmpty();
	}

	static void AddUniqueCandidate(TArray<FLoopCandidate>& Candidates, TSet<FString>& SeenKeys, const FLoopCandidate& Candidate)
	{
		if (SeenKeys.Contains(Candidate.Key))
		{
			return;
		}
		SeenKeys.Add(Candidate.Key);
		Candidates.Add(Candidate);
	}

	static void CollectCycleCandidatesFromGraph(
		const TArray<FColoredStroke>& Strokes,
		const CapOps::FGraph& G,
		const TCHAR* Source,
		int32 Priority,
		bool bRequireBlack,
		TArray<FLoopCandidate>& Candidates,
		TSet<FString>& SeenKeys)
	{
		using namespace CapOps;
		TArray<uint8> EdgeUsable;
		EdgeUsable.Init(1, G.StrokeId.Num());
		for (int32 e = 0; e < G.StrokeId.Num(); ++e)
		{
			if (G.bBlack[e])
			{
				continue;
			}
			if (G.NodeU[e] == G.NodeV[e])
			{
				continue;
			}

			TArray<int32> Path;
			if (!BfsPath(G, G.NodeU[e], G.NodeV[e], /*ExcludeEdge*/ e, EdgeUsable, Path))
			{
				continue;
			}
			TArray<int32> Cycle = Path;
			Cycle.Add(e);

			FLoopCandidate Candidate;
			if (!MakeLoopCandidateFromCycle(Strokes, G, Cycle, Source, Priority, G.StrokeId[e], Candidate))
			{
				continue;
			}
			if (bRequireBlack && Candidate.BlackStrokeIds.Num() == 0)
			{
				continue;
			}
			AddUniqueCandidate(Candidates, SeenKeys, Candidate);
		}
	}

	static void GroupRedStrokeComponents(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& RedIdx,
		float NodeSnapTol,
		int32 FirstSyntheticStrokeId,
		TArray<TArray<int32>>& OutGroups)
	{
		using namespace CapOps;
		OutGroups.Reset();
		if (RedIdx.Num() == 0)
		{
			return;
		}

		FGraph G;
		BuildGraph(Strokes, RedIdx, NodeSnapTol, FirstSyntheticStrokeId, G);
		if (G.NumNodes <= 0)
		{
			return;
		}

		TArray<int32> Parent;
		Parent.SetNumUninitialized(G.NumNodes);
		for (int32 n = 0; n < G.NumNodes; ++n)
		{
			Parent[n] = n;
		}
		for (int32 e = 0; e < G.StrokeId.Num(); ++e)
		{
			Parent[Find(Parent, G.NodeV[e])] = Find(Parent, G.NodeU[e]);
		}

		TMap<int32, int32> GroupByRoot;
		for (int32 e = 0; e < G.StrokeId.Num(); ++e)
		{
			const int32 Root = Find(Parent, G.NodeU[e]);
			int32 GroupIndex = INDEX_NONE;
			if (int32* Existing = GroupByRoot.Find(Root))
			{
				GroupIndex = *Existing;
			}
			else
			{
				GroupIndex = OutGroups.Num();
				GroupByRoot.Add(Root, GroupIndex);
				OutGroups.AddDefaulted();
			}
			OutGroups[GroupIndex].Add(G.StrokeId[e]);
		}

		for (TArray<int32>& Group : OutGroups)
		{
			SortUniqueInts(Group);
		}
	}

	static void CollectRedFirstLoopCandidates(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& RedIdx,
		const TArray<int32>& BlackIdx,
		float NodeSnapTol,
		int32 FirstSyntheticStrokeId,
		TArray<FLoopCandidate>& OutCandidates)
	{
		using namespace CapOps;
		OutCandidates.Reset();
		TSet<FString> SeenKeys;

		// Phase 1: all red-only endpoint cycles. These get first selection priority.
		{
			FGraph RedGraph;
			BuildGraph(Strokes, RedIdx, NodeSnapTol, FirstSyntheticStrokeId, RedGraph);
			CollectCycleCandidatesFromGraph(
				Strokes, RedGraph, TEXT("red_only"), /*Priority*/ 0, /*bRequireBlack*/ false,
				OutCandidates, SeenKeys);
		}

		// Phase 2: each red-only component closes its own endpoints through the global black pool.
		TArray<TArray<int32>> RedGroups;
		GroupRedStrokeComponents(Strokes, RedIdx, NodeSnapTol, FirstSyntheticStrokeId, RedGroups);
		for (const TArray<int32>& RedGroup : RedGroups)
		{
			TArray<int32> EdgeIds = RedGroup;
			EdgeIds.Append(BlackIdx);
			FGraph LocalGraph;
			BuildGraph(Strokes, EdgeIds, NodeSnapTol, FirstSyntheticStrokeId, LocalGraph);
			CollectCycleCandidatesFromGraph(
				Strokes, LocalGraph, TEXT("local_black"), /*Priority*/ 1, /*bRequireBlack*/ true,
				OutCandidates, SeenKeys);
		}

		// Phase 3: fallback trace over all remaining-style red combinations plus all black.
		// Selection happens later, so red conflicts with stronger candidates will be rejected.
		{
			TArray<int32> EdgeIds = RedIdx;
			EdgeIds.Append(BlackIdx);
			FGraph FallbackGraph;
			BuildGraph(Strokes, EdgeIds, NodeSnapTol, FirstSyntheticStrokeId, FallbackGraph);
			CollectCycleCandidatesFromGraph(
				Strokes, FallbackGraph, TEXT("fallback_trace"), /*Priority*/ 2, /*bRequireBlack*/ false,
				OutCandidates, SeenKeys);
		}
	}

	static bool IsRedOnlyLoopCandidate(const FLoopCandidate& Candidate)
	{
		return Candidate.Source == TEXT("red_only");
	}

	static bool IsBorrowedLoopCandidate(const FLoopCandidate& Candidate)
	{
		return Candidate.Source == TEXT("local_black") || Candidate.Source == TEXT("fallback_trace");
	}

	static void SelectLoopCandidates(TArray<FLoopCandidate>& Candidates, TArray<int32>& OutSelectedIndices)
	{
		OutSelectedIndices.Reset();
		TArray<int32> Order;
		Order.Reserve(Candidates.Num());
		for (int32 i = 0; i < Candidates.Num(); ++i)
		{
			Order.Add(i);
			Candidates[i].bSelected = false;
			Candidates[i].RejectReason.Reset();
		}

		Order.Sort([&Candidates](int32 AIndex, int32 BIndex)
		{
			const FLoopCandidate& A = Candidates[AIndex];
			const FLoopCandidate& B = Candidates[BIndex];
			const bool bARedOnly = IsRedOnlyLoopCandidate(A);
			const bool bBRedOnly = IsRedOnlyLoopCandidate(B);
			if (A.Priority != B.Priority) { return A.Priority < B.Priority; }
			if (A.BlackStrokeIds.Num() != B.BlackStrokeIds.Num()) { return A.BlackStrokeIds.Num() < B.BlackStrokeIds.Num(); }
			if (bARedOnly && bBRedOnly)
			{
				if (FMath::Abs(A.LoopArea - B.LoopArea) > 1.0) { return A.LoopArea > B.LoopArea; }
				if (A.SyntheticStrokeIds.Num() != B.SyntheticStrokeIds.Num()) { return A.SyntheticStrokeIds.Num() < B.SyntheticStrokeIds.Num(); }
				if (A.RealRedStrokeIds.Num() != B.RealRedStrokeIds.Num()) { return A.RealRedStrokeIds.Num() > B.RealRedStrokeIds.Num(); }
			}
			if (A.EdgeCount != B.EdgeCount) { return A.EdgeCount < B.EdgeCount; }
			if (A.RealRedStrokeIds.Num() != B.RealRedStrokeIds.Num()) { return A.RealRedStrokeIds.Num() > B.RealRedStrokeIds.Num(); }
			if (A.AnchorStrokeId != B.AnchorStrokeId) { return A.AnchorStrokeId < B.AnchorStrokeId; }
			return FCString::Strcmp(*A.Key, *B.Key) < 0;
		});

		TMap<int32, int32> ConsumedRedStrokeOwner;
		for (int32 CandidateIndex : Order)
		{
			FLoopCandidate& Candidate = Candidates[CandidateIndex];
			if (IsBorrowedLoopCandidate(Candidate) && Candidate.LoopBboxArea < CapBorrowedLoopMinBboxArea)
			{
				Candidate.RejectReason = FString::Printf(TEXT("loop bbox area %.3f below %.3f"), Candidate.LoopBboxArea, CapBorrowedLoopMinBboxArea);
				continue;
			}

			int32 ConflictingRed = -1;
			int32 ConflictingOwner = -1;
			for (int32 RedStrokeId : Candidate.RealRedStrokeIds)
			{
				if (const int32* Owner = ConsumedRedStrokeOwner.Find(RedStrokeId))
				{
					ConflictingRed = RedStrokeId;
					ConflictingOwner = *Owner;
					break;
				}
			}
			if (ConflictingRed >= 0)
			{
				Candidate.RejectReason = FString::Printf(TEXT("red stroke %d already selected by candidate %d"), ConflictingRed, ConflictingOwner);
				continue;
			}

			Candidate.bSelected = true;
			Candidate.RejectReason = TEXT("selected");
			OutSelectedIndices.Add(CandidateIndex);
			for (int32 RedStrokeId : Candidate.RealRedStrokeIds)
			{
				ConsumedRedStrokeOwner.Add(RedStrokeId, CandidateIndex);
			}
		}
	}

	static FString JsonEscaped(FString Text)
	{
		Text.ReplaceInline(TEXT("\\"), TEXT("\\\\"));
		Text.ReplaceInline(TEXT("\""), TEXT("\\\""));
		return Text;
	}

	static void AppendIntArrayJson(FString& Json, const TCHAR* Key, const TArray<int32>& Values, bool bTrailingComma)
	{
		Json += FString::Printf(TEXT("    \"%s\": ["), Key);
		for (int32 i = 0; i < Values.Num(); ++i)
		{
			Json += FString::Printf(TEXT("%s%d"), (i == 0 ? TEXT("") : TEXT(", ")), Values[i]);
		}
		Json += bTrailingComma ? TEXT("],\n") : TEXT("]\n");
	}

	static bool SaveLoopCandidatesJson(const TArray<FLoopCandidate>& Candidates, const FString& Path)
	{
		FString Json;
		Json += TEXT("{\n");
		Json += FString::Printf(TEXT("  \"candidate_count\": %d,\n"), Candidates.Num());
		int32 SelectedCount = 0;
		for (const FLoopCandidate& Candidate : Candidates)
		{
			if (Candidate.bSelected)
			{
				++SelectedCount;
			}
		}
		Json += FString::Printf(TEXT("  \"selected_count\": %d,\n"), SelectedCount);
		Json += TEXT("  \"candidates\": [\n");
		for (int32 i = 0; i < Candidates.Num(); ++i)
		{
			const FLoopCandidate& C = Candidates[i];
			Json += TEXT("  {\n");
			Json += FString::Printf(TEXT("    \"index\": %d,\n"), i);
			Json += FString::Printf(TEXT("    \"source\": \"%s\",\n"), *JsonEscaped(C.Source));
			Json += FString::Printf(TEXT("    \"priority\": %d,\n"), C.Priority);
			Json += FString::Printf(TEXT("    \"anchor_stroke_id\": %d,\n"), C.AnchorStrokeId);
			Json += FString::Printf(TEXT("    \"selected\": %s,\n"), C.bSelected ? TEXT("true") : TEXT("false"));
			Json += FString::Printf(TEXT("    \"reason\": \"%s\",\n"), *JsonEscaped(C.RejectReason));
			Json += FString::Printf(TEXT("    \"edge_count\": %d,\n"), C.EdgeCount);
			Json += FString::Printf(TEXT("    \"loop_area\": %.3f,\n"), C.LoopArea);
			Json += FString::Printf(TEXT("    \"loop_bbox_area\": %.3f,\n"), C.LoopBboxArea);
			Json += FString::Printf(TEXT("    \"used_black\": %s,\n"), C.BlackStrokeIds.Num() > 0 ? TEXT("true") : TEXT("false"));
			AppendIntArrayJson(Json, TEXT("stroke_ids"), C.StrokeIds, true);
			AppendIntArrayJson(Json, TEXT("red_stroke_ids"), C.RedStrokeIds, true);
			AppendIntArrayJson(Json, TEXT("real_red_stroke_ids"), C.RealRedStrokeIds, true);
			AppendIntArrayJson(Json, TEXT("synthetic_stroke_ids"), C.SyntheticStrokeIds, true);
			AppendIntArrayJson(Json, TEXT("black_stroke_ids"), C.BlackStrokeIds, false);
			Json += FString::Printf(TEXT("  }%s\n"), (i + 1 < Candidates.Num() ? TEXT(",") : TEXT("")));
		}
		Json += TEXT("  ]\n");
		Json += TEXT("}\n");
		return FFileHelper::SaveStringToFile(Json, *Path);
	}

	// Even-odd ray-cast point-in-polygon test (polygon is the closed cap loop).
	static bool PointInPolygon(const FStroke& Poly, const FVector2D& P)
	{
		const int32 N = Poly.Num();
		if (N < 3) { return false; }
		bool bIn = false;
		for (int32 i = 0, j = N - 1; i < N; j = i++)
		{
			const FVector2D& A = Poly[i];
			const FVector2D& B = Poly[j];
			if (((A.Y > P.Y) != (B.Y > P.Y)) &&
				(P.X < (B.X - A.X) * (P.Y - A.Y) / (B.Y - A.Y) + A.X))
			{
				bIn = !bIn;
			}
		}
		return bIn;
	}

	static constexpr double InteriorGreenMinInsideLengthPx = 10.0;

	struct FInteriorGreenStats
	{
		int32 StrokeId = -1;
		int32 InsidePoints = 0;
		int32 TotalPoints = 0;
		double InsideRatio = 0.0;
		double InsideLength = 0.0;
		double StrokeLength = 0.0;
	};

	static double StrokeArcLength(const FColoredStroke& Stroke)
	{
		if (Stroke.bHasMetrics)
		{
			return Stroke.Arc;
		}

		double Arc = 0.0;
		const FStroke& Points = Stroke.Points;
		for (int32 k = 1; k < Points.Num(); ++k)
		{
			Arc += (Points[k] - Points[k - 1]).Size();
		}
		return Arc;
	}

	static FInteriorGreenStats MeasureInteriorGreen(const FStroke& CapPolygon, const TArray<FColoredStroke>& Strokes, int32 StrokeId)
	{
		FInteriorGreenStats Stats;
		Stats.StrokeId = StrokeId;
		if (!Strokes.IsValidIndex(StrokeId))
		{
			return Stats;
		}

		const FColoredStroke& Stroke = Strokes[StrokeId];
		const FStroke& Points = Stroke.Points;
		Stats.TotalPoints = Points.Num();
		Stats.StrokeLength = StrokeArcLength(Stroke);
		if (Points.Num() == 0)
		{
			return Stats;
		}

		bool bPrevInside = PointInPolygon(CapPolygon, Points[0]);
		if (bPrevInside)
		{
			++Stats.InsidePoints;
		}

		for (int32 k = 1; k < Points.Num(); ++k)
		{
			const bool bInside = PointInPolygon(CapPolygon, Points[k]);
			if (bInside)
			{
				++Stats.InsidePoints;
			}

			if (bPrevInside && bInside)
			{
				Stats.InsideLength += (Points[k] - Points[k - 1]).Size();
			}
			bPrevInside = bInside;
		}

		Stats.InsideRatio = (Stats.TotalPoints > 0)
			? double(Stats.InsidePoints) / double(Stats.TotalPoints)
			: 0.0;
		return Stats;
	}

	static bool PassesInteriorGreenThresholds(const FInteriorGreenStats& Stats)
	{
		return Stats.InsideLength >= InteriorGreenMinInsideLengthPx;
	}

	struct FGreenPixelClass
	{
		bool bInside = false;
		bool bOutside = false;
		bool bBoundary = false;
	};

	struct FGreenActionStats
	{
		int32 LocalGreenCount = 0;
		double InsideLength = 0.0;
		double OutsideLength = 0.0;
		double BoundaryLength = 0.0;
		FString Action = TEXT("skip");
		FString DecisionReason = TEXT("no_local_green");
		bool bHasBestInterior = false;
		FInteriorGreenStats BestInterior;
	};

	static void BuildLoopPixelMasks(const FStroke& Loop, int32 Width, int32 Height, TArray<uint8>& OutBoundary, TArray<uint8>& OutInterior)
	{
		const int32 N = Width * Height;
		OutBoundary.Init(0, N);
		OutInterior.Init(0, N);
		if (Width <= 0 || Height <= 0 || Loop.Num() < 3)
		{
			return;
		}

		TArray<FIntPoint> LineBuf;
		auto MarkLine = [&](const FVector2D& A, const FVector2D& B)
		{
			SkelGraph::LinePoints(
				FIntPoint(FMath::RoundToInt(A.X), FMath::RoundToInt(A.Y)),
				FIntPoint(FMath::RoundToInt(B.X), FMath::RoundToInt(B.Y)), LineBuf);
			for (const FIntPoint& P : LineBuf)
			{
				if (P.X >= 0 && P.X < Width && P.Y >= 0 && P.Y < Height)
				{
					OutBoundary[P.Y * Width + P.X] = 255;
				}
			}
		};

		for (int32 k = 0; k + 1 < Loop.Num(); ++k)
		{
			MarkLine(Loop[k], Loop[k + 1]);
		}
		if ((Loop[0] - Loop.Last()).SizeSquared() > 1e-6)
		{
			MarkLine(Loop.Last(), Loop[0]);
		}

		TArray<uint8> Reachable;
		Reachable.Init(0, N);
		TArray<int32> Stack;
		Stack.Reserve(1024);
		auto TryPush = [&](int32 X, int32 Y)
		{
			if (X < 0 || X >= Width || Y < 0 || Y >= Height)
			{
				return;
			}
			const int32 Idx = Y * Width + X;
			if (OutBoundary[Idx] == 0 && Reachable[Idx] == 0)
			{
				Reachable[Idx] = 1;
				Stack.Push(Idx);
			}
		};
		for (int32 X = 0; X < Width; ++X)
		{
			TryPush(X, 0);
			TryPush(X, Height - 1);
		}
		for (int32 Y = 0; Y < Height; ++Y)
		{
			TryPush(0, Y);
			TryPush(Width - 1, Y);
		}
		while (Stack.Num() > 0)
		{
			const int32 P = Stack.Pop(EAllowShrinking::No);
			const int32 X = P % Width;
			const int32 Y = P / Width;
			TryPush(X + 1, Y);
			TryPush(X - 1, Y);
			TryPush(X, Y + 1);
			TryPush(X, Y - 1);
		}

		for (int32 Idx = 0; Idx < N; ++Idx)
		{
			if (OutBoundary[Idx] == 0 && Reachable[Idx] == 0)
			{
				OutInterior[Idx] = 255;
			}
		}
	}

	static FGreenPixelClass ClassifyGreenPixelAgainstLoop(int32 X, int32 Y, int32 Width, int32 Height, const TArray<uint8>& Boundary, const TArray<uint8>& Interior)
	{
		FGreenPixelClass C;
		if (X < 0 || X >= Width || Y < 0 || Y >= Height)
		{
			C.bOutside = true;
			return C;
		}
		const int32 Idx = Y * Width + X;
		C.bBoundary = Boundary.IsValidIndex(Idx) && Boundary[Idx] != 0;
		if (C.bBoundary)
		{
			C.bInside = true;
			C.bOutside = true;
			return C;
		}
		C.bInside = Interior.IsValidIndex(Idx) && Interior[Idx] != 0;
		C.bOutside = !C.bInside;
		return C;
	}

	static void AddGreenPixelStepLength(
		const FIntPoint& Prev, const FGreenPixelClass& PrevClass,
		const FIntPoint& Cur, const FGreenPixelClass& CurClass,
		double& InsideLength, double& OutsideLength, double& BoundaryLength)
	{
		const double Dx = double(Cur.X - Prev.X);
		const double Dy = double(Cur.Y - Prev.Y);
		const double Len = FMath::Sqrt(Dx * Dx + Dy * Dy);
		if (Len <= 0.0)
		{
			return;
		}

		const bool bBoundary = PrevClass.bBoundary || CurClass.bBoundary;
		if (bBoundary)
		{
			InsideLength += Len;
			OutsideLength += Len;
			BoundaryLength += Len;
			return;
		}

		if (PrevClass.bInside && CurClass.bInside)
		{
			InsideLength += Len;
		}
		else if (PrevClass.bOutside && CurClass.bOutside)
		{
			OutsideLength += Len;
		}
		else
		{
			// A one-pixel crossing without a rasterized boundary hit is ambiguous.
			InsideLength += Len;
			OutsideLength += Len;
		}
	}

	static FGreenActionStats MeasureLocalGreenAction(
		const FStroke& CapPolygon,
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& LocalGreen,
		int32 Width,
		int32 Height)
	{
		FGreenActionStats Out;
		Out.LocalGreenCount = LocalGreen.Num();
		if (LocalGreen.Num() == 0)
		{
			Out.Action = TEXT("skip");
			Out.DecisionReason = TEXT("no_local_green");
			return Out;
		}

		TArray<uint8> Boundary;
		TArray<uint8> Interior;
		BuildLoopPixelMasks(CapPolygon, Width, Height, Boundary, Interior);

		TArray<FIntPoint> LineBuf;
		for (int32 g : LocalGreen)
		{
			if (!Strokes.IsValidIndex(g))
			{
				continue;
			}

			const FInteriorGreenStats InteriorStats = MeasureInteriorGreen(CapPolygon, Strokes, g);
			if (!Out.bHasBestInterior || InteriorStats.InsideLength > Out.BestInterior.InsideLength)
			{
				Out.bHasBestInterior = true;
				Out.BestInterior = InteriorStats;
			}

			const FStroke& Points = Strokes[g].Points;
			for (int32 k = 0; k + 1 < Points.Num(); ++k)
			{
				SkelGraph::LinePoints(
					FIntPoint(FMath::RoundToInt(Points[k].X), FMath::RoundToInt(Points[k].Y)),
					FIntPoint(FMath::RoundToInt(Points[k + 1].X), FMath::RoundToInt(Points[k + 1].Y)), LineBuf);
				if (LineBuf.Num() == 0)
				{
					continue;
				}

				FIntPoint Prev = LineBuf[0];
				FGreenPixelClass PrevClass = ClassifyGreenPixelAgainstLoop(Prev.X, Prev.Y, Width, Height, Boundary, Interior);
				for (int32 p = 1; p < LineBuf.Num(); ++p)
				{
					const FIntPoint Cur = LineBuf[p];
					if (Cur == Prev)
					{
						continue;
					}
					const FGreenPixelClass CurClass = ClassifyGreenPixelAgainstLoop(Cur.X, Cur.Y, Width, Height, Boundary, Interior);
					AddGreenPixelStepLength(Prev, PrevClass, Cur, CurClass, Out.InsideLength, Out.OutsideLength, Out.BoundaryLength);
					Prev = Cur;
					PrevClass = CurClass;
				}
			}
		}

		if (Out.InsideLength > Out.OutsideLength)
		{
			Out.Action = TEXT("excavate");
			Out.DecisionReason = TEXT("inside_greater_than_outside");
		}
		else if (Out.OutsideLength > Out.InsideLength)
		{
			Out.Action = TEXT("attach");
			Out.DecisionReason = TEXT("outside_greater_than_inside");
		}
		else
		{
			Out.Action = TEXT("skip");
			Out.DecisionReason = TEXT("inside_outside_equal");
		}
		return Out;
	}
	static double DistancePointToSegmentSquared2D(const FVector2D& P, const FVector2D& A, const FVector2D& B)
	{
		const FVector2D AB = B - A;
		const double LenSq = AB.SizeSquared();
		if (LenSq <= 1e-12)
		{
			return (P - A).SizeSquared();
		}

		const double T = FMath::Clamp(FVector2D::DotProduct(P - A, AB) / LenSq, 0.0, 1.0);
		return (P - (A + AB * T)).SizeSquared();
	}

	static double DistancePointToPolylineSquared(const FVector2D& P, const FStroke& Poly)
	{
		if (Poly.Num() == 0)
		{
			return TNumericLimits<double>::Max();
		}
		if (Poly.Num() == 1)
		{
			return (P - Poly[0]).SizeSquared();
		}

		double Best = TNumericLimits<double>::Max();
		for (int32 i = 0; i + 1 < Poly.Num(); ++i)
		{
			Best = FMath::Min(Best, DistancePointToSegmentSquared2D(P, Poly[i], Poly[i + 1]));
		}
		if (Poly.Num() > 2 && (Poly[0] - Poly.Last()).SizeSquared() > 1e-6)
		{
			Best = FMath::Min(Best, DistancePointToSegmentSquared2D(P, Poly.Last(), Poly[0]));
		}
		return Best;
	}

	static constexpr double GreenTraceEndpointTolPx = 10.0;
	static constexpr double GreenTraceMaxChainDeviationDeg = 45.0;

	struct FGreenSideTraceCandidate
	{
		bool bValid = false;
		int32 SeedStrokeId = -1;
		TArray<int32> ChainStrokeIds;
		FVector2D ChainStart = FVector2D::ZeroVector;
		FVector2D ChainEnd = FVector2D::ZeroVector;
		FVector2D SeedDirection = FVector2D::ZeroVector;
		double SideLength = 0.0;
		double PathLength = 0.0;
		double TotalGap = 0.0;
		FString StopReason;
	};

	struct FGreenTraceEnd
	{
		FVector2D Position = FVector2D::ZeroVector;
		FVector2D OutwardDirection = FVector2D::ZeroVector;
	};

	static FString TraceGreenChainEnd(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& AllGreen,
		TSet<int32>& VisitedGreenIds,
		FGreenTraceEnd& ChainEnd,
		bool bPrepend,
		TArray<int32>& ChainStrokeIds,
		double& TotalGap)
	{
		const double EndpointTolSq = GreenTraceEndpointTolPx * GreenTraceEndpointTolPx;
		const double MinDirectionDot = FMath::Cos(FMath::DegreesToRadians(GreenTraceMaxChainDeviationDeg));

		for (;;)
		{
			if (ChainEnd.OutwardDirection.IsNearlyZero())
			{
				return TEXT("zero_chain_direction");
			}

			int32 BestStrokeId = -1;
			FVector2D BestNearEndpoint = FVector2D::ZeroVector;
			FVector2D BestFarEndpoint = FVector2D::ZeroVector;
			double BestDirectionDot = -1.0;
			double BestGapSq = TNumericLimits<double>::Max();
			bool bFoundNearbyEndpoint = false;
			bool bFoundUnvisitedNearbyEndpoint = false;

			for (int32 GreenStrokeId : AllGreen)
			{
				if (!Strokes.IsValidIndex(GreenStrokeId) || Strokes[GreenStrokeId].Points.Num() < 2)
				{
					continue;
				}

				const FStroke& CandidatePoints = Strokes[GreenStrokeId].Points;
				for (int32 EndpointIndex = 0; EndpointIndex < 2; ++EndpointIndex)
				{
					const FVector2D NearEndpoint = EndpointIndex == 0 ? CandidatePoints[0] : CandidatePoints.Last();
					const FVector2D FarEndpoint = EndpointIndex == 0 ? CandidatePoints.Last() : CandidatePoints[0];
					const double GapSq = FVector2D::DistSquared(ChainEnd.Position, NearEndpoint);
					if (GapSq > EndpointTolSq)
					{
						continue;
					}
					bFoundNearbyEndpoint = true;
					if (VisitedGreenIds.Contains(GreenStrokeId))
					{
						continue;
					}
					bFoundUnvisitedNearbyEndpoint = true;

					const FVector2D CandidateDirection = (FarEndpoint - NearEndpoint).GetSafeNormal();
					if (CandidateDirection.IsNearlyZero())
					{
						continue;
					}

					const double DirectionDot = FVector2D::DotProduct(
						ChainEnd.OutwardDirection, CandidateDirection);
					if (DirectionDot <= MinDirectionDot)
					{
						continue;
					}

					if (DirectionDot > BestDirectionDot ||
						(DirectionDot == BestDirectionDot && GapSq < BestGapSq) ||
						(DirectionDot == BestDirectionDot && GapSq == BestGapSq &&
							(BestStrokeId < 0 || GreenStrokeId < BestStrokeId)))
					{
						BestStrokeId = GreenStrokeId;
						BestNearEndpoint = NearEndpoint;
						BestFarEndpoint = FarEndpoint;
						BestDirectionDot = DirectionDot;
						BestGapSq = GapSq;
					}
				}
			}

			if (BestStrokeId < 0)
			{
				if (!bFoundNearbyEndpoint)
				{
					return TEXT("no_nearby_green");
				}
				if (!bFoundUnvisitedNearbyEndpoint)
				{
					return TEXT("all_nearby_green_visited");
				}
				return TEXT("no_chain_direction_match");
			}

			VisitedGreenIds.Add(BestStrokeId);
			if (bPrepend)
			{
				ChainStrokeIds.Insert(BestStrokeId, 0);
			}
			else
			{
				ChainStrokeIds.Add(BestStrokeId);
			}
			TotalGap += FMath::Sqrt(BestGapSq);
			ChainEnd.Position = BestFarEndpoint;
			ChainEnd.OutwardDirection = (BestFarEndpoint - BestNearEndpoint).GetSafeNormal();
		}
	}

	static FGreenSideTraceCandidate TraceGreenSideFromSeed(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& AllGreen,
		int32 SeedStrokeId,
		const FStroke& CapPolygon)
	{
		FGreenSideTraceCandidate Out;
		Out.SeedStrokeId = SeedStrokeId;
		if (!Strokes.IsValidIndex(SeedStrokeId) || Strokes[SeedStrokeId].Points.Num() < 2)
		{
			Out.StopReason = TEXT("invalid_seed");
			return Out;
		}

		const FStroke& SeedPoints = Strokes[SeedStrokeId].Points;
		Out.SeedDirection = (SeedPoints.Last() - SeedPoints[0]).GetSafeNormal();
		if (Out.SeedDirection.IsNearlyZero())
		{
			Out.StopReason = TEXT("zero_seed_direction");
			return Out;
		}

		TSet<int32> VisitedGreenIds;
		VisitedGreenIds.Add(SeedStrokeId);
		Out.ChainStrokeIds.Add(SeedStrokeId);

		FGreenTraceEnd FirstEnd;
		FirstEnd.Position = SeedPoints[0];
		FirstEnd.OutwardDirection = (SeedPoints[0] - SeedPoints.Last()).GetSafeNormal();
		FGreenTraceEnd LastEnd;
		LastEnd.Position = SeedPoints.Last();
		LastEnd.OutwardDirection = (SeedPoints.Last() - SeedPoints[0]).GetSafeNormal();

		const FString FirstStopReason = TraceGreenChainEnd(
			Strokes, AllGreen, VisitedGreenIds, FirstEnd, true,
			Out.ChainStrokeIds, Out.TotalGap);
		const FString LastStopReason = TraceGreenChainEnd(
			Strokes, AllGreen, VisitedGreenIds, LastEnd, false,
			Out.ChainStrokeIds, Out.TotalGap);

		const double FirstDistanceToCapSq =
			DistancePointToPolylineSquared(FirstEnd.Position, CapPolygon);
		const double LastDistanceToCapSq =
			DistancePointToPolylineSquared(LastEnd.Position, CapPolygon);
		if (FirstDistanceToCapSq <= LastDistanceToCapSq)
		{
			Out.ChainStart = FirstEnd.Position;
			Out.ChainEnd = LastEnd.Position;
		}
		else
		{
			Out.ChainStart = LastEnd.Position;
			Out.ChainEnd = FirstEnd.Position;
			Algo::Reverse(Out.ChainStrokeIds);
		}

		Out.PathLength = Out.TotalGap;
		for (int32 GreenStrokeId : Out.ChainStrokeIds)
		{
			if (Strokes.IsValidIndex(GreenStrokeId))
			{
				Out.PathLength += StrokeArcLength(Strokes[GreenStrokeId]);
			}
		}

		Out.SideLength = FVector2D::Distance(Out.ChainStart, Out.ChainEnd);
		Out.bValid = Out.SideLength > 0.0;
		Out.StopReason = FString::Printf(
			TEXT("bidirectional:first=%s,last=%s"), *FirstStopReason, *LastStopReason);
		return Out;
	}

	static FString MakeGreenChainKey(const TArray<int32>& ChainStrokeIds)
	{
		TArray<int32> SortedIds = ChainStrokeIds;
		SortedIds.Sort();

		FString Key;
		for (int32 StrokeId : SortedIds)
		{
			Key += FString::Printf(TEXT("%d,"), StrokeId);
		}
		return Key;
	}

	// Local green strokes only seed discovery. Each seed is traced from both ends,
	// duplicate chains are removed, and the longest full path is selected. The
	// selected path is oriented only after tracing, from its cap-near endpoint to
	// its cap-far endpoint.
	static void ApplyGreenSideForCap(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& AllGreen,
		const TArray<int32>& LocalGreenSeeds,
		FCapExtrusionResult& Out)
	{
		Out.SideCandidateVectors.Reset();
		Out.SideCandidateStarts.Reset();
		Out.SideCandidateEnds.Reset();
		Out.SideStrokeId = -1;
		Out.SideVector = FVector2D::ZeroVector;
		Out.SideLength = 0.0;
		Out.SideChainPathLength = 0.0;
		Out.SideChainStart = FVector2D::ZeroVector;
		Out.SideChainEnd = FVector2D::ZeroVector;
		Out.SideSeedDirection = FVector2D::ZeroVector;
		Out.SideChainStrokeIds.Reset();
		Out.SideTraceTotalGap = 0.0;
		Out.SideTraceStopReason = TEXT("no_local_green_seed");

		FGreenSideTraceCandidate Best;
		TSet<FString> SeenChainKeys;
		for (int32 SeedStrokeId : LocalGreenSeeds)
		{
			const FGreenSideTraceCandidate Candidate =
				TraceGreenSideFromSeed(Strokes, AllGreen, SeedStrokeId, Out.CapPolygon);
			if (!Candidate.bValid)
			{
				continue;
			}

			const FString ChainKey = MakeGreenChainKey(Candidate.ChainStrokeIds);
			if (SeenChainKeys.Contains(ChainKey))
			{
				continue;
			}
			SeenChainKeys.Add(ChainKey);

			const bool bLongerPath = !Best.bValid || Candidate.PathLength > Best.PathLength + 1e-6;
			const bool bSamePathButLongerChord =
				Best.bValid && FMath::IsNearlyEqual(Candidate.PathLength, Best.PathLength, 1e-6) &&
				Candidate.SideLength > Best.SideLength + 1e-6;
			if (bLongerPath || bSamePathButLongerChord)
			{
				Best = Candidate;
			}
		}

		if (Best.bValid)
		{
			const FVector2D ChainVector = Best.ChainEnd - Best.ChainStart;
			Out.SideStrokeId = Best.SeedStrokeId;
			Out.SideLength = ChainVector.Size();
			Out.SideChainPathLength = Best.PathLength;
			Out.SideChainStart = Best.ChainStart;
			Out.SideChainEnd = Best.ChainEnd;
			Out.SideSeedDirection = Best.SeedDirection;
			Out.SideChainStrokeIds = Best.ChainStrokeIds;
			Out.SideTraceTotalGap = Best.TotalGap;
			Out.SideTraceStopReason = Best.StopReason;
			Out.SideVector = ChainVector;
			Out.SideCandidateStarts.Add(Best.ChainStart);
			Out.SideCandidateEnds.Add(Best.ChainEnd);
			Out.SideCandidateVectors.Add(Out.SideVector);
		}
		Out.CapPolygonTranslated.Reset();
		Out.CapPolygonTranslated.Reserve(Out.CapPolygon.Num());
		for (const FVector2D& P : Out.CapPolygon) { Out.CapPolygonTranslated.Add(P + Out.SideVector); }
	}

	int32 RecoverCapExtrusionsPerComponent(const TArray<FColoredStroke>& Strokes, float ConnectorTol, float BlackSelectTol, int32 Width, int32 Height, const FString& PressDir, const FString& ActionPressDir, TArray<FCapExtrusionResult>& OutResults)
	{
		using namespace CapOps;
		OutResults.Reset();

		TArray<int32> SourceRedIdx, SourceBlackIdx;
		for (int32 i = 0; i < Strokes.Num(); ++i)
		{
			switch (Strokes[i].Color)
			{
			case EStrokeColor::Red:   SourceRedIdx.Add(i); break;
			case EStrokeColor::Black: SourceBlackIdx.Add(i); break;
			default: break;
			}
		}

		TArray<FColoredStroke> TraceStrokes;
		int32 FirstSyntheticStrokeId = Strokes.Num();
		BuildPlanarEndpointConnectorStrokes(
			Strokes, SourceRedIdx, SourceBlackIdx, ConnectorTol,
			TraceStrokes, FirstSyntheticStrokeId);

		TArray<int32> RedIdx, BlackIdx, GreenIdx;
		for (int32 i = 0; i < TraceStrokes.Num(); ++i)
		{
			switch (TraceStrokes[i].Color)
			{
			case EStrokeColor::Red:   RedIdx.Add(i); break;
			case EStrokeColor::Black: BlackIdx.Add(i); break;
			case EStrokeColor::Green: GreenIdx.Add(i); break;
			default: break;
			}
		}

		{
			TArray<int32> AllRedBlackIdx = RedIdx;
			AllRedBlackIdx.Append(BlackIdx);
			if (AllRedBlackIdx.Num() > 0)
			{
				FGraph AllRedBlackGraph;
				BuildGraph(TraceStrokes, AllRedBlackIdx, CapLoopGraphNodeSnapTol, FirstSyntheticStrokeId, AllRedBlackGraph);
				TArray<uint8> AllEdges;
				AllEdges.Init(1, AllRedBlackGraph.StrokeId.Num());
				SaveGraphPng(TraceStrokes, AllRedBlackGraph, AllEdges, Width, Height, PressDir / TEXT("09_all_red_black_graph.png"));
				SaveGraphJson(TraceStrokes, AllRedBlackGraph, AllEdges, PressDir / TEXT("09_all_red_black_graph.json"));
			}
		}

		if (RedIdx.Num() == 0)
		{
			return 0;
		}

		const double SelTol2 = double(BlackSelectTol) * double(BlackSelectTol);

		TArray<FLoopCandidate> Candidates;
		CollectRedFirstLoopCandidates(TraceStrokes, RedIdx, BlackIdx, CapLoopGraphNodeSnapTol, FirstSyntheticStrokeId, Candidates);

		TArray<int32> SelectedCandidateIndices;
		SelectLoopCandidates(Candidates, SelectedCandidateIndices);
		SaveLoopCandidatesJson(Candidates, PressDir / TEXT("09_loop_candidates.json"));

		if (SelectedCandidateIndices.Num() == 0)
		{
			return 0;
		}

		// Pre-warm the image-wrapper module on this thread before parallel writes.
		FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));

		// Each selected red-driven loop writes into its own Component_%% folder.
		OutResults.SetNum(SelectedCandidateIndices.Num());
		ParallelFor(SelectedCandidateIndices.Num(), [&](int32 i)
		{
			const FLoopCandidate Candidate = Candidates[SelectedCandidateIndices[i]];
			const FString CompDir = PressDir / FString::Printf(TEXT("Component_%02d"), i + 1);
			IFileManager::Get().MakeDirectory(*CompDir, /*Tree*/ true);

			FCapExtrusionResult R = Candidate.Result;

			{
				FGraph SelectedGraph;
				BuildGraph(TraceStrokes, R.CapStrokeIds, CapLoopGraphNodeSnapTol, FirstSyntheticStrokeId, SelectedGraph);
				TArray<uint8> All;
				All.Init(1, SelectedGraph.StrokeId.Num());
				SaveGraphPng(TraceStrokes, SelectedGraph, All, Width, Height, CompDir / TEXT("09a_caploop_candidate.png"));
				SaveGraphJson(TraceStrokes, SelectedGraph, All, CompDir / TEXT("09a_caploop_candidate_graph.json"));
				SaveGraphPng(TraceStrokes, SelectedGraph, All, Width, Height, CompDir / TEXT("09b_caploop_pruned.png"));
				SaveGraphJson(TraceStrokes, SelectedGraph, All, CompDir / TEXT("09b_caploop_pruned_graph.json"));
			}

			// Every local green is an independent discovery seed. Recover its full chain
			// from both ends, select the longest unique path, then orient that path from
			// the endpoint nearest the cap toward the endpoint farthest from the cap.
			TArray<FVector2D> CompRedPts;
			for (int32 r : Candidate.RealRedStrokeIds) { CompRedPts.Append(TraceStrokes[r].Points); }
			TArray<int32> LocalGreen;
			for (int32 g : GreenIdx)
			{
				const FStroke& GP = TraceStrokes[g].Points;
				if (GP.Num() == 0) { continue; }
				bool bNear = false;
				for (const FVector2D& RP : CompRedPts)
				{
					if (FVector2D::DistSquared(RP, GP[0]) <= SelTol2 || FVector2D::DistSquared(RP, GP.Last()) <= SelTol2)
					{
						bNear = true; break;
					}
				}
				if (bNear) { LocalGreen.Add(g); }
			}
			ApplyGreenSideForCap(TraceStrokes, GreenIdx, LocalGreen, R);

			SaveCapExtrusionPng(TraceStrokes, R, Width, Height, CompDir / TEXT("09_cap_extrusion.png"));

			// Action test: only this component's local green strokes vote. Green pixels
			// on the cap boundary count as both inside and outside.
			const FGreenActionStats ActionStats = MeasureLocalGreenAction(R.CapPolygon, TraceStrokes, LocalGreen, Width, Height);
			R.Action = ActionStats.Action;
			R.ActionDecisionReason = ActionStats.DecisionReason;
			R.LocalGreenStrokeCount = ActionStats.LocalGreenCount;
			R.GreenInsideTotalLength = ActionStats.InsideLength;
			R.GreenOutsideTotalLength = ActionStats.OutsideLength;
			R.bHasInteriorGreen = R.Action == TEXT("excavate");
			if (ActionStats.bHasBestInterior)
			{
				R.InteriorGreenStrokeId = ActionStats.BestInterior.StrokeId;
				R.InteriorGreenInsidePoints = ActionStats.BestInterior.InsidePoints;
				R.InteriorGreenTotalPoints = ActionStats.BestInterior.TotalPoints;
				R.InteriorGreenInsideRatio = ActionStats.BestInterior.InsideRatio;
				R.InteriorGreenInsideLength = ActionStats.BestInterior.InsideLength;
				R.InteriorGreenStrokeLength = ActionStats.BestInterior.StrokeLength;
			}
			SaveCapExtrusionJson(R, CompDir / TEXT("09_cap_extrusion.json"));

			const FString ActionCompDir = ActionPressDir / FString::Printf(TEXT("Component_%02d"), i + 1);
			IFileManager::Get().MakeDirectory(*ActionCompDir, /*Tree*/ true);
			const FString ActionJson = FString::Printf(
				TEXT("{\n")
				TEXT("  \"action\": \"%s\",\n")
				TEXT("  \"has_interior_green\": %s,\n")
				TEXT("  \"local_green_count\": %d,\n")
				TEXT("  \"inside_green_total\": %.3f,\n")
				TEXT("  \"outside_green_total\": %.3f,\n")
				TEXT("  \"boundary_green_total\": %.3f,\n")
				TEXT("  \"decision_reason\": \"%s\",\n")
				TEXT("  \"interior_green_stroke_id\": %d,\n")
				TEXT("  \"interior_green_inside_length\": %.3f,\n")
				TEXT("  \"interior_green_min_inside_length\": %.3f\n")
				TEXT("}\n"),
				*R.Action,
				R.bHasInteriorGreen ? TEXT("true") : TEXT("false"),
				R.LocalGreenStrokeCount,
				R.GreenInsideTotalLength,
				R.GreenOutsideTotalLength,
				ActionStats.BoundaryLength,
				*JsonEscaped(R.ActionDecisionReason),
				R.InteriorGreenStrokeId,
				R.InteriorGreenInsideLength,
				InteriorGreenMinInsideLengthPx);
			FFileHelper::SaveStringToFile(ActionJson, *(ActionCompDir / TEXT("Action.json")));

			OutResults[i] = R;
		});

		return SelectedCandidateIndices.Num();
	}

	bool SaveCapExtrusionPng(const TArray<FColoredStroke>& Strokes, const FCapExtrusionResult& Res, int32 Width, int32 Height, const FString& Path, int32 Thickness)
	{
		const int32 N = Width * Height;
		TArray<uint8> RGBA;
		RGBA.Init(255, N * 4);
		const int32 Rad = FMath::Max(0, Thickness / 2);

		auto PlotCol = [&](int32 x, int32 y, uint8 r, uint8 g, uint8 b)
		{
			for (int32 oy = -Rad; oy <= Rad; ++oy)
			{
				for (int32 ox = -Rad; ox <= Rad; ++ox)
				{
					const int32 xx = x + ox;
					const int32 yy = y + oy;
					if (xx < 0 || xx >= Width || yy < 0 || yy >= Height)
					{
						continue;
					}
					const int32 Off = (yy * Width + xx) * 4;
					RGBA[Off + 0] = r; RGBA[Off + 1] = g; RGBA[Off + 2] = b; RGBA[Off + 3] = 255;
				}
			}
		};
		TArray<FIntPoint> LineBuf;
		auto DrawPoly = [&](const FStroke& Poly, uint8 r, uint8 g, uint8 b)
		{
			for (int32 k = 0; k + 1 < Poly.Num(); ++k)
			{
				SkelGraph::LinePoints(
					FIntPoint(FMath::RoundToInt(Poly[k].X), FMath::RoundToInt(Poly[k].Y)),
					FIntPoint(FMath::RoundToInt(Poly[k + 1].X), FMath::RoundToInt(Poly[k + 1].Y)), LineBuf);
				for (const FIntPoint& P : LineBuf)
				{
					PlotCol(P.X, P.Y, r, g, b);
				}
			}
		};

		// Selected source green chain. Its start-to-end vector determines both the
		// extrusion direction and side length.
		for (int32 StrokeId : Res.SideChainStrokeIds)
		{
			if (Strokes.IsValidIndex(StrokeId))
			{
				DrawPoly(Strokes[StrokeId].Points, 0, 200, 0);
			}
		}

		// Side connectors (green) between corresponding cap nodes.
		for (const FVector2D& Nd : Res.CapNodes)
		{
			const FVector2D Nd2 = Nd + Res.SideVector;
			SkelGraph::LinePoints(
				FIntPoint(FMath::RoundToInt(Nd.X), FMath::RoundToInt(Nd.Y)),
				FIntPoint(FMath::RoundToInt(Nd2.X), FMath::RoundToInt(Nd2.Y)), LineBuf);
			for (const FIntPoint& P : LineBuf)
			{
				PlotCol(P.X, P.Y, 34, 139, 34);
			}
		}

		DrawPoly(Res.CapPolygonTranslated, 255, 140, 0); // translated cap: orange
		DrawPoly(Res.CapPolygon, 220, 20, 60);           // original cap: red

		IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
		if (!IW.IsValid())
		{
			return false;
		}
		IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
		const TArray64<uint8>& C = IW->GetCompressed();
		return FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *Path);
	}

	bool SaveCapExtrusionJson(const FCapExtrusionResult& Res, const FString& Path)
	{
		FString Json;
		Json += TEXT("{\n");
		Json += FString::Printf(TEXT("  \"found\": %s,\n"), Res.bFound ? TEXT("true") : TEXT("false"));
		Json += FString::Printf(TEXT("  \"used_black\": %s,\n"), Res.bUsedBlack ? TEXT("true") : TEXT("false"));
		Json += FString::Printf(TEXT("  \"side_stroke_id\": %d,\n"), Res.SideStrokeId);
		Json += FString::Printf(TEXT("  \"side_vector\": [%.3f, %.3f],\n"), Res.SideVector.X, Res.SideVector.Y);
		Json += FString::Printf(TEXT("  \"side_length\": %.3f,\n"), Res.SideLength);
		Json += FString::Printf(TEXT("  \"side_chain_path_length\": %.3f,\n"), Res.SideChainPathLength);
		Json += FString::Printf(TEXT("  \"side_chain_start\": [%.3f, %.3f],\n"), Res.SideChainStart.X, Res.SideChainStart.Y);
		Json += FString::Printf(TEXT("  \"side_chain_end\": [%.3f, %.3f],\n"), Res.SideChainEnd.X, Res.SideChainEnd.Y);
		Json += FString::Printf(TEXT("  \"side_seed_direction\": [%.6f, %.6f],\n"), Res.SideSeedDirection.X, Res.SideSeedDirection.Y);
		Json += FString::Printf(TEXT("  \"side_trace_total_gap\": %.3f,\n"), Res.SideTraceTotalGap);
		Json += FString::Printf(TEXT("  \"side_trace_stop_reason\": \"%s\",\n"), *JsonEscaped(Res.SideTraceStopReason));
		Json += FString::Printf(TEXT("  \"green_trace_endpoint_tol\": %.3f,\n"), GreenTraceEndpointTolPx);
		Json += FString::Printf(TEXT("  \"green_trace_max_chain_deviation_degrees\": %.3f,\n"), GreenTraceMaxChainDeviationDeg);
		Json += TEXT("  \"side_chain_stroke_ids\": [");
		for (int32 i = 0; i < Res.SideChainStrokeIds.Num(); ++i)
		{
			Json += FString::Printf(TEXT("%s%d"), i == 0 ? TEXT("") : TEXT(", "), Res.SideChainStrokeIds[i]);
		}
		Json += TEXT("],\n");
		Json += FString::Printf(TEXT("  \"action\": \"%s\",\n"), *JsonEscaped(Res.Action));
		Json += FString::Printf(TEXT("  \"decision_reason\": \"%s\",\n"), *JsonEscaped(Res.ActionDecisionReason));
		Json += FString::Printf(TEXT("  \"local_green_count\": %d,\n"), Res.LocalGreenStrokeCount);
		Json += FString::Printf(TEXT("  \"inside_green_total\": %.3f,\n"), Res.GreenInsideTotalLength);
		Json += FString::Printf(TEXT("  \"outside_green_total\": %.3f,\n"), Res.GreenOutsideTotalLength);
		Json += FString::Printf(TEXT("  \"has_interior_green\": %s,\n"), Res.bHasInteriorGreen ? TEXT("true") : TEXT("false"));
		Json += FString::Printf(TEXT("  \"interior_green_stroke_id\": %d,\n"), Res.InteriorGreenStrokeId);
		Json += FString::Printf(TEXT("  \"interior_green_inside_points\": %d,\n"), Res.InteriorGreenInsidePoints);
		Json += FString::Printf(TEXT("  \"interior_green_total_points\": %d,\n"), Res.InteriorGreenTotalPoints);
		Json += FString::Printf(TEXT("  \"interior_green_inside_ratio\": %.6f,\n"), Res.InteriorGreenInsideRatio);
		Json += FString::Printf(TEXT("  \"interior_green_inside_length\": %.3f,\n"), Res.InteriorGreenInsideLength);
		Json += FString::Printf(TEXT("  \"interior_green_stroke_length\": %.3f,\n"), Res.InteriorGreenStrokeLength);
		Json += FString::Printf(TEXT("  \"interior_green_min_inside_length\": %.3f,\n"), InteriorGreenMinInsideLengthPx);
		Json += FString::Printf(TEXT("  \"candidate_source\": \"%s\",\n"), *Res.CandidateSource);
		Json += FString::Printf(TEXT("  \"candidate_anchor_stroke_id\": %d,\n"), Res.CandidateAnchorStrokeId);

		// Selected green side stroke (chord vector + endpoint segment).
		Json += TEXT("  \"side_vectors\": [");
		for (int32 i = 0; i < Res.SideCandidateVectors.Num(); ++i)
		{
			Json += FString::Printf(TEXT("%s[%.3f, %.3f]"), (i == 0 ? TEXT("") : TEXT(", ")),
				Res.SideCandidateVectors[i].X, Res.SideCandidateVectors[i].Y);
		}
		Json += TEXT("],\n");

		Json += TEXT("  \"side_segments\": [");
		for (int32 i = 0; i < Res.SideCandidateStarts.Num(); ++i)
		{
			Json += FString::Printf(TEXT("%s[[%.2f, %.2f], [%.2f, %.2f]]"), (i == 0 ? TEXT("") : TEXT(", ")),
				Res.SideCandidateStarts[i].X, Res.SideCandidateStarts[i].Y,
				Res.SideCandidateEnds[i].X, Res.SideCandidateEnds[i].Y);
		}
		Json += TEXT("],\n");

		Json += TEXT("  \"cap_stroke_ids\": [");
		for (int32 i = 0; i < Res.CapStrokeIds.Num(); ++i)
		{
			Json += FString::Printf(TEXT("%s%d"), (i == 0 ? TEXT("") : TEXT(", ")), Res.CapStrokeIds[i]);
		}
		Json += TEXT("],\n");

		auto WritePoly = [&](const TCHAR* Key, const FStroke& Poly, bool bTrailingComma)
		{
			Json += FString::Printf(TEXT("  \"%s\": ["), Key);
			for (int32 k = 0; k < Poly.Num(); ++k)
			{
				Json += FString::Printf(TEXT("%s[%.2f, %.2f]"), (k == 0 ? TEXT("") : TEXT(", ")), Poly[k].X, Poly[k].Y);
			}
			Json += bTrailingComma ? TEXT("],\n") : TEXT("]\n");
		};
		WritePoly(TEXT("cap_polygon"), Res.CapPolygon, true);
		WritePoly(TEXT("cap_polygon_translated"), Res.CapPolygonTranslated, false);
		Json += TEXT("}\n");

		return FFileHelper::SaveStringToFile(Json, *Path);
	}

	static void StrokeColorRGB(EStrokeColor Color, uint8& R, uint8& G, uint8& B)
	{
		switch (Color)
		{
		case EStrokeColor::Red:   R = 220; G = 20;  B = 60;  break;
		case EStrokeColor::Green: R = 34;  G = 139; B = 34;  break;
		case EStrokeColor::Blue:  R = 30;  G = 144; B = 255; break;
		case EStrokeColor::Black: R = 0;   G = 0;   B = 0;   break;
		default:                  R = 160; G = 160; B = 160; break; // None -> gray
		}
	}

	bool SaveColoredStrokesPng(const TArray<FColoredStroke>& Strokes, int32 Width, int32 Height, const FString& Path, int32 Thickness)
	{
		const int32 N = Width * Height;
		TArray<uint8> RGBA;
		RGBA.Init(255, N * 4);
		const int32 Rad = FMath::Max(0, Thickness / 2);

		TArray<FIntPoint> LineBuf;
		for (const FColoredStroke& CS : Strokes)
		{
			uint8 CR, CG, CB;
			StrokeColorRGB(CS.Color, CR, CG, CB);
			for (int32 k = 0; k + 1 < CS.Points.Num(); ++k)
			{
				const FIntPoint P0(FMath::RoundToInt(CS.Points[k].X), FMath::RoundToInt(CS.Points[k].Y));
				const FIntPoint P1(FMath::RoundToInt(CS.Points[k + 1].X), FMath::RoundToInt(CS.Points[k + 1].Y));
				SkelGraph::LinePoints(P0, P1, LineBuf);
				for (const FIntPoint& P : LineBuf)
				{
					for (int32 oy = -Rad; oy <= Rad; ++oy)
					{
						for (int32 ox = -Rad; ox <= Rad; ++ox)
						{
							const int32 xx = P.X + ox;
							const int32 yy = P.Y + oy;
							if (xx < 0 || xx >= Width || yy < 0 || yy >= Height)
							{
								continue;
							}
							const int32 Off = (yy * Width + xx) * 4;
							RGBA[Off + 0] = CR;
							RGBA[Off + 1] = CG;
							RGBA[Off + 2] = CB;
							RGBA[Off + 3] = 255;
						}
					}
				}
			}
		}

		IImageWrapperModule& IWM = FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));
		TSharedPtr<IImageWrapper> IW = IWM.CreateImageWrapper(EImageFormat::PNG);
		if (!IW.IsValid())
		{
			return false;
		}
		IW->SetRaw(RGBA.GetData(), RGBA.Num(), Width, Height, ERGBFormat::RGBA, 8);
		const TArray64<uint8>& C = IW->GetCompressed();
		return FFileHelper::SaveArrayToFile(TArrayView<const uint8>(C.GetData(), static_cast<int32>(C.Num())), *Path);
	}

	bool SaveColoredStrokesJson(const TArray<FColoredStroke>& Strokes, int32 Width, int32 Height, const FString& Path, float EndpointTol)
	{
		const int32 M = Strokes.Num();

		// Endpoint adjacency: two strokes are neighbors when any of their endpoints
		// lie within EndpointTol pixels of each other.
		const double Tol2 = double(EndpointTol) * double(EndpointTol);
		auto Endpoints = [](const FColoredStroke& S, FVector2D& A, FVector2D& B)
		{
			A = S.Points.Num() > 0 ? S.Points[0] : FVector2D::ZeroVector;
			B = S.Points.Num() > 0 ? S.Points.Last() : FVector2D::ZeroVector;
		};

		TArray<TArray<int32>> Neighbors;
		Neighbors.SetNum(M);
		for (int32 i = 0; i < M; ++i)
		{
			FVector2D Ai, Bi;
			Endpoints(Strokes[i], Ai, Bi);
			for (int32 j = i + 1; j < M; ++j)
			{
				FVector2D Aj, Bj;
				Endpoints(Strokes[j], Aj, Bj);
				const double D[4] = {
					FVector2D::DistSquared(Ai, Aj), FVector2D::DistSquared(Ai, Bj),
					FVector2D::DistSquared(Bi, Aj), FVector2D::DistSquared(Bi, Bj)
				};
				bool bAdj = false;
				for (double d : D)
				{
					if (d <= Tol2) { bAdj = true; break; }
				}
				if (bAdj)
				{
					Neighbors[i].Add(j);
					Neighbors[j].Add(i);
				}
			}
		}

		FString Json;
		Json.Reserve(M * 256 + 256);
		Json += TEXT("{\n");
		Json += FString::Printf(TEXT("  \"width\": %d,\n  \"height\": %d,\n  \"endpoint_tol\": %.2f,\n  \"stroke_count\": %d,\n"),
			Width, Height, EndpointTol, M);
		Json += TEXT("  \"strokes\": [\n");
		for (int32 i = 0; i < M; ++i)
		{
			const FColoredStroke& S = Strokes[i];
			FVector2D A, B;
			Endpoints(S, A, B);
			Json += TEXT("    {\n");
			Json += FString::Printf(TEXT("      \"id\": %d,\n"), i);
			Json += FString::Printf(TEXT("      \"color\": \"%s\",\n"), StrokeColorToString(S.Color));
			Json += FString::Printf(TEXT("      \"num_points\": %d,\n"), S.Points.Num());
			Json += FString::Printf(TEXT("      \"connection_point_count\": %d,\n"), S.ConnectionPointCount);
			Json += FString::Printf(TEXT("      \"endpoints\": [[%.2f, %.2f], [%.2f, %.2f]],\n"), A.X, A.Y, B.X, B.Y);

			if (S.bHasMetrics)
			{
				const TCHAR* Kind = (S.Straightness >= 0.9) ? TEXT("line") : TEXT("curve");
				Json += FString::Printf(TEXT("      \"kind\": \"%s\",\n"), Kind);
				Json += FString::Printf(TEXT("      \"arc\": %.3f,\n"), S.Arc);
				Json += FString::Printf(TEXT("      \"chord\": %.3f,\n"), S.Chord);
				Json += FString::Printf(TEXT("      \"straightness\": %.4f,\n"), S.Straightness);
				Json += FString::Printf(TEXT("      \"direction\": [%.4f, %.4f],\n"), S.Direction.X, S.Direction.Y);
				Json += FString::Printf(TEXT("      \"p90_pca_line_error\": %.3f,\n"), S.P90PcaError);
				Json += FString::Printf(TEXT("      \"pca_rms_error\": %.3f,\n"), S.PcaRmsError);
				Json += FString::Printf(TEXT("      \"p90_chord_deviation\": %.3f,\n"), S.P90ChordDev);
				Json += FString::Printf(TEXT("      \"chord_deviation_ratio\": %.4f,\n"), S.ChordDevRatio);
			}

			Json += TEXT("      \"neighbors\": [");
			for (int32 n = 0; n < Neighbors[i].Num(); ++n)
			{
				Json += FString::Printf(TEXT("%s%d"), (n == 0 ? TEXT("") : TEXT(", ")), Neighbors[i][n]);
			}
			Json += TEXT("],\n");

			Json += TEXT("      \"points\": [");
			for (int32 k = 0; k < S.Points.Num(); ++k)
			{
				Json += FString::Printf(TEXT("%s[%.2f, %.2f]"), (k == 0 ? TEXT("") : TEXT(", ")), S.Points[k].X, S.Points[k].Y);
			}
			Json += TEXT("]\n");
			Json += (i + 1 < M) ? TEXT("    },\n") : TEXT("    }\n");
		}
		Json += TEXT("  ]\n}\n");

		return FFileHelper::SaveStringToFile(Json, *Path);
	}

	bool SaveMaskPng(const TArray<uint8>& Mask, int32 Width, int32 Height, const FString& Path, bool bInvertForDisplay)
	{
		const int32 N = Width * Height;
		TArray<uint8> RGBA;
		RGBA.SetNumUninitialized(N * 4);
		for (int32 i = 0; i < N; ++i)
		{
			uint8 v = Mask[i];
			if (bInvertForDisplay)
			{
				// foreground -> black, background -> white
				v = Mask[i] > 0 ? 0 : 255;
			}
			const int32 Off = i * 4;
			RGBA[Off + 0] = v;
			RGBA[Off + 1] = v;
			RGBA[Off + 2] = v;
			RGBA[Off + 3] = 255;
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
			*Path
		);
	}
}
