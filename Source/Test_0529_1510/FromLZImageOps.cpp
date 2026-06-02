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
		TArray<uint8>& OutConnected, TArray<uint8>& OutSmallLoopPruned, TArray<uint8>& OutCleaned)
	{
		const int32 N = Width * Height;

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

		struct FConn { FIntPoint P0; FIntPoint P1; };
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
			Connections.Add({ EndpointsBefore[i], EndpointsBefore[j] });
			Used[i] = true;
			Used[j] = true;
		}

		// connected = work + drawn 1px connection lines.
		TArray<uint8> Connected = Work;
		TArray<FIntPoint> LineBuf;
		for (const FConn& C : Connections)
		{
			SkelGraph::LinePoints(C.P0, C.P1, LineBuf);
			for (const FIntPoint& P : LineBuf)
			{
				if (P.X >= 0 && P.X < Width && P.Y >= 0 && P.Y < Height)
				{
					Connected[P.Y * Width + P.X] = 255;
				}
			}
		}
		OutConnected = Connected;

		// ---- 2. prune connections that only close a small loop -----------------
		TArray<uint8> Current = Connected;
		const float LoopThresh = FMath::Max(0.0f, SmallLoopBboxAreaThresh);
		if (LoopThresh > 0.0f)
		{
			for (const FConn& C : Connections)
			{
				SkelGraph::LinePoints(C.P0, C.P1, LineBuf);

				// Pixels added by this connection (background in original Work) that are
				// currently still foreground.
				TArray<FIntPoint> AddedPts;
				for (const FIntPoint& P : LineBuf)
				{
					if (P.X < 0 || P.X >= Width || P.Y < 0 || P.Y >= Height)
					{
						continue;
					}
					const int32 Idx = P.Y * Width + P.X;
					if (Work[Idx] == 0 && Current[Idx] > 0)
					{
						AddedPts.Add(P);
					}
				}
				if (AddedPts.Num() == 0)
				{
					continue;
				}

				// Temporarily remove the added pixels and check for an alternate path.
				for (const FIntPoint& P : AddedPts)
				{
					Current[P.Y * Width + P.X] = 0;
				}
				TArray<FIntPoint> Path;
				const bool bAltPath = SkelGraph::ShortestPath(Current.GetData(), Width, Height, C.P0, C.P1, Path);
				if (!bAltPath)
				{
					// No alternate path => not a loop; restore the connection.
					for (const FIntPoint& P : AddedPts)
					{
						Current[P.Y * Width + P.X] = 255;
					}
					continue;
				}

				// Loop bbox over the alternate path + the added pixels.
				int32 MinX = MAX_int32, MinY = MAX_int32, MaxX = MIN_int32, MaxY = MIN_int32;
				auto Acc = [&](const FIntPoint& P)
				{
					MinX = FMath::Min(MinX, P.X); MinY = FMath::Min(MinY, P.Y);
					MaxX = FMath::Max(MaxX, P.X); MaxY = FMath::Max(MaxY, P.Y);
				};
				for (const FIntPoint& P : Path) { Acc(P); }
				for (const FIntPoint& P : AddedPts) { Acc(P); }
				const int64 BboxArea = int64(MaxX - MinX + 1) * int64(MaxY - MinY + 1);

				if (double(BboxArea) < double(LoopThresh))
				{
					// Small loop: keep the added pixels removed.
				}
				else
				{
					// Large loop: restore the connection.
					for (const FIntPoint& P : AddedPts)
					{
						Current[P.Y * Width + P.X] = 255;
					}
				}
			}
		}
		OutSmallLoopPruned = Current;

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
		// Union-find clustering of endpoints within NodeTol into shared loop nodes.
		struct FGraph
		{
			TArray<int32> NodeU;   // per edge: start node
			TArray<int32> NodeV;   // per edge: end node
			TArray<int32> StrokeId;// per edge: index into the source stroke array
			TArray<uint8> bBlack;  // per edge: 1 if black, 0 if red
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

		// Build a graph over the given edge strokes (by index into Strokes), clustering
		// their endpoints with NodeTol.
		void BuildGraph(const TArray<FColoredStroke>& Strokes, const TArray<int32>& EdgeStrokes, float NodeTol, FGraph& G)
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
			const double Tol2 = double(NodeTol) * double(NodeTol);
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

		// Render a graph: red edges in red, black edges in black, nodes as blue squares.
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
				const uint8 r = G.bBlack[e] ? 0 : 220;
				const uint8 g = G.bBlack[e] ? 0 : 20;
				const uint8 b = G.bBlack[e] ? 0 : 60;
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
				Json += FString::Printf(TEXT("    {\"stroke_id\": %d, \"u\": %d, \"v\": %d, \"color\": \"%s\", \"kept\": %s}%s\n"),
					G.StrokeId[e], G.NodeU[e], G.NodeV[e],
					G.bBlack[e] ? TEXT("black") : TEXT("red"),
					bKept ? TEXT("true") : TEXT("false"),
					(e + 1 < E ? TEXT(",") : TEXT("")));
			}
			Json += TEXT("  ]\n}\n");
			return FFileHelper::SaveStringToFile(Json, *Path);
		}
	} // namespace CapOps

	// Detect a single cap loop within a subset of strokes (one connected component).
	// Phase 1 tries a red-only loop; phase 2 prunes dangling black branches and takes the
	// smallest red-anchored cycle over red + surviving black. Writes 09a/09b graph debug into
	// CompDir when non-empty. Does NOT compute the green side / translation.
	static bool DetectCapLoopOnSubset(
		const TArray<FColoredStroke>& Strokes,
		const TArray<int32>& RedSub, const TArray<int32>& BlackSub,
		float NodeTol, int32 Width, int32 Height,
		const FString& CompDir, FCapExtrusionResult& Out)
	{
		using namespace CapOps;
		Out = FCapExtrusionResult();
		if (RedSub.Num() == 0)
		{
			return false;
		}

		// ============================ Phase 1: red only ============================
		{
			FGraph Gr;
			BuildGraph(Strokes, RedSub, NodeTol, Gr);
			TArray<uint8> AllUsable;  AllUsable.Init(1, Gr.StrokeId.Num());
			TArray<uint8> AllRed;     AllRed.Init(1, Gr.StrokeId.Num());
			TArray<int32> Cycle;
			if (FindRedCycle(Gr, AllRed, AllUsable, Cycle))
			{
				Out.bFound = true;
				Out.bUsedBlack = false;
				for (int32 e : Cycle) { Out.CapStrokeIds.Add(Gr.StrokeId[e]); }
				BuildPolygon(Strokes.GetData(), Gr, Cycle, Out.CapPolygon, Out.CapNodes);
				if (!CompDir.IsEmpty())
				{
					TArray<uint8> All; All.Init(1, Gr.StrokeId.Num());
					SaveGraphPng(Strokes, Gr, All, Width, Height, CompDir / TEXT("09a_caploop_candidate.png"));
					SaveGraphJson(Strokes, Gr, All, CompDir / TEXT("09a_caploop_candidate_graph.json"));
					SaveGraphPng(Strokes, Gr, All, Width, Height, CompDir / TEXT("09b_caploop_pruned.png"));
					SaveGraphJson(Strokes, Gr, All, CompDir / TEXT("09b_caploop_pruned_graph.json"));
				}
				return true;
			}
		}

		// ================== Phase 2: red + this component's black ==================
		// Build the candidate graph over all red + this component's black, mark red edges.
		TArray<int32> Edges2 = RedSub;
		Edges2.Append(BlackSub);
		FGraph G2;
		BuildGraph(Strokes, Edges2, NodeTol, G2);
		const int32 E2 = G2.StrokeId.Num();

		TArray<uint8> bRed2; bRed2.Init(0, E2);
		for (int32 e = 0; e < E2; ++e)
		{
			bRed2[e] = G2.bBlack[e] ? 0 : 1;
		}

		if (!CompDir.IsEmpty())
		{
			TArray<uint8> All; All.Init(1, E2);
			SaveGraphPng(Strokes, G2, All, Width, Height, CompDir / TEXT("09a_caploop_candidate.png"));
			SaveGraphJson(Strokes, G2, All, CompDir / TEXT("09a_caploop_candidate_graph.json"));
		}

		// 2d. iteratively prune dangling black branches (degree-1 black edges). Red is kept.
		TArray<uint8> Kept; Kept.Init(1, E2);
		for (;;)
		{
			TArray<int32> Deg; Deg.Init(0, G2.NumNodes);
			for (int32 e = 0; e < E2; ++e)
			{
				if (!Kept[e]) { continue; }
				Deg[G2.NodeU[e]] += 1;
				Deg[G2.NodeV[e]] += 1;
			}
			bool bChanged = false;
			for (int32 e = 0; e < E2; ++e)
			{
				if (!Kept[e] || !G2.bBlack[e]) { continue; }
				if (Deg[G2.NodeU[e]] <= 1 || Deg[G2.NodeV[e]] <= 1)
				{
					Kept[e] = 0;
					bChanged = true;
				}
			}
			if (!bChanged) { break; }
		}

		if (!CompDir.IsEmpty())
		{
			SaveGraphPng(Strokes, G2, Kept, Width, Height, CompDir / TEXT("09b_caploop_pruned.png"));
			SaveGraphJson(Strokes, G2, Kept, CompDir / TEXT("09b_caploop_pruned_graph.json"));
		}

		// 2e. find the smallest cycle anchored on a red edge, traversing red strokes plus
		// the surviving (non-spur) black strokes. Red tails/spurs hang off the loop and are
		// simply not part of the smallest cycle, so they are ignored automatically.
		// Step-7 strokes are already split at every junction/crossing during tracing, so the
		// loop closes purely through shared endpoints (no mid-segment intersection needed).
		TArray<uint8> Usable; Usable.Init(0, E2);
		for (int32 e = 0; e < E2; ++e)
		{
			Usable[e] = (bRed2[e] || Kept[e]) ? 1 : 0; // all red + surviving black
		}

		TArray<int32> Cycle;
		if (FindRedCycle(G2, bRed2, Usable, Cycle))
		{
			Out.bFound = true;
			Out.bUsedBlack = false;
			for (int32 e : Cycle)
			{
				Out.CapStrokeIds.Add(G2.StrokeId[e]);
				if (G2.bBlack[e]) { Out.bUsedBlack = true; }
			}
			BuildPolygon(Strokes.GetData(), G2, Cycle, Out.CapPolygon, Out.CapNodes);
			return true;
		}

		return false;
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

	static constexpr double InteriorGreenMinInsideLengthPx = 50.0;

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

	// Pick the longest green among GreenCandidates as the extrusion side and translate-copy
	// the cap along it (away from the cap centre).
	static void ApplyGreenSideForCap(const TArray<FColoredStroke>& Strokes, const TArray<int32>& GreenCandidates, FCapExtrusionResult& Out)
	{
		// Cap centre, used to orient every green chord away from the cap.
		FVector2D CapCenter = FVector2D::ZeroVector;
		{
			const TArray<FVector2D>& Ref = (Out.CapNodes.Num() > 0) ? Out.CapNodes : Out.CapPolygon;
			for (const FVector2D& P : Ref) { CapCenter += P; }
			if (Ref.Num() > 0) { CapCenter /= double(Ref.Num()); }
		}

		Out.SideCandidateVectors.Reset();
		Out.SideCandidateStarts.Reset();
		Out.SideCandidateEnds.Reset();

		int32 BestGreen = -1;
		double BestArc = -1.0;
		for (int32 g : GreenCandidates)
		{
			const FStroke& P = Strokes[g].Points;
			if (P.Num() < 2) { continue; }
			double Arc = Strokes[g].Arc;
			if (!Strokes[g].bHasMetrics)
			{
				Arc = 0.0;
				for (int32 k = 1; k < P.Num(); ++k) { Arc += (P[k] - P[k - 1]).Size(); }
			}

			// Record this connected green's chord (oriented away from the cap centre)
			// so step 10 can test the face normal against every green, not just the longest.
			const double DistToStart = FVector2D::DistSquared(P[0], CapCenter);
			const double DistToEnd = FVector2D::DistSquared(P.Last(), CapCenter);
			const FVector2D Start = (DistToStart <= DistToEnd) ? P[0] : P.Last();
			const FVector2D End = (DistToStart <= DistToEnd) ? P.Last() : P[0];
			Out.SideCandidateStarts.Add(Start);
			Out.SideCandidateEnds.Add(End);
			Out.SideCandidateVectors.Add(End - Start);

			if (Arc > BestArc) { BestArc = Arc; BestGreen = g; }
		}
		if (BestGreen >= 0 && Strokes[BestGreen].Points.Num() >= 2)
		{
			const FStroke& GP = Strokes[BestGreen].Points;
			Out.SideStrokeId = BestGreen;
			const double DistToStart = FVector2D::DistSquared(GP[0], CapCenter);
			const double DistToEnd = FVector2D::DistSquared(GP.Last(), CapCenter);
			Out.SideVector = (DistToStart <= DistToEnd) ? (GP.Last() - GP[0]) : (GP[0] - GP.Last());
		}
		Out.CapPolygonTranslated.Reset();
		Out.CapPolygonTranslated.Reserve(Out.CapPolygon.Num());
		for (const FVector2D& P : Out.CapPolygon) { Out.CapPolygonTranslated.Add(P + Out.SideVector); }
	}

	int32 RecoverCapExtrusionsPerComponent(const TArray<FColoredStroke>& Strokes, float NodeTol, float BlackSelectTol, int32 Width, int32 Height, const FString& PressDir, const FString& ActionPressDir, TArray<FCapExtrusionResult>& OutResults)
	{
		using namespace CapOps;
		OutResults.Reset();

		TArray<int32> RedIdx, BlackIdx, GreenIdx;
		for (int32 i = 0; i < Strokes.Num(); ++i)
		{
			switch (Strokes[i].Color)
			{
			case EStrokeColor::Red:   RedIdx.Add(i); break;
			case EStrokeColor::Black: BlackIdx.Add(i); break;
			case EStrokeColor::Green: GreenIdx.Add(i); break;
			default: break;
			}
		}
		if (RedIdx.Num() == 0)
		{
			return 0;
		}

		// Candidate connectivity graph: all red + black whose endpoint is within
		// BlackSelectTol of any red vertex. Connected components separate distinct objects.
		TArray<FVector2D> RedPts;
		for (int32 r : RedIdx) { RedPts.Append(Strokes[r].Points); }
		const double SelTol2 = double(BlackSelectTol) * double(BlackSelectTol);

		TArray<int32> SelBlack;
		for (int32 b : BlackIdx)
		{
			const FStroke& BP = Strokes[b].Points;
			if (BP.Num() == 0) { continue; }
			bool bNear = false;
			for (const FVector2D& RP : RedPts)
			{
				if (FVector2D::DistSquared(RP, BP[0]) <= SelTol2 || FVector2D::DistSquared(RP, BP.Last()) <= SelTol2)
				{
					bNear = true; break;
				}
			}
			if (bNear) { SelBlack.Add(b); }
		}

		TArray<int32> Edges = RedIdx;
		Edges.Append(SelBlack);
		FGraph G;
		BuildGraph(Strokes, Edges, NodeTol, G);

		// Union-find connected components over the candidate graph nodes.
		TArray<int32> Parent; Parent.SetNumUninitialized(G.NumNodes);
		for (int32 n = 0; n < G.NumNodes; ++n) { Parent[n] = n; }
		for (int32 e = 0; e < G.StrokeId.Num(); ++e)
		{
			Parent[Find(Parent, G.NodeV[e])] = Find(Parent, G.NodeU[e]);
		}

		// Group stroke indices per component (in discovery order).
		TMap<int32, int32> CompOf;
		TArray<TArray<int32>> CompRed, CompBlack;
		for (int32 e = 0; e < G.StrokeId.Num(); ++e)
		{
			const int32 Root = Find(Parent, G.NodeU[e]);
			int32 Idx;
			if (int32* Found = CompOf.Find(Root)) { Idx = *Found; }
			else { Idx = CompRed.Num(); CompOf.Add(Root, Idx); CompRed.AddDefaulted(); CompBlack.AddDefaulted(); }
			if (G.bBlack[e]) { CompBlack[Idx].Add(G.StrokeId[e]); }
			else { CompRed[Idx].Add(G.StrokeId[e]); }
		}

		// Pass 1 (sequential, cheap): which red-bearing components actually close a loop?
		TArray<int32> SuccessComps;
		for (int32 c = 0; c < CompRed.Num(); ++c)
		{
			if (CompRed[c].Num() == 0) { continue; }
			FCapExtrusionResult Tmp;
			if (DetectCapLoopOnSubset(Strokes, CompRed[c], CompBlack[c], NodeTol, Width, Height, FString(), Tmp))
			{
				SuccessComps.Add(c);
			}
		}
		if (SuccessComps.Num() == 0)
		{
			return 0;
		}

		// Pre-warm the image-wrapper module on this thread before parallel writes.
		FModuleManager::LoadModuleChecked<IImageWrapperModule>(TEXT("ImageWrapper"));

		// Pass 2 (parallel): each component writes into its own Component_%% folder.
		OutResults.SetNum(SuccessComps.Num());
		ParallelFor(SuccessComps.Num(), [&](int32 i)
		{
			const int32 c = SuccessComps[i];
			const FString CompDir = PressDir / FString::Printf(TEXT("Component_%02d"), i + 1);
			IFileManager::Get().MakeDirectory(*CompDir, /*Tree*/ true);

			FCapExtrusionResult R;
			DetectCapLoopOnSubset(Strokes, CompRed[c], CompBlack[c], NodeTol, Width, Height, CompDir, R);

			// Local side: longest green whose endpoint is near this component's red vertices;
			// fall back to the globally longest green if none is nearby.
			TArray<FVector2D> CompRedPts;
			for (int32 r : CompRed[c]) { CompRedPts.Append(Strokes[r].Points); }
			TArray<int32> LocalGreen;
			for (int32 g : GreenIdx)
			{
				const FStroke& GP = Strokes[g].Points;
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
			ApplyGreenSideForCap(Strokes, LocalGreen.Num() > 0 ? LocalGreen : GreenIdx, R);

			SaveCapExtrusionPng(Strokes, R, Width, Height, CompDir / TEXT("09_cap_extrusion.png"));

			// Interior-green test: any green stroke with at least 50px of accumulated
			// inside length within the cap polygon means excavate.
			bool bInteriorGreen = false;
			FInteriorGreenStats BestInterior;
			for (int32 g : GreenIdx)
			{
				const FInteriorGreenStats Stats = MeasureInteriorGreen(R.CapPolygon, Strokes, g);
				if (!PassesInteriorGreenThresholds(Stats))
				{
					continue;
				}
				if (!bInteriorGreen || Stats.InsideLength > BestInterior.InsideLength)
				{
					bInteriorGreen = true;
					BestInterior = Stats;
				}
			}
			R.bHasInteriorGreen = bInteriorGreen;
			if (bInteriorGreen)
			{
				R.InteriorGreenStrokeId = BestInterior.StrokeId;
				R.InteriorGreenInsidePoints = BestInterior.InsidePoints;
				R.InteriorGreenTotalPoints = BestInterior.TotalPoints;
				R.InteriorGreenInsideRatio = BestInterior.InsideRatio;
				R.InteriorGreenInsideLength = BestInterior.InsideLength;
				R.InteriorGreenStrokeLength = BestInterior.StrokeLength;
			}
			SaveCapExtrusionJson(R, CompDir / TEXT("09_cap_extrusion.json"));

			const FString ActionCompDir = ActionPressDir / FString::Printf(TEXT("Component_%02d"), i + 1);
			IFileManager::Get().MakeDirectory(*ActionCompDir, /*Tree*/ true);
			const FString ActionJson = FString::Printf(
				TEXT("{\n")
				TEXT("  \"action\": \"%s\",\n")
				TEXT("  \"has_interior_green\": %s,\n")
				TEXT("  \"interior_green_stroke_id\": %d,\n")
				TEXT("  \"interior_green_inside_length\": %.3f,\n")
				TEXT("  \"interior_green_min_inside_length\": %.3f\n")
				TEXT("}\n"),
				bInteriorGreen ? TEXT("excavate") : TEXT("attach"),
				bInteriorGreen ? TEXT("true") : TEXT("false"),
				R.InteriorGreenStrokeId,
				R.InteriorGreenInsideLength,
				InteriorGreenMinInsideLengthPx);
			FFileHelper::SaveStringToFile(ActionJson, *(ActionCompDir / TEXT("Action.json")));

			OutResults[i] = R;
		});

		return SuccessComps.Num();
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
		Json += FString::Printf(TEXT("  \"has_interior_green\": %s,\n"), Res.bHasInteriorGreen ? TEXT("true") : TEXT("false"));
		Json += FString::Printf(TEXT("  \"interior_green_stroke_id\": %d,\n"), Res.InteriorGreenStrokeId);
		Json += FString::Printf(TEXT("  \"interior_green_inside_points\": %d,\n"), Res.InteriorGreenInsidePoints);
		Json += FString::Printf(TEXT("  \"interior_green_total_points\": %d,\n"), Res.InteriorGreenTotalPoints);
		Json += FString::Printf(TEXT("  \"interior_green_inside_ratio\": %.6f,\n"), Res.InteriorGreenInsideRatio);
		Json += FString::Printf(TEXT("  \"interior_green_inside_length\": %.3f,\n"), Res.InteriorGreenInsideLength);
		Json += FString::Printf(TEXT("  \"interior_green_stroke_length\": %.3f,\n"), Res.InteriorGreenStrokeLength);
		Json += FString::Printf(TEXT("  \"interior_green_min_inside_length\": %.3f,\n"), InteriorGreenMinInsideLengthPx);

		// All green strokes connected to this cap (chord vectors + endpoint segments).
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
