#pragma once

#include "CoreMinimal.h"

class UWorld;

class FFromLZFaceReconstructor
{
public:
	static void ProcessPress(const FString& PressDir, const FString& ActionPressDir, TWeakObjectPtr<UWorld> World);
};
