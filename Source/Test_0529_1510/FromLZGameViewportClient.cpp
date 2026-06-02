#include "FromLZGameViewportClient.h"

#include "FromLZCaptureUtils.h"
#include "FromLZFaceReconstructor.h"
#include "FromLZSketchProcessor.h"
#include "Engine/GameInstance.h"
#include "InputCoreTypes.h"

void UFromLZGameViewportClient::Init(FWorldContext& WorldContext, UGameInstance* OwningGameInstance, bool bCreateNewAudioDevice)
{
	Super::Init(WorldContext, OwningGameInstance, bCreateNewAudioDevice);

	const FString WorldName = GetWorld() ? GetWorld()->GetName() : TEXT("<null>");
	UE_LOG(LogTemp, Log, TEXT("FromLZGameViewportClient initialized. World=%s GameInstance=%s"), *WorldName, *GetNameSafe(OwningGameInstance));
}

bool UFromLZGameViewportClient::InputKey(const FInputKeyEventArgs& EventArgs)
{
	if (EventArgs.Event == IE_Pressed)
	{
		if (EventArgs.Key == EKeys::Enter)
		{
			UE_LOG(LogTemp, Log, TEXT("CaptureFromLZ invoked from viewport input. Key=%s"), *EventArgs.Key.ToString());
			FFromLZCaptureUtils::CaptureFromWorld(GetWorld());
		}
		else if (EventArgs.Key == EKeys::LeftShift || EventArgs.Key == EKeys::RightShift)
		{
			UE_LOG(LogTemp, Log, TEXT("Step11 restore invoked from viewport input. Key=%s"), *EventArgs.Key.ToString());
			FFromLZFaceReconstructor::RestoreStep11RuntimeBooleans(GetWorld());
		}
		else if (EventArgs.Key == EKeys::SpaceBar)
		{
			UE_LOG(LogTemp, Log, TEXT("ProcessSketch invoked from viewport input. Key=%s"), *EventArgs.Key.ToString());
			FFromLZSketchProcessor::ProcessLatestSketch(GetWorld());
		}
	}

	return Super::InputKey(EventArgs);
}
