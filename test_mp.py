import mediapipe as mp
print("Direct import:", dir(mp))
if hasattr(mp, "solutions"):
    print("Solutions found.")
else:
    print("NO SOLUTIONS ATTR")
