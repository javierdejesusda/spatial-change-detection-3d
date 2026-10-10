# Capture guide

How to record a before/after video pair with a phone so the system can
reconstruct the scene and compare it reliably.

## Same area, shared reference view

- Record both videos in the same area.
- Start and end each recording at the same reference view (for example, the
  doorway facing the far wall). This anchors both videos to one frame of
  reference and helps loop closure.

## How to move

- Walk slowly and steadily. A full pass of about 30 to 60 seconds is enough
  for a room.
- Keep consecutive views overlapping. Do not jump between unrelated spots.
- Close the loop: end where you started, so the path overlaps itself.
- Cover each region you care about from at least two different angles.

## What to avoid

- Fast pans and sudden turns, which cause motion blur.
- Mirrors, glass, and screens showing moving content.
- Large lighting changes between the two recordings (curtains opened or
  closed, lights switched on or off).
- People or pets in the frame. Moving subjects confuse the reconstruction and
  can look like changes.

## Camera settings

- Hold the phone in the same orientation for both videos: landscape or
  portrait, not a mix.
- Record at 720p or 1080p, 30 fps. Higher resolutions are slower to process
  and add little.
- Keep exposure and focus steady if your camera app allows locking them.

## What the system cannot detect

- Regions seen in only one of the two videos cannot be compared. They are
  reported as unobserved, not as change.
- Objects hidden behind furniture or inside closed containers in either video
  are likewise unobserved.
- Very small objects, or objects with little texture, may be missed. Cover
  important areas from several angles to reduce this.

## Checklist

1. Pick a reference view and start there.
2. Walk the area slowly, covering key regions from two angles.
3. Return to the reference view and stop.
4. Repeat later, with the same route and orientation.
