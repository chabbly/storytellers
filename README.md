# Storytellers

Software for the _Storytellers_ installation at the NFSA in October 2024. Code
mostly by Ben Swift, but check the `git log` for the real story.

## Install

It's a [rye](https://rye.astral.sh) snafu, so `rye sync` will set you up. Other
ways work too, but... [y'know](https://xkcd.com/1987/).

Note: currently set up for use on an Apple Silicon Mac, but should work with an
NVIDIA card too (in fact, would probably work better). You'll just need to grep
through the codebase and change all the `"mps"`s to `"cuda"`.

Note: video files aren't committed to this repo, because (a) we don't have the
licence to put them on GitHub and (b) they'd bloat the repo anyway. So to use
this, create your own video frames---see [the assets readme](/assets/README.md)
for more info.

## Use

1. ensure you've got your image frames in `assets/<video_name>/`
2. modify the code in `src/storytellers/__init__.py` to point at your folder of
   image frames
3. set up your webcam (might need to change the index at the top of `image.py`
   to select the right webcam)
4. `rye run storytellers` and you're away

## TODO

- make it go brrrrrr (it's currently only ~2fps on my very beefy MBP)
- test with the actual videos
- sound
- move from using matplotlib (gross!) to something nicer for the display
- modulate the "generative-ness" setting over time
- we need to find a way to preserve the integrity of the original footage while maximising the integration and expressiveness of the user input. Some ideas for how to do this are:
  - segment the film around the user input and do generative fill instead of combining the user input and frame
  - layer a low opacity copy of the film over the generative frame + input image so that the generated image is showing through the original
  - pad the user input with a border of white to make it pop

## Licence

MIT
