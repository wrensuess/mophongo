"""Coverage for scene partitioning (`mophongo.scene.generate_scenes`).

Scenes decompose the global sparse-fit problem into independent blocks; the
critical property is that the partitioning groups genuinely-overlapping
templates together while keeping well-separated templates apart, and that a
source flagged as a star is excluded from the bright/astrometry set (A5 in
docs/test_suite_cleanup_plan.md).
"""
import numpy as np

from mophongo.scene import generate_scenes
from mophongo.templates import Template

# A small, symmetric footprint reused for all hand-placed templates below.
_STAMP = np.array(
    [
        [0.1, 0.2, 0.1],
        [0.2, 0.4, 0.2],
        [0.1, 0.2, 0.1],
    ]
)


def _place(img, position, label, flux, image, is_star=False):
    tmpl = Template(img, position, (3, 3), label=label)
    tmpl.data[:] = _STAMP
    tmpl.is_star = is_star
    image[tmpl.slices_original] += flux * tmpl.data[tmpl.slices_cutout]
    return tmpl


def test_generate_scenes_partitions_overlap_vs_isolated():
    """Overlapping templates land in one scene; a distant one lands in another."""
    img = np.zeros((30, 30))
    weights = np.ones_like(img)
    image = np.zeros_like(img)

    # t1/t2 overlap by one column -> should be coupled into the same scene.
    t1 = _place(img, (5, 5), 1, 50.0, image)
    t2 = _place(img, (5, 6), 2, 50.0, image)
    # t3 sits far away with no pixel overlap -> should form its own scene.
    t3 = _place(img, (25, 25), 3, 50.0, image)

    templates = [t1, t2, t3]
    scenes, labels = generate_scenes(
        templates, image, weights, minimum_bright=1, snr_thresh_astrom=5.0
    )

    # t1 and t2 share a scene label...
    assert labels[0] == labels[1]
    # ...while t3 is partitioned separately.
    assert labels[2] != labels[0]

    scene_by_id = {t.id: lbl for t, lbl in zip(templates, labels)}
    assert scene_by_id[1] == scene_by_id[2]
    assert scene_by_id[3] != scene_by_id[1]

    # Every template must appear in exactly one returned Scene.
    all_ids = sorted(tid for s in scenes for tid in (t.id for t in s.templates))
    assert all_ids == [1, 2, 3]


def test_generate_scenes_excludes_star_from_bright_mask():
    """A bright source flagged `is_star` is excluded from the bright/astrometry set."""
    img = np.zeros((30, 30))
    weights = np.ones_like(img)
    image = np.zeros_like(img)

    t1 = _place(img, (5, 5), 1, 50.0, image)
    t2 = _place(img, (5, 6), 2, 50.0, image)
    # Same flux/SNR as t1/t2, but flagged as a star and kept well separated
    # (max_merge_radius below keeps it from being folded back into scene 1
    # purely because it would otherwise look "too small" to stand alone).
    t3 = _place(img, (25, 25), 3, 50.0, image, is_star=True)

    templates = [t1, t2, t3]
    scenes, labels = generate_scenes(
        templates,
        image,
        weights,
        minimum_bright=1,
        snr_thresh_astrom=5.0,
        max_merge_radius=5.0,
    )

    scene_of_star = next(s for s in scenes if t3.id in (t.id for t in s.templates))
    idx_in_scene = [t.id for t in scene_of_star.templates].index(t3.id)

    # t3 has the same SNR as t1/t2 but must be excluded from the bright mask
    # purely because it is a star.
    assert scene_of_star.is_bright[idx_in_scene] == False  # noqa: E712

    scene_of_t1 = next(s for s in scenes if t1.id in (t.id for t in s.templates))
    idx_t1 = [t.id for t in scene_of_t1.templates].index(t1.id)
    assert scene_of_t1.is_bright[idx_t1] == True  # noqa: E712
