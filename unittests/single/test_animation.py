from __future__ import annotations

from elisa.single_system.graphic.animation import Animation
from unittests import utils as testutils
from unittests.utils import ElisaTestCase, prepare_single_system


class AnimationTestCase(ElisaTestCase):
    """Unit tests for `elisa.single_system.graphic.animation.Animation`.

    The tests exercise the animation helper in a lightweight way by
    requesting a single-frame animation so the plotting backend work is
    executed but the runtime stays fast and deterministic.
    """

    def test_rotational_motion_runs_minimal(self):
        """Run a minimal (single-frame) animation and ensure it completes.

        :returns: None
        :rtype: None
        """
        single = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS["spherical"])
        single.init()

        anim = Animation(single)

        import matplotlib.animation as mpa
        import matplotlib.pyplot as plt

        # Prevent the interactive window from appearing during the test by
        # temporarily replacing plt.show with a no-op. Also silence the
        # Animation deletion warning by stubbing Animation.__del__ during
        # the call. Ensure figures are closed afterward so the test
        # environment stays clean.
        old_show = plt.show
        old_del = getattr(mpa.Animation, "__del__", None)
        plt.show = lambda *a, **k: None
        mpa.Animation.__del__ = lambda self: None
        try:
            # noinspection PyNoneFunctionAssignment
            result = anim.rotational_motion(
                start_phase=0.0,
                stop_phase=0.02,
                phase_step=0.02,
                scale="linear",
                colormap=None,
                savepath=None,
                plot_axis=False,
                subtract_equilibrium=False,
                edges=False,
            )
            self.assertIsNone(result)
        finally:
            plt.show = old_show
            # restore original __del__ if present
            if old_del is None:
                delattr(mpa.Animation, "__del__")
            else:
                mpa.Animation.__del__ = old_del
            plt.close("all")

    def test_rotational_motion_invalid_range_raises(self):
        """Providing start_phase > stop_phase must raise ValueError.

        :returns: None
        :rtype: None
        """
        single = prepare_single_system(testutils.SINGLE_SYSTEM_PARAMS["spherical"])
        single.init()

        anim = Animation(single)

        with self.assertRaises(ValueError):
            anim.rotational_motion(start_phase=0.5, stop_phase=-0.5, phase_step=0.01)
