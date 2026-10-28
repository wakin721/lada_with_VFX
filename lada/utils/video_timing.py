# SPDX-License-Identifier: AGPL-3.0
"""Interpret ffprobe rates without mistaking a timestamp clock for video FPS."""
from fractions import Fraction


def select_frame_rate(stream, duration=None):
    def rate(value):
        try:
            result = Fraction(str(value))
            return result if result > 0 else None
        except (ValueError, ZeroDivisionError, TypeError):
            return None

    nominal = rate(stream.get('r_frame_rate'))
    average = rate(stream.get('avg_frame_rate'))
    # For VFR, r_frame_rate can be the least common timestamp denominator,
    # e.g. 90000/1 for a 2.39 FPS screen recording. PTS still govern playback.
    if average is not None and (nominal is None or abs(nominal / average - 1) > Fraction(1, 100)):
        return average
    if nominal is not None and nominal <= 1000:
        return nominal
    if duration and stream.get('nb_frames'):
        estimate = rate(stream['nb_frames']) / Fraction(str(duration))
        if estimate > 0:
            return estimate
    return average or Fraction(30)


def has_variable_frame_rate(stream, chosen_rate):
    try:
        return abs(Fraction(stream.get('r_frame_rate', '0/1')) / chosen_rate - 1) > Fraction(1, 100)
    except (ValueError, TypeError, ZeroDivisionError):
        return True
