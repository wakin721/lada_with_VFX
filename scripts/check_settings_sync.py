# SPDX-License-Identifier: AGPL-3.0
"""Exercise real GTK watching/export controls without modifying user settings."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gtk-runtime', type=Path)
    args = parser.parse_args()
    dll_handle = None
    if args.gtk_runtime:
        runtime = args.gtk_runtime.resolve()
        os.environ['PATH'] = str(runtime / 'bin') + os.pathsep + os.environ.get('PATH', '')
        os.environ['GI_TYPELIB_PATH'] = str(runtime / 'lib/girepository-1.0')
        os.environ['GST_PLUGIN_PATH'] = str(runtime / 'lib/gstreamer-1.0')
        os.environ['GST_PLUGIN_SYSTEM_PATH'] = ''
        if hasattr(os, 'add_dll_directory'):
            dll_handle = os.add_dll_directory(str(runtime / 'bin'))
    with tempfile.TemporaryDirectory(prefix='lada-settings-check-') as directory:
        os.environ['GST_REGISTRY'] = str(Path(directory) / 'registry.bin')
        import gi
        gi.require_version('Gtk', '4.0')
        gi.require_version('Adw', '1')
        gi.require_version('Gst', '1.0')
        gi.require_version('GstApp', '1.0')
        from gi.repository import Adw
        Adw.init()
        from lada.gui import utils  # Preserve the application's import order.
        from lada.gui.config.config import Config
        from lada.gui.config.config_sidebar import ConfigSidebar
        from lada.utils.vfx import VfxOptions
        config = Config(Adw.StyleManager.get_default())
        saved = []
        config.save = lambda: saved.append(config._as_dict())
        watch, export = ConfigSidebar(), ConfigSidebar()
        watch.config = config
        export.config = config
        watch.combo_row_vfx_scale.set_selected(3)
        watch.combo_row_vfx_denoise.set_selected(2)
        watch.combo_row_vfx_multiplier.set_selected(1)
        watch.combo_row_vfx_quality.set_selected(3)
        watch.combo_row_vfx_frame_quality.set_selected(2)
        expected = VfxOptions(4, 'MEDIUM', 2, 'ULTRA', 'HIGH')
        assert VfxOptions.from_config(config) == expected
        for _ in range(4):
            export.init_sidebar_from_config(config)
            watch.init_sidebar_from_config(config)
            assert VfxOptions.from_config(config) == expected
            assert export.combo_row_vfx_scale.get_selected() == 3
            assert export.combo_row_vfx_quality.get_selected() == 3
        loaded = Config(Adw.StyleManager.get_default())
        loaded.save = lambda: None
        loaded._from_dict({key: value for key, value in saved[-1].items() if key.startswith('vfx_')})
        assert VfxOptions.from_config(loaded) == expected
    print('PASS: watching/export controls and persisted VFX settings survive tab refreshes')


if __name__ == '__main__':
    main()
